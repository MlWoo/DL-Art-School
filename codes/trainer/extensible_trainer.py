import copy
import os
from math import sqrt
from pathlib import Path
from time import time

import maybe_bnb as mbnb
import torch
import torch.nn as nn
import torchvision.utils as utils
import trainer.lr_scheduler as lr_scheduler
import trainer.networks as networks
from torch import distributed
from torch.nn.parallel import DataParallel
from trainer.base_trainer import BaseTrainer
from trainer.batch_size_optimizer import create_batch_size_optimizer
from trainer.experiments.experiments import get_experiment_for_name
from trainer.inject import create_injector
from trainer.loss_accumulator import InfStorageLossAccumulator

# from trainer.injectors.audio_injectors import normalize_mel
from trainer.steps import ConfigurableStep
from utils import logging
from utils.options import opt_get


# State is immutable to reduce complexity. Overwriting existing state keys is not supported.
class OverwrittenStateError(Exception):
    def __init__(self, k, keys):
        super().__init__(
            f"Attempted to overwrite state key: {k}.  The state should be considered "
            f"immutable and keys should not be overwritten. Current keys: {keys}"
        )


class ExtensibleTrainer(BaseTrainer):
    def __init__(self, opt, cached_networks={}):
        super(ExtensibleTrainer, self).__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]
        self.logger = logging.getLogger("base")
        # env is used as a global state to store things that subcomponents might need.
        self.env = {"device": self.device, "rank": self.rank, "opt": opt, "step": 0, "dist": opt["dist"]}
        if opt["path"]["models"] is not None:
            self.env["base_path"] = os.path.join(opt["path"]["models"])

        self.mega_batch_factor = 1
        if self.is_train:
            self.mega_batch_factor = train_opt["mega_batch_factor"]
            self.env["mega_batch_factor"] = self.mega_batch_factor
            self.batch_factor = self.mega_batch_factor
            self.ema_rate = opt_get(train_opt, ["ema_rate"], 0.999)
            # It is advantageous for large networks to do this to save an extra copy of the model weights.
            # It does come at the cost of a round trip to CPU memory at every batch.
            self.do_emas = opt_get(train_opt, ["ema_enabled"], True)
            self.ema_on_cpu = opt_get(train_opt, ["ema_on_cpu"], False)
        self.checkpointing_cache = opt["checkpointing_enabled"]
        self.auto_recover = opt_get(opt, ["automatically_recover_nan_by_reverting_n_saves"], None)
        self.batch_size_optimizer = create_batch_size_optimizer(train_opt)
        self.auto_scale_grads = opt_get(opt, ["automatically_scale_grads_for_fanin"], False)
        self.auto_scale_basis = opt_get(opt, ["automatically_scale_base_layer_size"], 1024)

        self.netsG = {}
        self.netsD = {}
        for name, net in opt["networks"].items():
            # Trainable is a required parameter, but the default is simply true. Set it here.
            if "trainable" not in net.keys():
                net["trainable"] = True

            if "save_freeze" not in net.keys():
                net["save_freeze"] = True

            if name in cached_networks.keys():
                new_net = cached_networks[name]
            else:
                new_net = None
            if net["type"] == "generator":
                if new_net is None:
                    new_net = networks.create_model(opt, net, self.netsG).to(self.device)
                self.netsG[name] = new_net
            elif net["type"] == "discriminator":
                if new_net is None:
                    new_net = networks.create_model(opt, net, self.netsD).to(self.device)
                self.netsD[name] = new_net
            else:
                raise NotImplementedError("Can only handle generators and discriminators")

            if not net["trainable"]:
                new_net.eval()

            if net["wandb_debug"] and self.rank <= 0:
                import wandb

                wandb.watch(new_net, log="all", log_freq=3)

        # Initialize the train/eval steps
        self.step_names = []
        self.steps = []
        for step_name, step in opt["steps"].items():
            step = ConfigurableStep(step_name, step, self.env)
            self.step_names.append(
                step_name
            )  # This could be an OrderedDict, but it's a PITA to integrate with AMP below.
            self.steps.append(step)

        # step.define_optimizers() relies on the networks being placed in the env, so put them there. Even though
        # they aren't wrapped yet.
        self.env["generators"] = self.netsG
        self.env["discriminators"] = self.netsD

        # Define the optimizers from the steps
        for s in self.steps:
            s.define_optimizers()
            self.optimizers.extend(s.get_optimizers())

        if self.is_train:
            # Find the optimizers that are using the default scheduler, then build them.
            def_opt = []
            for s in self.steps:
                def_opt.extend(s.get_optimizers_with_default_scheduler())
            self.schedulers = lr_scheduler.get_scheduler_for_name(train_opt["default_lr_scheme"], def_opt, train_opt)

            # Set the starting step count for the scheduler.
            for sched in self.schedulers:
                sched.last_epoch = opt["current_step"]
        else:
            self.schedulers = []

        # Wrap networks in distributed shells.
        dnets = []
        all_networks = [g for g in self.netsG.values()] + [d for d in self.netsD.values()]
        for anet in all_networks:
            has_any_trainable_params = False
            for p in anet.parameters():
                if not hasattr(p, "DO_NOT_TRAIN"):
                    has_any_trainable_params = True
                    break
            if has_any_trainable_params and opt["dist"]:
                if opt["dist_backend"] == "apex":
                    # Use Apex to enable delay_allreduce, which is compatible with gradient checkpointing.
                    from apex.parallel import DistributedDataParallel

                    dnet = DistributedDataParallel(anet, delay_allreduce=True)
                elif opt["dist_backend"] == "ddp":
                    from torch.nn.parallel.distributed import DistributedDataParallel

                    # Do NOT be tempted to put find_unused_parameters=True here. It will not work when checkpointing is
                    # used and in a few other cases. But you can try it if you really want.
                    dnet = DistributedDataParallel(
                        anet,
                        device_ids=[torch.cuda.current_device()],
                        output_device=torch.cuda.current_device(),
                        find_unused_parameters=opt_get(opt, ["ddp_find_unused_parameters"], False),
                        gradient_as_bucket_view=True,
                    )
                    # DDP graphs cannot be used with gradient checkpointing unless you use find_unused_parameters=True,
                    # which does not work with this trainer (as stated above). However, if the graph is not subject
                    # to control flow alterations, you can set this option to allow gradient checkpointing. Beware that
                    # if you are wrong about control flow, DDP will not train all your model parameters! User beware!
                    if opt_get(opt, ["ddp_static_graph"], False):
                        dnet._set_static_graph()
                elif opt["dist_backend"] == "fsdp":
                    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

                    dnet = FullyShardedDataParallel(
                        anet, device_ids=[torch.cuda.current_device()], sharding_strategy=ShardingStrategy.FULL_SHARD
                    )
                else:
                    dnet = anet.to(torch.cuda.current_device())
            else:
                dnet = DataParallel(anet, device_ids=[torch.cuda.current_device()])
            if self.is_train:
                dnet.train()
            else:
                dnet.eval()
            dnets.append(dnet)

        # Backpush the wrapped networks into the network dicts. Also build the EMA parameters.
        self.networks = {}
        self.emas = {}
        found = 0
        for dnet in dnets:
            for net_dict in [self.netsD, self.netsG]:
                for k, v in net_dict.items():
                    if v == dnet.module:
                        net_dict[k] = dnet
                        self.networks[k] = dnet
                        if self.is_train and self.do_emas:
                            self.emas[k] = copy.deepcopy(v)
                            if self.ema_on_cpu:
                                self.emas[k] = self.emas[k].cpu()
                            if hasattr(v, "provide_ema"):
                                v.provide_ema(self.emas[k])
                        found += 1
        assert found == len(self.netsG) + len(self.netsD)

        # Replace the env networks with the wrapped networks
        self.env["generators"] = self.netsG
        self.env["discriminators"] = self.netsD
        self.env["emas"] = self.emas
        self.print_network()  # print network
        self.load()  # load networks from save states as needed

        # Load experiments
        self.experiments = []
        if "experiments" in opt.keys():
            self.experiments = [get_experiment_for_name(e) for e in opt["experiments"]]

        # Setting this to false triggers SRGAN to call the models update_model() function on the first iteration.
        self.updated = True

    def feed_run_info(self, epoch, step, will_log=False, will_visual=False, **kwargs):
        # Some generators can do their own metric logging.
        for net_name, net in self.networks.items():
            if hasattr(net.module, "feed_run_info"):
                lr = self.get_current_learning_rate()[0]
                net.module.feed_run_info(
                    epoch=epoch, step=step, will_log=will_log, will_visual=will_visual, lr=lr, **kwargs
                )

    def apply_compile(self):
        for net in self.networks.values():
            if hasattr(net.module, "apply_compile"):
                net.module.apply_compile()

    def apply_auto_batch(self, auto_batch: bool = False):
        for net in self.networks.values():
            if hasattr(net.module, "apply_auto_batch"):
                net.module.apply_auto_batch(auto_batch)

    def create_dummy_input(self, batch_size, bucket_boundary):
        pass

    def feed_data(self, data, step, reduce_batch_factor=None, need_GT=True, perform_micro_batching=True, profile=False):
        self.env["step"] = step
        self.batch_factor = (
            self.mega_batch_factor if reduce_batch_factor is None else self.mega_batch_factor * reduce_batch_factor
        )
        self.opt["checkpointing_enabled"] = self.checkpointing_cache
        # The batch factor can be adjusted on a period to allow known high-memory steps to fit in GPU memory.
        if (
            "train" in self.opt.keys()
            and "mod_batch_factor" in self.opt["train"].keys()
            and self.env["step"] % self.opt["train"]["mod_batch_factor_every"] == 0
        ):
            self.batch_factor = self.opt["train"]["mod_batch_factor"]
            if self.opt["train"]["mod_batch_factor_also_disable_checkpointing"]:
                self.opt["checkpointing_enabled"] = False
        if profile:
            start = time()
        self.eval_state = {}
        for o in self.optimizers:
            o.zero_grad()
        # torch.cuda.empty_cache()

        if profile:
            end = time()
            print(f"feed data optimizier preparing: {end - start}")
            start = end

        sort_key = opt_get(self.opt, ["train", "sort_key"], None)
        if sort_key is not None:
            sort_indices = torch.sort(data[sort_key], descending=True).indices
        else:
            sort_indices = None

        batch_factor = self.batch_factor if perform_micro_batching else 1
        self.dstate = {}

        from itertools import islice

        def chunk(it, size):
            it = iter(it)
            return iter(lambda: tuple(islice(it, size)), ())

        for k, v in data.items():
            if sort_indices is not None:
                if isinstance(v, list):
                    v = [v[i] for i in sort_indices]
                else:
                    v = v[sort_indices]
            if isinstance(v, torch.Tensor):
                self.dstate[k] = [t.to(self.device) for t in torch.chunk(v, chunks=batch_factor, dim=0)]
            else:
                self.dstate[k] = list(chunk(v, len(v) // batch_factor))

        if profile:
            end = time()
            print(f"feed data chunking preparing: {end - start}")
            start = end

        if opt_get(self.opt, ["train", "auto_collate"], False):
            for k, v in self.dstate.items():
                if f"{k}_lengths" in self.dstate.keys():
                    for c in range(len(v)):
                        maxlen = self.dstate[f"{k}_lengths"][c].max()
                        if len(v[c].shape) == 2:
                            self.dstate[k][c] = self.dstate[k][c][:, :maxlen]
                        elif len(v[c].shape) == 3:
                            self.dstate[k][c] = self.dstate[k][c][:, :, :maxlen]
                        elif len(v[c].shape) == 4:
                            self.dstate[k][c] = self.dstate[k][c][:, :, :, :maxlen]

    def optimize_parameters(self, it, optimize=True, return_grad_norms=False, raise_oom=None):
        # return_grad_norms is also the flag to record
        grad_norms = {}
        # Some models need to make parametric adjustments per-step. Do that here.
        for net in self.networks.values():
            if hasattr(net.module, "update_for_step"):
                net.module.update_for_step(it, os.path.join(self.opt["path"]["models"], ".."))

        # Iterate through the steps, performing them one at a time.
        state = self.dstate
        for step_num, step in enumerate(self.steps):
            train_step = True
            # 'every' is used to denote steps that should only occur at a certain integer factor rate.
            # e.g. '2' occurs every 2 steps.
            # Note that the injection points for the step might still be required, so address this by setting
            # train_step=False
            if "every" in step.step_opt.keys() and it % step.step_opt["every"] != 0:
                train_step = False
            # Steps can opt out of early (or late) training, make sure that happens here.
            if (
                "after" in step.step_opt.keys()
                and it < step.step_opt["after"]
                or "before" in step.step_opt.keys()
                and it > step.step_opt["before"]
            ):
                continue
            # Steps can choose to not execute if a state key is missing.
            if "requires" in step.step_opt.keys():
                requirements_met = True
                for requirement in step.step_opt["requires"]:
                    if requirement not in state.keys():
                        requirements_met = False
                if not requirements_met:
                    continue

            if train_step:
                # Only set requires_grad=True for the network being trained.
                nets_to_train = step.get_networks_trained()
                enabled = 0
                for name, net in self.networks.items():
                    net_enabled = name in nets_to_train
                    if net_enabled:
                        enabled += 1
                    # Networks can opt out of training before a certain iteration by
                    # declaring 'after' in their definition.
                    if "after" in self.opt["networks"][name].keys() and it < self.opt["networks"][name]["after"]:
                        net_enabled = False
                    if name in opt_get(step.step_opt, ["reuse_net"], []):
                        reuse_net_enabled = True
                    else:
                        reuse_net_enabled = False
                    for p in net.parameters():
                        do_not_train_flag = hasattr(p, "DO_NOT_TRAIN") or (
                            hasattr(p, "DO_NOT_TRAIN_UNTIL") and it < p.DO_NOT_TRAIN_UNTIL
                        )
                        if p.dtype != torch.int64 and p.dtype != torch.bool and not do_not_train_flag:
                            p.requires_grad = net_enabled or reuse_net_enabled
                        else:
                            p.requires_grad = False
                assert enabled == len(nets_to_train)

                # Update experiments
                [
                    e.before_step(self.opt, self.step_names[step_num], self.env, nets_to_train, state)
                    for e in self.experiments
                ]
                for o in step.get_optimizers():
                    o.zero_grad()

            # Now do a forward and backward pass for each gradient accumulation step.
            new_states = {}
            self.batch_size_optimizer.focus(net)
            is_oom = False
            for m in range(self.batch_factor):
                is_oom, ns = step.do_forward_backward(
                    state,
                    m,
                    step_num,
                    train=train_step,
                    no_ddp_sync=(m + 1 < self.batch_factor),
                    reuse_out=opt_get(step.step_opt, ["reuse_out"], [None]),
                    raise_oom=raise_oom,
                )
                if is_oom:
                    break
                if ns is None:
                    continue
                # Call into post-backward hooks.
                for name, net in self.networks.items():
                    if hasattr(net.module, "after_backward"):
                        net.module.after_backward(it)

                for k, v in ns.items():
                    if k not in new_states.keys():
                        new_states[k] = [v]
                    else:
                        new_states[k].append(v)
            if is_oom:
                return is_oom, grad_norms
            # Push the detached new state tensors into the state map for use with the next step.
            for k, v in new_states.items():
                if k in state.keys():
                    raise OverwrittenStateError(k, list(state.keys()))
                state[k] = v
            # (Maybe) perform a step.
            if train_step and optimize and self.batch_size_optimizer.should_step(it):
                # Unscale gradients within the step. (This is admittedly pretty messy but the API contract between
                # step & ET is pretty much broken at this point)
                # This is needed to accurately log the grad norms.
                for opt in step.optimizers:
                    from torch.cuda.amp.grad_scaler import OptState

                    if (
                        step.scaler.is_enabled()
                        and step.scaler._per_optimizer_states[id(opt)]["stage"] is not OptState.UNSCALED
                        and not step.nan_loss_counter
                    ):
                        step.scaler.unscale_(opt)

                # Call into pre-step hooks.
                for name, net in self.networks.items():
                    if hasattr(net.module, "before_step"):
                        net.module.before_step(it)

                if self.auto_scale_grads:
                    asb = sqrt(self.auto_scale_basis)
                    for net in self.networks.values():
                        for mod in net.modules():
                            fan_in = -1
                            if isinstance(mod, mbnb.nn.Linear):
                                fan_in = mod.weight.data.shape[1]
                            elif isinstance(mod, nn.Conv1d):
                                fan_in = mod.weight.data.shape[0]
                            elif isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Conv3d):
                                assert "Not yet implemented!"
                            if fan_in != -1:
                                p = mod.weight
                                if hasattr(p, "grad") and p.grad is not None:
                                    p.grad = p.grad * asb / sqrt(fan_in)

                if return_grad_norms and train_step:
                    for name in nets_to_train:
                        model = self.networks[name]
                        if hasattr(model.module, "get_grad_norm_parameter_groups"):
                            pgroups = {
                                f"{name}_{k}": v for k, v in model.module.get_grad_norm_parameter_groups().items()
                            }
                        else:
                            pgroups = {f"{name}_all_parameters": list(model.parameters())}
                    for name in pgroups.keys():
                        stacked_grads = []
                        for p in pgroups[name]:
                            if hasattr(p, "grad") and p.grad is not None:
                                stacked_grads.append(torch.norm(p.grad.detach(), 2))
                        if not stacked_grads:
                            continue
                        grad_norms[name] = torch.norm(torch.stack(stacked_grads), 2)
                        if distributed.is_available() and distributed.is_initialized():
                            # Gather the metric from all devices if in a distributed setting.
                            distributed.all_reduce(grad_norms[name], op=distributed.ReduceOp.SUM)
                            grad_norms[name] /= distributed.get_world_size()
                        grad_norms[name] = grad_norms[name].cpu()

                self.consume_gradients(state, step, it)

        # Record visual outputs for usage in debugging and testing.
        if (
            "visuals" in self.opt["logger"].keys()
            and self.rank <= 0
            and it % self.opt["logger"]["visual_debug_rate"] == 0
            and raise_oom is None
        ):

            def fix_image(img):
                # if opt_get(self.opt, ["logger", "is_mel_spectrogram"], False):
                #     if img.min() < -2:
                #         img = normalize_mel(img)
                #     img = img.unsqueeze(dim=1)
                if img.shape[1] > 3:
                    img = img[:, :3, :, :]
                if opt_get(self.opt, ["logger", "reverse_n1_to_1"], False):
                    img = (img + 1) / 2
                # if opt_get(self.opt, ["logger", "reverse_imagenet_norm"], False):
                #     img = denormalize(img)
                # if opt_get(self.opt, ['logger', 'audio'], False):
                #    img = plot_wav(img)
                return img

            sample_save_path = os.path.join(self.opt["path"]["models"], "..", "visual_dbg")
            for v in self.opt["logger"]["visuals"]:
                if v not in state.keys():
                    continue  # This can happen for several reasons (ex: 'after' defs), just ignore it.
                for i, dbgv in enumerate(state[v]):
                    visuals_batch_size = opt_get(self.opt["logger"], ["visuals_batch_size"], 64)
                    dbgv = dbgv[:visuals_batch_size]
                    if "recurrent_visual_indices" in self.opt["logger"].keys() and len(dbgv.shape) == 5:
                        for rvi in self.opt["logger"]["recurrent_visual_indices"]:
                            rdbgv = fix_image(dbgv[:, rvi])
                            os.makedirs(os.path.join(sample_save_path, v), exist_ok=True)
                            utils.save_image(
                                rdbgv.float(), os.path.join(sample_save_path, v, "%05i_%02i_%02i.png" % (it, rvi, i))
                            )
                    else:
                        if len(dbgv.shape) == 3:
                            dbgv = dbgv.unsqueeze(dim=1)
                        dbgv = fix_image(dbgv)
                        os.makedirs(os.path.join(sample_save_path, v), exist_ok=True)
                        utils.save_image(dbgv.float(), os.path.join(sample_save_path, v, "%05i_%02i.png" % (it, i)))

            # Some models have their own specific visual debug routines.
            for net_name, net in self.networks.items():
                if hasattr(net.module, "visual_dbg"):
                    model_vdbg_dir = os.path.join(sample_save_path, net_name)
                    os.makedirs(model_vdbg_dir, exist_ok=True)
                    net.module.visual_dbg(it, model_vdbg_dir)

        return is_oom, grad_norms

    def consume_gradients(self, state, step, it):
        # if self.rank <= 0:
        #     start = time()
        [e.before_optimize(state) for e in self.experiments]
        self.restore_optimizers()
        step.do_step(it)
        self.stash_optimizers()

        # Call into custom step hooks as well as update EMA params.
        for name, net in self.networks.items():
            if hasattr(net.module, "after_step"):
                net.module.after_step(it)
            if self.do_emas:
                # When the EMA is on the CPU, only update every 10 steps to save processing time.
                if self.ema_on_cpu and it % 10 != 0:
                    continue
                ema_params = self.emas[name].parameters()
                net_params = net.parameters()
                for ep, np in zip(ema_params, net_params):
                    ema_rate = self.ema_rate
                    new_rate = 1 - ema_rate
                    if self.ema_on_cpu:
                        np = np.cpu()
                        ema_rate = ema_rate**10  # Because it only happens every 10 steps.
                        mid = (1 - (ema_rate + new_rate)) / 2
                        ema_rate += mid
                        new_rate += mid
                    ep.detach().mul_(ema_rate).add_(np, alpha=1 - ema_rate)
        [e.after_optimize(state) for e in self.experiments]
        # if self.rank <= 0:
        #     end = time()
        #     print(f"consume_gradients: {end - start}")

    def test(self, raise_oom=None):
        for net in self.netsG.values():
            net.eval()

        is_oom = False
        accum_metrics = InfStorageLossAccumulator()
        with torch.no_grad():
            # This can happen one of two ways: Either a 'validation injector' is provided, in which case we run that.
            # Or, we run the entire chain of steps in "train" mode and use eval.output_state.
            if "injectors" in self.opt["eval"].keys():
                state = {}
                for inj in self.opt["eval"]["injectors"].values():
                    # Need to move from mega_batch mode to batch mode (remove chunks)
                    for k, v in self.dstate.items():
                        state[k] = v[0]
                    inj = create_injector(inj, self.env)
                    sole_state = inj(state)
                    for n in inj.output:
                        # Doesn't really work for training setups where multiple of the same injector are used.
                        accum_metrics.add_loss(n, sole_state[n])
                    state.update(state)
            else:
                # Iterate through the steps, performing them one at a time.
                state = self.dstate
                for step_num, s in enumerate(self.steps):
                    is_oom, ns = s.do_forward_backward(
                        state, 0, step_num, train=False, loss_accumulator=accum_metrics, raise_oom=raise_oom
                    )
                    if not is_oom:
                        for k, v in ns.items():
                            state[k] = [v]
                    else:
                        break

            self.eval_state = {}
            for k, v in state.items():
                if isinstance(v, list):
                    self.eval_state[k] = [s.detach().cpu() if isinstance(s, torch.Tensor) else s for s in v]
                else:
                    self.eval_state[k] = [v.detach().cpu() if isinstance(v, torch.Tensor) else v]

        for net in self.netsG.values():
            net.train()
        return is_oom, accum_metrics

    # Fetches a summary of the log.
    def get_current_log(self, step):
        log = {}
        for s in self.steps:
            log.update(s.get_metrics())

        for e in self.experiments:
            log.update(e.get_log_data())

        # Some generators can do their own metric logging.
        for net_name, net in self.networks.items():
            if hasattr(net.module, "get_debug_values"):
                log.update(net.module.get_debug_values(step, net_name))

        # Log learning rate (from first param group) too.
        for o in self.optimizers:
            for group_name, pg in zip(o._groups_name, o.param_groups):
                log["learning_rate_%s_%s" % (o._config["network"], group_name)] = pg["lr"]

        # The batch size optimizer also outputs loggable data.
        log.update(self.batch_size_optimizer.get_statistics())

        # In distributed mode, get agreement on all single tensors.
        if distributed.is_available() and distributed.is_initialized():
            for k, v in log.items():
                if not isinstance(v, torch.Tensor):
                    continue
                if len(v.shape) != 1 or v.dtype != torch.float:
                    continue
                distributed.all_reduce(v, op=distributed.ReduceOp.SUM)
                log[k] = v / distributed.get_world_size()
        return log

    def get_current_visuals(self, need_GT=True):
        # Conforms to an archaic format from MMSR.
        res = {"rlt": self.eval_state[self.opt["eval"]["output_state"]][0].float().cpu()}
        if "hq" in self.eval_state.keys():
            res["hq"] = (self.eval_state["hq"][0].float().cpu(),)
        return res

    def print_network(self):
        for name, net in self.networks.items():
            s, n = self.get_network_description(net)
            net_struc_str = "{}".format(net.__class__.__name__)
            if self.rank <= 0:
                self.logger.info("Network {} structure: {}, with parameters: {:,d}".format(name, net_struc_str, n))
                self.logger.info(s)

    def load(self):
        for netdict in [self.netsG, self.netsD]:
            for name, net in netdict.items():
                load_path = self.opt["path"]["pretrain_model_%s" % (name,)]
                if load_path is None:
                    return
                if self.rank <= 0:
                    self.logger.info("Loading model for [%s]" % (load_path,))
                self.load_network(
                    load_path,
                    net,
                    self.opt["path"]["strict_load"],
                    opt_get(self.opt, ["path", f"pretrain_base_path_{name}"]),
                )
                load_path_ema = load_path.replace(".pth", "_ema.pth")
                if self.is_train and self.do_emas:
                    ema_model = self.emas[name]
                    if os.path.exists(load_path_ema):
                        self.load_network(
                            load_path_ema,
                            ema_model,
                            self.opt["path"]["strict_load"],
                            opt_get(self.opt, ["path", f"pretrain_base_path_{name}"]),
                        )
                    else:
                        print("WARNING! Unable to find EMA network! Starting a new EMA from given model parameters.")
                        self.emas[name] = copy.deepcopy(net)
                    if self.ema_on_cpu:
                        self.emas[name] = self.emas[name].cpu()
                if hasattr(net.module, "network_loaded"):
                    net.module.network_loaded()

    def limit_number_of_checkpoints_and_states(self, models_number: int = 2, state_number: int = 2) -> None:
        for network_name in self.networks.keys():
            models_path = Path(self.opt["path"]["models"]).parent / "models"
            states_path = Path(self.opt["path"]["models"]).parent / "training_state"
            files_pth, files_ema_pth, files_state = [], [], []

            if models_number > 0:
                files_pth = sorted(
                    models_path.glob(f"*_{network_name}.pth"),
                    reverse=True,
                    key=lambda p: int(p.stem.split("_")[0]),
                )
                files_ema_pth = sorted(
                    models_path.glob(f"*_{network_name}_ema.pth"),
                    reverse=True,
                    key=lambda p: int(p.stem.split("_")[0]),
                )

            if not self.opt["logger"]["disable_state_saving"] and state_number > 0:
                files_state = sorted(states_path.glob("*.state"), reverse=True, key=lambda p: int(p.stem))

            files_to_keep = (
                files_pth[: models_number - 1] + files_ema_pth[: models_number - 1] + files_state[: state_number - 1]
            )

            for file_path in files_pth + files_ema_pth + files_state:
                if file_path not in files_to_keep:
                    print(f"Removing: {file_path}")
                    open(file_path, "w").close()
                    os.remove(file_path)

    def save(self, iter_step):
        for name, net in self.networks.items():
            # Don't save non-trainable networks.
            if self.opt["networks"][name]["trainable"]:
                self.save_network(net, name, iter_step, save_freeze=self.opt["networks"][name]["save_freeze"])
                if self.do_emas:
                    self.save_network(
                        self.emas[name], f"{name}_ema", iter_step, save_freeze=self.opt["networks"][name]["save_freeze"]
                    )

    def force_restore_swapout(self):
        # Legacy method. Do nothing.
        pass
