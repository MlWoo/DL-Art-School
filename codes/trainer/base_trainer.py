import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel.distributed import DistributedDataParallel
from utils.distributed import dist_utils, map_to_device, optimizer_to
from utils.io import torch_save
from utils.options import opt_get
from utils.path_ext import copy_files_to_server


class BaseTrainer:
    def __init__(self, opt):
        self.opt = opt
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        self.device = torch.cuda.current_device() if opt["gpu_ids"] else torch.device("cpu")
        self.amp_level = "O0" if opt["amp_opt_level"] is None else opt["amp_opt_level"]
        self.is_train = opt["is_train"]
        self.opt_in_cpu = opt_get(opt, ["keep_optimizer_states_on_cpu"], False)
        self.schedulers = []
        self.optimizers = []
        self.disc_optimizers = []
        self.save_history = {}

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler"""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.last_epoch = cur_iter
            scheduler.step()
        # set up warm-up learning rate
        if False and cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group["lr"] for param_group in self.optimizers[0].param_groups]

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))

    def save_network(self, network, network_label, iter_label, save_freeze=True):
        save_filename = "{}_{}.pth".format(iter_label, network_label)
        save_path = os.path.join(self.opt["path"]["models"], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module

        def is_trainable(param_name):
            key_chain = param_name.split(".")
            param = network
            for key in key_chain:
                param = getattr(param, key)
            if param.requires_grad and param.grad is not None:
                return True
            else:
                return False

        raw_state_dict = network.state_dict()
        state_dict = {}
        for key, param in raw_state_dict.items():
            if is_trainable(key) or save_freeze:
                state_dict[key] = param.cpu()
        torch_save(state_dict, save_path, async_file=True)

        if network_label not in self.save_history.keys():
            self.save_history[network_label] = []
        self.save_history[network_label].append(save_path)

        # Also save to the 'alt_path' which is useful for caching to Google Drive in colab, for example.
        if "alt_path" in self.opt["path"].keys():
            torch.save(state_dict, os.path.join(self.opt["path"]["alt_path"], save_filename))
        if self.opt["colab_mode"]:
            copy_files_to_server(
                self.opt["ssh_server"],
                self.opt["ssh_username"],
                self.opt["ssh_password"],
                save_path,
                os.path.join(self.opt["remote_path"], "models", save_filename),
            )
        return save_path

    def load_network(self, load_path, network, strict=True, pretrain_base_path=None):
        # Sometimes networks are passed in as DDP modules, we want the raw parameters.
        if hasattr(network, "module"):
            network = network.module
        load_net = torch.load(load_path, map_location=dist_utils.map_cuda_to_correct_device, weights_only=True)

        # Support loading torch.save()s for whole models as well as just state_dicts.
        if "state_dict" in load_net:
            load_net = load_net["state_dict"]
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'

        if pretrain_base_path is not None:
            t = load_net
            load_net = {}
            for k, v in t.items():
                if k.startswith(pretrain_base_path):
                    load_net[k[len(pretrain_base_path) :]] = v

        for k, v in load_net.items():
            if k.startswith("module."):
                load_net_clean[k.replace("module.", "")] = v
            else:
                load_net_clean[k] = v
        current_model_dict = network.state_dict()

        new_state_dict = {}
        for k, v in zip(current_model_dict.keys(), current_model_dict.values()):
            if k in load_net_clean:
                if v.size() == load_net_clean[k].size():
                    new_state_dict[k] = load_net_clean[k]
                else:
                    self.logger.info(f"Skipping {k} because size mismatch: {v.size()} != {load_net_clean[k].size()}")
                    new_state_dict[k] = current_model_dict[k]
            else:
                new_state_dict[k] = current_model_dict[k]

        network.load_state_dict(new_state_dict, strict=strict)

    def consolidate_state(self):
        for o in self.optimizers:
            if isinstance(o, ZeroRedundancyOptimizer):
                o.consolidate_state_dict(to=0)

    def save_training_state(self, state):
        """Save training state during training, which will be used for resuming"""
        state.update({"schedulers": [], "optimizers": []})
        for s in self.schedulers:
            state["schedulers"].append(s.state_dict())
        for o in self.optimizers:
            state["optimizers"].append(o.state_dict())
        if "amp_opt_level" in self.opt.keys():
            from apex import amp

            state["amp"] = amp.state_dict()
        save_filename = "{}.state".format(opt_get(state, ["iter"], "no_step_provided"))
        save_path = os.path.join(self.opt["path"]["training_state"], save_filename)
        torch_save(map_to_device(state, "cpu"), save_path, async_file=True)
        if "__state__" not in self.save_history.keys():
            self.save_history["__state__"] = []
        self.save_history["__state__"].append(save_path)

        # Also save to the 'alt_path' which is useful for caching to Google Drive in colab, for example.
        if "alt_path" in self.opt["path"].keys():
            torch.save(state, os.path.join(self.opt["path"]["alt_path"], "latest.state"))
        if self.opt["colab_mode"]:
            copy_files_to_server(
                self.opt["ssh_server"],
                self.opt["ssh_username"],
                self.opt["ssh_password"],
                save_path,
                os.path.join(self.opt["remote_path"], "training_state", save_filename),
            )

    def stash_optimizers(self):
        """
        When enabled, puts all optimizer states in CPU memory, allowing forward and backward passes more memory
        headroom.
        """
        if not self.opt_in_cpu:
            return
        for opt in self.optimizers:
            optimizer_to(opt, "cpu")

    def restore_optimizers(self):
        """
        Puts optimizer states back into device memory.
        """
        if not self.opt_in_cpu:
            return
        for opt in self.optimizers:
            optimizer_to(opt, self.device)

    def resume_training(self, resume_state, load_amp=True, load_optimizers=True, load_schedulers=True):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]
        assert len(resume_optimizers) == len(self.optimizers), "Wrong lengths of optimizers"
        assert len(resume_schedulers) == len(self.schedulers), "Wrong lengths of schedulers"
        if load_optimizers:
            for i, o in enumerate(resume_optimizers):
                self.optimizers[i].load_state_dict(o)
        if load_schedulers:
            for i, s in enumerate(resume_schedulers):
                self.schedulers[i].load_state_dict(s)
        else:
            for i, s in enumerate(resume_schedulers):
                self.schedulers[i].last_epoch = s["last_epoch"]
                self.schedulers[i]._step_count = s["_step_count"]
        if load_amp and "amp" in resume_state.keys():
            from apex import amp

            amp.load_state_dict(resume_state["amp"])
        self.stash_optimizers()
