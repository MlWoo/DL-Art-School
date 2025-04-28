import argparse
import math
import os
import shutil
from time import time

import maybe_bnb
import torch
import torch.distributed as dist
import utils
from model.builder import build_stat
from tqdm import tqdm
from trainer.eval.evaluator import create_evaluator
from trainer.extensible_trainer import ExtensibleTrainer
from utils import logging, logging_utils, options
from utils.distributed import broadcast_object, init_dist, map_cuda_to_correct_device
from utils.io import torch_load
from utils.options import opt_get

from data import create_dataloader, create_dataset_collator
from data.sampler import DistIterSampler


class Trainer:

    def init(self, opt_path, opt):
        self.use_8bit = opt_get(opt, ["use_8bit"], False)
        if self.use_8bit:
            maybe_bnb.populate()
        else:
            maybe_bnb.populate(False, False, False, embedding=None)

        self.profile = opt_get(opt, ["profile_info"], None) and self.rank <= 0
        self.current_step = 0
        self.total_training_data_encountered = 0

        # loading resume state if exists
        if opt["path"].get("resume_state", None):
            # distributed resuming: all load into default GPU
            try:
                resume_state = torch_load(
                    opt["path"]["resume_state"], map_location=map_cuda_to_correct_device, endswith=".state"
                )
            except:  # noqa: E722
                resume_state = None
        else:
            resume_state = None
        # mkdir and loggers
        if self.rank <= 0:  # normal training (self.rank -1) OR distributed training (self.rank 0)
            if resume_state is None:
                utils.mkdir_and_rename(opt["path"]["experiments_root"])  # rename experiment folder if exists
                utils.mkdirs(
                    (
                        path
                        for key, path in opt["path"].items()
                        if not key == "experiments_root"
                        and path is not None
                        and "pretrain_model" not in key
                        and "resume" not in key
                    )
                )
            shutil.copy(
                opt_path,
                os.path.join(opt["path"]["experiments_root"], f"{utils.get_timestamp()}_{os.path.basename(opt_path)}"),
            )

            # config loggers. Before it, the log will not work
            utils.setup_logger(
                "base", opt["path"]["log"], "train_" + opt["name"], level=logging.INFO, screen=True, tofile=True
            )
            self.logger = logging_utils.get_root_logger()
            self.logger.info(options.dict2str(opt))
            # tensorboard logger
            if opt["use_tb_logger"] and "debug" not in opt["name"]:
                self.tb_logger_path = os.path.join(opt["path"]["experiments_root"], "tb_logger")
                from torch.utils.tensorboard import SummaryWriter

                self.tb_logger = SummaryWriter(log_dir=self.tb_logger_path)
        else:
            utils.setup_logger("base", opt["path"]["log"], "train", level=logging.INFO, screen=True)
            self.logger = logging_utils.get_root_logger()

        if resume_state is not None:
            options.check_resume(opt, resume_state["iter"])  # check resume optionss

        # convert to NoneDict, which returns None for missing keys
        opt = options.dict_to_nonedict(opt)
        self.opt = opt

        # wandb init
        if opt["wandb"] and self.rank <= 0:
            import wandb

            os.makedirs(os.path.join(opt["path"]["log"], "wandb"), exist_ok=True)
            project_name = opt_get(opt, ["wandb_project_name"], opt["name"])
            run_name = opt_get(opt, ["wandb_run_name"], None)
            wandb.init(project=project_name, dir=opt["path"]["log"], config=opt, name=run_name)

        # random seed
        seed = opt["train"]["manual_seed"]
        """
        # Different multiprocessing instances should behave differently.
        if seed is None:
            seed = random.randint(1, 10000)
        if self.rank <= 0:
            self.logger.info('Random seed: {}'.format(seed))
        seed += self.rank
        """
        utils.set_random_seed(seed)

        torch.backends.cudnn.benchmark = opt_get(opt, ["cuda_benchmarking_enabled"], True)
        torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.deterministic = True
        if opt_get(opt, ["anomaly_detection"], False):
            torch.autograd.set_detect_anomaly(True)

        # Save the compiled opt dict to the global loaded_optionss variable.
        utils.loaded_optionss = opt

        # create model
        self.model = ExtensibleTrainer(opt)
        if opt_get(opt, ["apply_compile"], False):
            self.model.apply_compile()

        # create train and val dataloader
        dataset_ratio = 1  # enlarge the size of each epoch

        # processing dataloader
        for phase, dataloader_opt in opt["dataloaders"].items():
            if phase == "uni":
                self.batch_size = dataloader_opt["sampler"]["batch_size"]
                self.train_val_set, collate_fn = create_dataset_collator(dataloader_opt, return_collate=True)
                loaders = create_dataloader(
                    self.train_val_set, dataloader_opt, opt, None, collate_fn=collate_fn, shuffle=True
                )
                self.train_loader = loaders["train"]
                self.val_loader = loaders["test"]
                total_iters = int(opt["train"]["niter"])
                train_size = len(self.train_loader)
                self.total_epochs = int(math.ceil(total_iters / train_size))
                if self.rank <= 0:
                    train_len = len(self.train_val_set.phases_indice_dict["train"])
                    val_len = len(self.train_val_set.phases_indice_dict["test"])
                    self.logger.info(
                        "Number of training data elements: {:,d}, iters: {:,d}".format(train_len, train_size)
                    )
                    self.logger.info("Total epochs needed: {:d} for iters {:,d}".format(self.total_epochs, total_iters))

                    self.logger.info(
                        "Number of val  data elements in [{:s}]: {:d}".format(
                            dataloader_opt["dataset"]["name"], val_len
                        )
                    )
            elif phase == "val":
                self.batch_size = dataloader_opt["sampler"]["batch_size"]
                self.val_set, collate_fn = create_dataset_collator(dataloader_opt, return_collate=True)
                self.val_loader = create_dataloader(self.val_set, dataloader_opt, opt, None, collate_fn=collate_fn)
                if self.rank <= 0:
                    self.logger.info(
                        "Number of val data elements in [{:s}]: {:d}".format(
                            dataloader_opt["dataset"]["name"], len(self.val_set)
                        )
                    )
            elif phase == "train":
                self.batch_size = dataloader_opt["sampler"]["batch_size"]
                self.train_set, collate_fn = create_dataset_collator(dataloader_opt, return_collate=True)
                train_size = int(math.ceil(len(self.train_set) / self.batch_size))
                total_iters = int(opt["train"]["niter"])
                self.total_epochs = int(math.ceil(total_iters / train_size))
                if opt["dist"]:
                    self.train_sampler = DistIterSampler(self.train_set, self.world_size, self.rank, dataset_ratio)
                    self.total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
                    shuffle = False
                else:
                    self.train_sampler = None
                    shuffle = True
                loader = create_dataloader(
                    self.train_set, dataloader_opt, opt, self.train_sampler, collate_fn=collate_fn, shuffle=shuffle
                )

                self.train_loader = loader
                if self.rank <= 0:
                    self.logger.info(
                        "Number of training data elements: {:,d}, iters: {:,d}".format(len(self.train_set), train_size)
                    )
                    self.logger.info("Total epochs needed: {:d} for iters {:,d}".format(self.total_epochs, total_iters))
            else:
                raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))

        assert self.train_loader is not None

        # search for batch size for bucketed dataloader
        search_batch_size_for_bucket = opt_get(opt, ["search_batch_size_for_bucket"], False)
        if search_batch_size_for_bucket:
            if self.train_loader is not None:
                self.logger.info("Searching for best batch size for train loader if batch mode is bucketed")
                self.search_and_sync_batch_size_for_bucket(self.train_loader, "train")

            if self.val_loader is not None:
                self.logger.info("Searching for best batch size for val loader if batch mode is bucketed")
                self.search_and_sync_batch_size_for_bucket(self.val_loader, "val")

        # Evaluators
        self.evaluators = []
        if "eval" in opt.keys() and "evaluators" in opt["eval"].keys():
            # In "pure" mode, we propagate through the normal training steps, but use validation data
            # instead and average the total loss. A validation dataloader is required.
            if opt_get(opt, ["eval", "pure"], False):
                assert hasattr(self, "val_loader")

            for ev_key, ev_opt in opt["eval"]["evaluators"].items():
                self.evaluators.append(create_evaluator(self.model.networks[ev_opt["for"]], ev_opt, self.model.env))

        # resume training
        if resume_state:
            if self.rank <= 0:
                self.logger.info(
                    "Resuming training from epoch: {}, iter: {}.".format(resume_state["epoch"], resume_state["iter"])
                )

            self.start_epoch = resume_state["epoch"]
            self.current_step = resume_state["iter"]
            self.total_training_data_encountered = opt_get(resume_state, ["total_data_processed"], 0)
            optimizer_reset = opt_get(opt, ["path", "optimizer_reset"], False)
            scheduler_reset = opt_get(opt, ["path", "scheduler_reset"], False)
            if self.rank <= 0:
                if optimizer_reset:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("!! RESETTING OPTIMIZER STATES")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                if scheduler_reset:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("!! RESETTING SCHEDULER STATES")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.model.resume_training(
                resume_state,
                "amp_opt_level" in opt.keys(),
                load_optimizers=not optimizer_reset,
                load_schedulers=not scheduler_reset,
            )  # handle optimizers and schedulers
        else:
            self.current_step = -1 if "start_step" not in opt.keys() else opt["start_step"]
            self.total_training_data_encountered = (
                0 if "training_data_encountered" not in opt.keys() else opt["training_data_encountered"]
            )
            self.start_epoch = 0
        if "force_start_step" in opt.keys():
            self.current_step = opt["force_start_step"]
            self.total_training_data_encountered = self.current_step * self.batch_size
        opt["current_step"] = self.current_step

        # validation
        if "val_freq" in opt["train"].keys():
            self.val_freq = opt["train"]["val_freq"] * self.batch_size
        else:
            self.val_freq = int(opt["train"]["val_freq_megasamples"] * 1000000)

        self.next_eval_step = self.total_training_data_encountered + self.val_freq
        # For whatever reason, this relieves a memory burden on the first GPU for some training sessions.
        del resume_state

    def do_step(self, train_data, _t):
        if self.profile:
            data_fetch_time = time() - _t
            # print("Data fetch: %f" % (time() - _t))
            _t = time()
        else:
            data_fetch_time = -1.0  # noqa: F841

        opt = self.opt
        batch_size = self.batch_size  # it's werid
        self.current_step += 1
        self.total_training_data_encountered += batch_size
        will_log = self.current_step % opt["logger"]["print_freq"] == 0
        will_visual = (
            "visuals" in self.opt["logger"].keys()
            and self.rank <= 0
            and self.current_step % self.opt["logger"]["visual_debug_rate"] == 0
        )

        # training
        if self.profile:
            lr_update_time = time() - _t  # noqa: F841
            # print("Update LR: %f" % (time() - _t))
            _t = time()
        else:
            lr_update_time = -1.0  # noqa: F841

        self.model.feed_run_info(step=self.current_step, epoch=self.epoch, will_log=will_log, will_visual=will_visual)
        if opt_get(self.opt, ["oom", "raise_error"], False):
            self.model.feed_data(train_data, self.current_step, profile=self.profile)
            if self.profile:
                model_feed_time = time() - _t  # noqa: F841
                # print("Model feed: %f" % (time() - _t))
                _t = time()
            else:
                model_feed_time = -1.0  # noqa: F841

            is_oom, naked_gradient_norms_dict = self.model.optimize_parameters(
                self.current_step, return_grad_norms=will_log
            )
            if is_oom:
                raise RuntimeError("OOM occurs in training stage.")
        else:
            m = 0
            while m < opt_get(self.opt, ["oom", "oom_trials"], 2):
                self.model.feed_data(train_data, self.current_step, reduce_batch_factor=2**m, profile=self.profile)
                if self.profile:
                    model_feed_time = time() - _t  # noqa: F841
                    # print("Model feed: %f" % (time() - _t))
                    _t = time()
                else:
                    model_feed_time = -1.0  # noqa: F841

                is_oom, naked_gradient_norms_dict = self.model.optimize_parameters(
                    self.current_step, return_grad_norms=will_log
                )
                if m > 0 and not is_oom and self.rank <= 0:
                    print("OOM reduction trial {} succeeded".format(m))
                    break
                elif not is_oom:
                    break
                m += 1

        gradient_norms_dict = dict()
        for k, v in naked_gradient_norms_dict.items():
            gradient_norms_dict[f"gradnorm_{k}"] = v

        iteration_rate = (time() - _t) / batch_size
        if self.profile:
            model_optimize_time = time() - _t  # noqa: F841
            # print("Model model: %f" % (time() - _t))
            _t = time()
        else:
            model_optimize_time = -1.0  # noqa: F841

        # log
        if will_log:
            # Must be run by all instances to gather consensus.
            current_model_logs = self.model.get_current_log(self.current_step)
        if will_log and self.rank <= 0:
            logs = {
                "step": self.current_step,
                "samples": self.total_training_data_encountered,
                "megasamples": self.total_training_data_encountered / 1000000,
                "iteration_rate": iteration_rate,
            }
            if self.profile:
                for info in self.profile:
                    logs[info] = self.profile[info]

            logs.update(current_model_logs)
            logs.update(gradient_norms_dict)
            message = "[rank: {} epoch:{:3d}, iter:{:8,d}, lr:(".format(self.rank, self.epoch, self.current_step)
            for v in self.model.get_current_learning_rate():
                message += "{:.3e},".format(v)
            message += ")] "
            for k, v in logs.items():
                if "histogram" in k:
                    self.tb_logger.add_histogram(k, v, self.current_step)
                elif isinstance(v, dict):
                    self.tb_logger.add_scalars(k, v, self.current_step)
                else:
                    if v is not None:
                        message += "{:s}: {:.4e} ".format(k, v)
                        # tensorboard logger
                        if opt["use_tb_logger"] and "debug" not in opt["name"]:
                            self.tb_logger.add_scalar(k, v, self.current_step)

            if opt["wandb"] and self.rank <= 0:
                import wandb

                wandb_logs = {}
                for k, v in logs.items():
                    if "histogram" in k:
                        wandb_logs[k] = wandb.Histogram(v)
                    else:
                        wandb_logs[k] = v
                if opt_get(opt, ["wandb_progress_use_raw_steps"], False):
                    wandb.log(wandb_logs, step=self.current_step)
                else:
                    wandb.log(wandb_logs, step=self.total_training_data_encountered)

            self.logger.info(message)

        # update learning rate
        self.model.update_learning_rate(self.current_step, warmup_iter=opt["train"]["warmup_iter"])

        # save models and training states
        if self.current_step % opt["logger"]["save_checkpoint_freq"] == 0:
            self.model.consolidate_state()
            if self.rank <= 0:
                if opt["logger"]["disable_state_saving"] is False:
                    self.logger.info("Saving models and training states.")
                else:
                    self.logger.info("Saving model.")

                if (
                    opt["upgrades"]["number_of_checkpoints_to_save"] > 0
                    or opt["upgrades"]["number_of_states_to_save"] > 0
                ):

                    number_of_states_to_save = (
                        opt["upgrades"]["number_of_states_to_save"] if not opt["logger"]["disable_state_saving"] else 0
                    )

                    self.logger.info(
                        f"Leaving only {opt['upgrades']['number_of_checkpoints_to_save']} checkpoints and "
                        f"{number_of_states_to_save} states"
                    )
                    self.model.limit_number_of_checkpoints_and_states(
                        models_number=opt["upgrades"]["number_of_checkpoints_to_save"],
                        state_number=opt["upgrades"]["number_of_states_to_save"],
                    )

                self.model.save(self.current_step)
                state = {
                    "epoch": self.epoch,
                    "iter": self.current_step,
                    "total_data_processed": self.total_training_data_encountered,
                }
                if not opt["logger"]["disable_state_saving"]:
                    self.model.save_training_state(state)
                else:
                    self.logger.info(
                        "State saving is disabled. Skipping state save, "
                        "you won't be able to resume training from this session."
                    )
            if "alt_path" in opt["path"].keys():
                import shutil

                print("Synchronizing tb_logger to alt_path..")
                alt_tblogger = os.path.join(opt["path"]["alt_path"], "tb_logger")
                shutil.rmtree(alt_tblogger, ignore_errors=True)
                shutil.copytree(self.tb_logger_path, alt_tblogger)

        do_eval = self.total_training_data_encountered > self.next_eval_step
        if do_eval:
            self.next_eval_step = self.total_training_data_encountered + self.val_freq

        if opt_get(opt, ["eval", "pure"], False) and do_eval:
            metrics = []
            if hasattr(self.val_loader, "set_epoch"):
                self.val_loader.set_epoch(0)
            for val_data in tqdm(self.val_loader):
                self.model.feed_data(val_data, self.current_step, perform_micro_batching=False)
                is_oom, result = self.model.test()
                metrics.append(result)

            reduced_metrics = {}
            for metric in metrics:
                for k, v in metric.as_dict().items():
                    if isinstance(v, torch.Tensor) and len(v.shape) == 0:
                        if k in reduced_metrics.keys():
                            reduced_metrics[k].append(v)
                        else:
                            reduced_metrics[k] = [v]
            if self.rank <= 0:
                for k, v in reduced_metrics.items():
                    val = torch.stack(v).mean().item()
                    self.tb_logger.add_scalar(f"val_{k}", val, self.current_step)
                    print(f">>Eval {k}: {val}")
                if opt["wandb"]:
                    import wandb

                    wandb.log({f"eval_{k}": torch.stack(v).mean().item() for k, v in reduced_metrics.items()})

        if len(self.evaluators) != 0 and do_eval:
            eval_dict = {}
            for eval in self.evaluators:
                if eval.uses_all_ddp or self.rank <= 0:
                    eval_dict.update(eval.perform_eval())
            if self.rank <= 0:
                print("Evaluator results: ", eval_dict)
                for ek, ev in eval_dict.items():
                    self.tb_logger.add_scalar(ek, ev, self.current_step)
                if opt["wandb"]:
                    import wandb

                    wandb.log(eval_dict)

        # Should not be necessary, but make absolutely sure that there is no grad leakage from validation runs.
        for net in self.model.networks.values():
            net.zero_grad()

    def do_training(self):
        if self.rank <= 0:
            self.logger.info("Start training from epoch: {:d}, iter: {:d}".format(self.start_epoch, self.current_step))

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self.epoch = epoch
            if False and self.opt["dist"]:
                self.train_sampler.set_epoch(epoch)
            if hasattr(self.train_loader, "set_epoch"):
                self.train_loader.set_epoch(epoch)

            tq_ldr = (
                tqdm(self.train_loader, miniters=opt["logger"]["print_freq"]) if self.rank <= 0 else self.train_loader
            )
            _t = time()
            for train_data in tq_ldr:
                self.do_step(train_data, _t)
                _t = time()

    def create_training_generator(self, index):
        if self.rank <= 0:
            self.logger.info("Start training from epoch: {:d}, iter: {:d}".format(self.start_epoch, self.current_step))
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self.epoch = epoch
            if False and self.opt["dist"]:
                self.train_sampler.set_epoch(epoch)
            if hasattr(self.train_loader, "set_epoch"):
                self.train_loader.set_epoch(epoch)
            tq_ldr = tqdm(self.train_loader, position=index)

            _t = time()  # noqa: F841
            for train_data in tq_ldr:
                yield self.model
                self.do_step(train_data)

    def online_stat(self):
        if self.rank <= 0:
            self.logger.info("Statistic from epoch: {:d}, iter: {:d}".format(self.start_epoch, self.current_step))
        online_stat = opt_get(self.opt, ["online_stat"], None)
        if online_stat is not None and opt_get(online_stat, ["enabled"], False):
            stat = build_stat(online_stat["stat_cfg"])
            stat_batches = opt_get(online_stat, ["stat_batches"], 100)
            save_dir = opt_get(online_stat, ["save_dir"], "stat/dir")
            os.makedirs(save_dir, exist_ok=True)
            if hasattr(self.train_loader, "set_epoch"):
                self.train_loader.set_epoch(0)
            tq_ldr = tqdm(self.train_loader)

            cnt = 0
            for train_data in tq_ldr:
                stat.update(train_data)
                cnt += 1
                if cnt > stat_batches:
                    break
            stat.save(cnt, save_dir=save_dir)
            self.logger.info(f"Online stat saved at {save_dir}")

    def search_and_sync_batch_size_for_bucket(self, dataloader, phase):
        sampler = dataloader.sampler
        if sampler.batch_mode != "bucketed":
            self.logger.info(f"Batch mode at {phase=} is not bucketed, skipping search for best batch size")
            return
        bucket_boundaries_batch_size_map = self.search_batch_size_for_bucket(dataloader, phase)
        if self.opt["dist"]:
            dist.barrier()
            bucket_boundaries_batch_size_map = broadcast_object(bucket_boundaries_batch_size_map, src_rank=0)
        self.sync_dataloader_config(dataloader, bucket_boundaries_batch_size_map)

    def search_batch_size_for_bucket(self, dataloader, phase):
        self.logger.info(f"Searching for best batch size for {phase=}, waiting for a moment...")
        sampler = dataloader.sampler
        dataset = dataloader.dataset
        assert hasattr(sampler, "bucket_boundaries"), "sampler must have bucket_boundaries attribute"
        assert hasattr(sampler, "bucket_max_samples"), "sampler must have bucket_max_batch_size attribute"
        assert hasattr(sampler, "bucket_min_samples"), "sampler must have bucket_min_batch_size attribute"
        assert hasattr(dataset, "create_dummy_input"), "dataset must have create_dummy_input attribute"

        def check_oom(batch_size, bucket_boundary):
            dummy_input = dataset.create_dummy_input(batch_size, bucket_boundary)
            self.model.feed_data(dummy_input, self.current_step, perform_micro_batching=False)
            if phase == "train":
                is_oom, _ = self.model.optimize_parameters(self.current_step, return_grad_norms=False, raise_oom=False)
            else:
                is_oom, _ = self.model.test(raise_oom=False)
            return is_oom

        def search_batch_size(batch_size_start, bucket_boundary):
            batch_size = batch_size_start
            is_oom = False
            step = 1
            while not is_oom:
                is_oom = check_oom(batch_size, bucket_boundary)
                if not is_oom:
                    step *= 2
                batch_size += step

            trials = 3
            while is_oom or batch_size == 0:
                is_oom = check_oom(batch_size, bucket_boundary)
                if not is_oom:
                    for i in range(trials):
                        is_oom = check_oom(batch_size, bucket_boundary)
                        if not is_oom:
                            break
                batch_size -= 1

            return batch_size

        trials = 3
        is_oom = False
        while not is_oom and trials > 0:
            is_oom = check_oom(1, sampler.bucket_boundaries[-1].item())
            trials -= 1
        if is_oom:
            raise RuntimeError(
                f"No valid batch size found at phase: {phase} for boundary {sampler.bucket_boundaries[-1].item()} "
                "because of insufficient GPU memory."
            )
        else:
            self.logger.info(
                f"Found valid batch size at phase: {phase} for boundary {sampler.bucket_boundaries[-1].item()}"
            )

        pre_bucket_boundary = None
        pre_batch_size = sampler.bucket_max_samples
        batch_sizes = []
        bucket_boundaries_batch_size_map = {}
        torch.cuda.set_per_process_memory_fraction(
            opt_get(self.opt, ["search_batch_size_for_bucket_max_memory_fraction"], 0.8)
        )
        for bucket_boundary in sampler.bucket_boundaries:
            if pre_bucket_boundary is not None:
                assert bucket_boundary > pre_bucket_boundary, "bucket_boundary must be greater than pre_bucket_boundary"
            found = False
            for batch_size in range(pre_batch_size, sampler.bucket_min_samples, -2):
                for i in range(3):
                    is_oom = check_oom(batch_size, bucket_boundary)
                    if is_oom:
                        break
                if not is_oom:
                    batch_size += 1
                    for i in range(3):
                        is_oom = check_oom(batch_size, bucket_boundary)
                        if is_oom:
                            break
                    if is_oom:
                        batch_size -= 1
                        is_oom = False

                if not is_oom:
                    if batch_size == sampler.bucket_max_samples:
                        self.logger.warning(
                            f"Found best {batch_size=} for {bucket_boundary=} is equal to bucket_max_samples, "
                            "the bucket_max_samples is maybe too small."
                        )
                    else:
                        self.logger.info(f"Found best {batch_size=} for {bucket_boundary=}")
                    bucket_boundaries_batch_size_map[bucket_boundary.item()] = torch.LongTensor([batch_size])[0]
                    batch_sizes.append(batch_size)
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"No valid batch size found at phase: {phase} for boundary {bucket_boundary} "
                    "because of insufficient GPU memory."
                )
            else:
                pre_batch_size = batch_sizes[-1]
                pre_bucket_boundary = bucket_boundary

        # for bucket_boundary in torch.flip(sampler.bucket_boundaries, dims=(0,)):
        #     if pre_bucket_boundary is not None:
        #         assert bucket_boundary < pre_bucket_boundary, "bucket_boundary must be less than pre_bucket_boundary"

        #     batch_size_found = search_batch_size(pre_batch_size, bucket_boundary)
        #     if batch_size_found > 0:
        #         if batch_size_found > sampler.bucket_max_samples:
        #             self.logger.warning(
        #                 f"Found best {batch_size_found=} for {bucket_boundary=} is greater than bucket_max_samples, "
        #                 "the bucket_max_samples is maybe too small."
        #             )
        #         bucket_boundaries_batch_size_map[bucket_boundary.item()] = torch.LongTensor([batch_size_found])[0]
        #         pre_batch_size = batch_size_found
        #         pre_bucket_boundary = bucket_boundary
        #     else:
        #         raise RuntimeError(
        #             f"No valid batch size found at phase: {phase} for boundary {bucket_boundary} "
        #             "because of insufficient GPU memory."
        #         )

        self.logger.info(
            f"Searching done. The bucket_boundaries_batch_size_map for {phase=} is: {bucket_boundaries_batch_size_map}"
        )
        torch.cuda.set_per_process_memory_fraction(1.0)

        return bucket_boundaries_batch_size_map

    def sync_dataloader_config(self, dataloader, bucket_boundaries_batch_size_map):
        dataloader.sampler.bucket_boundaries_batch_size_map = bucket_boundaries_batch_size_map
        dataloader.collate_fn.set_bucketed_batch_size(dataloader.sampler.similar_type, bucket_boundaries_batch_size_map)
        self.model.apply_compile()


if __name__ == "__main__":
    # from multiprocess import set_start_method
    # set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to options YAML file.", default="../optionss/train_vit_latent.yml")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "mpi", "torchrun"], default="none", help="job launcher"
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus", type=int, help="number of gpus to use " "(only applicable to non-distributed training)"
    )
    group_gpus.add_argument(
        "--gpu-ids", type=int, nargs="+", help="ids of gpus to use " "(only applicable to non-distributed training)"
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        default="12345",
    )
    args = parser.parse_args()
    opt = options.parse(args.opt, is_train=True)

    if args.gpu_ids is not None:
        opt["gpu_ids"] = args.gpu_ids
    else:
        opt["gpu_ids"] = opt["gpu_ids"] if args.gpus is None else range(args.gpus)

    if len(opt["gpu_ids"]) > 1 and args.launcher == "none":
        launcher = "mpi"
    else:
        launcher = args.launcher

    if launcher != "none":
        # export CUDA_VISIBLE_DEVICES for running in distributed mode.
        if "gpu_ids" in opt.keys():
            gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
            print("export CUDA_VISIBLE_DEVICES=" + gpu_list)
    trainer = Trainer()

    # distributed training settings
    if launcher == "none":  # disabled distributed training
        opt["dist"] = False
        trainer.rank = -1
        if len(opt["gpu_ids"]) == 1:
            torch.cuda.set_device(opt["gpu_ids"][0])
    elif launcher == "torchrun":
        opt["dist"] = True
        init_dist(launcher, "nccl")
        trainer.world_size = torch.distributed.get_world_size()
        trainer.rank = torch.distributed.get_rank()
    else:
        opt["dist"] = True
        rank_device_map = dict()
        for i, gpu_id in enumerate(opt["gpu_ids"]):
            rank_device_map[i] = gpu_id

        init_dist(
            launcher,
            "nccl",
            rank_device_map=rank_device_map,
            ngpus=len(opt["gpu_ids"]),
            port=args.port,
            master_addr=args.master_addr,
        )
        trainer.world_size = torch.distributed.get_world_size()
        trainer.rank = torch.distributed.get_rank()
        # torch.cuda.set_device(rank)
    trainer.init(args.opt, opt)
    trainer.online_stat()
    trainer.do_training()
