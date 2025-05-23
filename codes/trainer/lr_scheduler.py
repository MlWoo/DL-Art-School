import math
from collections import Counter, defaultdict

import torch
from torch.optim.lr_scheduler import _LRScheduler
from utils.options import opt_get


def get_scheduler_for_name(name, optimizers, scheduler_opt):
    schedulers = []
    for o in optimizers:
        # Hack to support LARC, which wraps an underlying optimizer.
        if hasattr(o, "optim"):
            o = o.optim

        if name == "MultiStepLR":
            sched = MultiStepLR_Restart(
                o,
                scheduler_opt["gen_lr_steps"],
                restarts=scheduler_opt["restarts"],
                weights=scheduler_opt["restart_weights"],
                gamma=scheduler_opt["lr_gamma"],
                clear_state=scheduler_opt["clear_state"],
                force_lr=scheduler_opt["force_lr"],
                warmup=opt_get(scheduler_opt, ["warmup"], 0),
            )
        elif name == "ProgressiveMultiStepLR":
            sched = ProgressiveMultiStepLR(
                o, scheduler_opt["gen_lr_steps"], scheduler_opt["progressive_starts"], scheduler_opt["lr_gamma"]
            )
        elif name == "CosineAnnealingLR_Restart":
            sched = CosineAnnealingLR_Restart(
                o,
                scheduler_opt["T_period"],
                scheduler_opt["warmup"],
                eta_min=scheduler_opt["eta_min"],
                restarts=scheduler_opt["restarts"],
                weights=scheduler_opt["restart_weights"],
            )
        elif name == "TransformerLR":
            sched = TransformerLrScheduler(
                o,
                scheduler_opt["warmup"] or scheduler_opt["warmup_iter"],
                opt_get(scheduler_opt, ["num_model_dim"], None),
                max_lr=scheduler_opt["max_lr"],
            )
        else:
            raise NotImplementedError("Scheduler not available")
        schedulers.append(sched)
    return schedulers


# This scheduler is specifically designed to modulate the learning rate of several different param groups configured
# by a generator or discriminator that slowly adds new stages one at a time, e.g. like progressive growing of GANs.
class ProgressiveMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, group_starts, gamma=0.1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.group_starts = group_starts
        super(ProgressiveMultiStepLR, self).__init__(optimizer)

    def get_lr(self):
        group_lrs = []
        assert len(self.optimizer.param_groups) == len(self.group_starts)
        for group, group_start in zip(self.optimizer.param_groups, self.group_starts):
            if self.last_epoch - group_start not in self.milestones:
                group_lrs.append(group["lr"])
            else:
                group_lrs.append(group["lr"] * self.gamma)
        return group_lrs


class MultiStepLR_Restart(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        restarts=None,
        weights=None,
        gamma=0.1,
        clear_state=False,
        force_lr=False,
        last_epoch=-1,
        warmup=0,
    ):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.force_lr = force_lr
        if force_lr:
            print(f"!!Forcing the learning rate to: {force_lr}")
        self.warmup = warmup
        assert len(self.restarts) == len(self.restart_weights), "restarts and their weights do not match."
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Note to self: for the purposes of this trainer, "last_epoch" should read "last_step"
        if self.force_lr is not None:
            return [self.force_lr for _ in self.optimizer.param_groups]
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group["initial_lr"] * weight for group in self.optimizer.param_groups]
        if self.last_epoch < self.warmup:
            factor = 1 - (self.warmup - self.last_epoch) / self.warmup
            return [group["initial_lr"] * factor for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma ** self.milestones[self.last_epoch] for group in self.optimizer.param_groups]

    # Allow this scheduler to use newly appointed milestones partially through a training run..
    def load_state_dict(self, s):
        milestones_cache = self.milestones
        force_lr_cache = self.force_lr
        super(MultiStepLR_Restart, self).load_state_dict(s)
        self.milestones = milestones_cache
        self.force_lr = force_lr_cache


class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, warmup=0, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.warmup = warmup
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(self.restart_weights), "restarts and their weights do not match."
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch - self.warmup
        if step <= 0:
            return self.base_lrs
        elif step in self.restarts:
            self.last_restart = step
            self.T_max = self.T_period[self.restarts.index(step) + 1]
            weight = self.restart_weights[self.restarts.index(step)]
            return [group["initial_lr"] * weight for group in self.optimizer.param_groups]
        elif (step - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * (step - self.last_restart) / self.T_max))
            / (1 + math.cos(math.pi * ((step - self.last_restart) - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]


class TransformerLrScheduler(_LRScheduler):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, warmup, num_model_dim=None, max_lr=None, last_epoch=-1):
        self.warmup = warmup
        self.max_lr = max_lr
        self.init_lr = math.pow(num_model_dim, -0.5)
        if num_model_dim is not None:
            self.init_base_lr(optimizer)
        self.epoch_offset = 1 - last_epoch
        super(TransformerLrScheduler, self).__init__(optimizer, last_epoch)

    def init_base_lr(self, optimizer):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in optimizer.param_groups:
            group["lr"] = self.init_lr

    def get_lr(self):
        process = self.epoch_offset + self.last_epoch
        weight = min([math.pow(process, -0.5), math.pow(self.warmup, -1.5) * process])
        lr_groups = [group["initial_lr"] * weight for group in self.optimizer.param_groups]

        if self.max_lr is None:
            return lr_groups
        else:
            return [min(lr, self.max_lr) for lr in lr_groups]


if __name__ == "__main__":
    # torch.optim.Adam
    optimizer = torch.optim.Adam([torch.zeros(3, 64, 3, 3)], lr=1e-4, weight_decay=0, betas=(0.9, 0.99))
    ##############################
    # MultiStepLR_Restart
    ##############################
    # Original
    lr_steps = [200000, 400000, 600000, 800000]
    restarts = None
    restart_weights = None

    # two
    lr_steps = [100000, 200000, 300000, 400000, 490000, 600000, 700000, 800000, 900000, 990000]
    restarts = [500000]
    restart_weights = [1]

    # four
    lr_steps = [
        50000,
        100000,
        150000,
        200000,
        240000,
        300000,
        350000,
        400000,
        450000,
        490000,
        550000,
        600000,
        650000,
        700000,
        740000,
        800000,
        850000,
        900000,
        950000,
        990000,
    ]
    restarts = [250000, 500000, 750000]
    restart_weights = [1, 1, 1]

    scheduler = MultiStepLR_Restart(
        optimizer, lr_steps, restarts, restart_weights, gamma=0.5, clear_state=False, warmup=20000
    )
    """
    ##############################
    # Cosine Annealing Restart
    ##############################
    ## two
    T_period = [500000, 500000]
    restarts = [500000]
    restart_weights = [1]

    ## four
    T_period = [200000, 100000, 200000]
    restarts = [200000, 300000]
    restart_weights = [.5, .25]

    scheduler = CosineAnnealingLR_Restart(optimizer, T_period, warmup=10000, eta_min=1e-8, restarts=restarts,
                                          weights=restart_weights)
    """

    ##############################
    # Draw figure
    ##############################
    N_iter = 100000
    lr_l = list(range(N_iter))
    for i in range(N_iter):
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        lr_l[i] = current_lr

    import matplotlib as mpl
    import matplotlib.ticker as mtick
    from matplotlib import pyplot as plt

    mpl.style.use("default")
    import seaborn

    seaborn.set(style="whitegrid")
    seaborn.set_context("paper")

    plt.figure(1)
    plt.subplot(111)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    plt.title("Title", fontsize=16, color="k")
    plt.plot(list(range(N_iter)), lr_l, linewidth=1.5, label="learning rate scheme")
    legend = plt.legend(loc="upper right", shadow=False)
    ax = plt.gca()
    labels = ax.get_xticks().tolist()
    for k, v in enumerate(labels):
        labels[k] = str(int(v / 1000)) + "K"
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))

    ax.set_ylabel("Learning rate")
    ax.set_xlabel("Iteration")
    fig = plt.gcf()
    plt.show()
