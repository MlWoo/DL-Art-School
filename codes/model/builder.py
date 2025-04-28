from utils.registry import Registry, build_from_cfg

STAT = Registry("stat_type", registor="DLAS")


def build_stat(cfg, default_args=None, default_type="type"):
    stat = build_from_cfg(cfg, STAT, default_args, default_type=default_type)
    return stat
