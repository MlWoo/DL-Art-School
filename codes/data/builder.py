import copy

from utils.registry import Registry, build_from_cfg

DATASETS = Registry("dataset", registor="DLAS")
COLLATIONS = Registry("collation", registor="DLAS")
TOKENIZERS = Registry("tokenizer", registor="DLAS")
TRANSFORMS = Registry("transform", registor="DLAS")


def build_tokenizer(cfg, default_args=None, default_type="type"):
    tokenizer = build_from_cfg(cfg, TOKENIZERS, default_args, default_type=default_type)
    return tokenizer


def build_collation(cfg, default_args=None, default_type="type"):
    collation = build_from_cfg(cfg, COLLATIONS, default_args, default_type=default_type)
    return collation


def build_dataset(cfg, default_args=None, default_type="type"):
    dataset = build_from_cfg(cfg, DATASETS, default_args, default_type=default_type)
    return dataset


def build_datasets(cfg, default_args=None):
    common_dataset_cfg = dict()
    for key, val in cfg.items():
        if key != "datasets_list":
            common_dataset_cfg[key] = val
    datasets_cfg = []
    if cfg.get("datasets_list", None) is None:
        normal_cfg = dict(opt=cfg)
        normal_cfg["type"] = cfg["mode"]
        datasets_cfg.append(normal_cfg)
    else:
        for dataset_cfg in cfg["datasets_list"]:
            instance_cfg = copy.deepcopy(common_dataset_cfg)
            instance_cfg.update(dataset_cfg)
            normal_cfg = dict(opt=instance_cfg)
            normal_cfg["type"] = instance_cfg["mode"]
            datasets_cfg.append(normal_cfg)

    if len(datasets_cfg) == 1:
        datasets_cfg = datasets_cfg[0]

    from data.base.dataset import ConcatDataset

    if isinstance(datasets_cfg, (list, tuple)):
        default_args_list = [
            default_args,
        ] * len(datasets_cfg)
        dataset = ConcatDataset([build_from_cfg(c, DATASETS, args) for c, args in zip(datasets_cfg, default_args_list)])
    elif datasets_cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset([build_from_cfg(c, DATASETS, default_args) for c in datasets_cfg["datasets_list"]])
    else:
        dataset = build_from_cfg(datasets_cfg, DATASETS, default_args)

    return dataset


def build_transforms(cfg, default_args=None, default_type="type"):
    if isinstance(cfg, list):
        transforms = [build_from_cfg(c, TRANSFORMS, default_args, default_type=default_type) for c in cfg]
    else:
        transforms = [build_from_cfg(cfg, TRANSFORMS, default_args, default_type=default_type)]

    return transforms
