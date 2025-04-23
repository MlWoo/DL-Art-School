from utils.registry import Registry, build_from_cfg

DATASETS = Registry("dataset", registor="DLAS")
COLLATIONS = Registry("collation", registor="DLAS")
TOKENIZERS = Registry("tokenizer", registor="DLAS")


def build_collation(cfg, default_args=None, default_type="type"):
    collation = build_from_cfg(cfg, COLLATIONS, default_args, default_type=default_type)
    return collation


def build_dataset(cfg, default_args=None, default_type="type"):
    dataset = build_from_cfg(cfg, DATASETS, default_args, default_type=default_type)
    return dataset


"""
def build_dataset(cfg, default_args=None):
    whole_datasets_cfg = []
    common_dataset_cfg = dict()
    for key, val in cfg.items():
        if key != "datasets_info":
            common_dataset_cfg[key] = val
    for dataset_cfg in cfg["datasets_info"]:
        dataset_cfg.update(common_dataset_cfg)
        whole_datasets_cfg.append(dataset_cfg)
    if len(whole_datasets_cfg) == 1:
        whole_datasets_cfg = whole_datasets_cfg[0]

    from data.base.dataset import ConcatDataset

    if isinstance(whole_datasets_cfg, (list, tuple)):
        default_args_list = [
            default_args,
        ] * len(whole_datasets_cfg)
        dataset = ConcatDataset(
            [build_from_cfg(c, DATASETS, args) for c, args in zip(whole_datasets_cfg, default_args_list)]
        )
    elif whole_datasets_cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset(
            [build_from_cfg(c, DATASETS, default_args) for c in whole_datasets_cfg["datasets_info"]]
        )
    else:
        dataset = build_from_cfg(whole_datasets_cfg, DATASETS, default_args)

    return dataset
"""


def build_tokenizer(cfg, default_args=None, default_type="type"):
    tokenizer = build_from_cfg(cfg, TOKENIZERS, default_args, default_type=default_type)
    return tokenizer
