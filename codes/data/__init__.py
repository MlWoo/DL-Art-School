"""create dataset and dataloader"""

import copy
import importlib
import pkgutil

import torch
import torch.utils.data
from utils.options import opt_get

from data.builder import build_collation, build_datasets
from data.dataloader import GeneralSeqDataloader


def registered_module(base_path="model"):
    found_fns = {}
    module_iter = pkgutil.walk_packages([base_path])
    for mod in module_iter:
        if mod.ispkg:
            EXCLUSION_LIST = ["flownet2"]
            if mod.name not in EXCLUSION_LIST:
                found_fns.update(registered_module(f"{base_path}/{mod.name}"))
        else:
            mod_name = f"{base_path}/{mod.name}".replace("/", ".")
            importlib.import_module(mod_name)


def create_dataset_collator(dataloader_opt, return_collate=False):
    collate = None
    dataset_opt = dataloader_opt["dataset"]
    collator_opt = dataloader_opt["collator"]
    # datasets for audio
    registered_module("data/audio")
    dataset = build_datasets(dataset_opt)
    collate_mode = opt_get(collator_opt, ["mode"], None)
    if collate_mode is None:
        collate = None
    else:
        cfg = dict(opt=collator_opt)
        cfg["type"] = collate_mode
        collate = build_collation(cfg)

    if return_collate:
        return dataset, collate
    else:
        return dataset


def create_dataloader(dataset, dataloader_opt, opt=None, sampler=None, collate_fn=None, shuffle=True):
    sampler_opt = opt_get(dataloader_opt, ["sampler"])
    buffer_batch_group = opt_get(sampler_opt, ["buffer_batch_group"], 0)
    pin_memory = opt_get(dataloader_opt, ["pin_memory"], True)
    if buffer_batch_group > 0:
        new_dataloader_opt = copy.deepcopy(dataloader_opt)
        new_dataloader_opt["dataset"] = dataset
        new_dataloader_opt["collate_fn"] = collate_fn
        new_dataloader_opt["sampler_opt"] = sampler_opt
        new_dataloader_opt["workers_per_gpu"] = new_dataloader_opt["n_workers"]
        new_dataloader_opt["pin_memory"] = pin_memory
        dataloaders = GeneralSeqDataloader(**new_dataloader_opt)
        phase = opt_get(dataloader_opt, ["dataset", "phase"])
        if phase in dataloaders.named_dataloaders:
            return dataloaders.dataloader(phase)
        else:
            return dataloaders.named_dataloaders
    else:
        if phase == "train":
            if opt_get(opt, ["dist"], False):
                world_size = torch.distributed.get_world_size()
                num_workers = dataloader_opt["n_workers"]
                assert dataloader_opt["batch_size"] % world_size == 0
                batch_size = dataloader_opt["batch_size"] // world_size
            else:
                num_workers = dataloader_opt["n_workers"]
                batch_size = dataloader_opt["batch_size"]
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                sampler=sampler,
                drop_last=True,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )
        else:
            batch_size = dataloader_opt["batch_size"] or 1
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )
