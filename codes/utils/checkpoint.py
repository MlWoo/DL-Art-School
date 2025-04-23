import re
from collections import OrderedDict

import torch
from utils.distributed import get_dist_info
from utils.io import torch_load
from utils.options import loaded_options


# Conditionally uses torch's checkpoint functionality if it is enabled in the opt file.
def checkpoint(fn, *args):
    if loaded_options is None:
        enabled = False
    else:
        enabled = loaded_options["checkpointing_enabled"] if "checkpointing_enabled" in loaded_options.keys() else True
    if enabled:
        return torch.utils.checkpoint.checkpoint(fn, *args)
    else:
        return fn(*args)


def sequential_checkpoint(fn, partitions, *args):
    if loaded_options is None:
        enabled = False
    else:
        enabled = loaded_options["checkpointing_enabled"] if "checkpointing_enabled" in loaded_options.keys() else True
    if enabled:
        return torch.utils.checkpoint.checkpoint_sequential(fn, partitions, *args)
    else:
        return fn(*args)


# A fancy alternative to if <flag> checkpoint() else <call>
def possible_checkpoint(opt_en, fn, *args):
    if loaded_options is None:
        enabled = False
    else:
        enabled = loaded_options["checkpointing_enabled"] if "checkpointing_enabled" in loaded_options.keys() else True
    if enabled and opt_en:
        return torch.utils.checkpoint.checkpoint(fn, *args)
    else:
        return fn(*args)


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, all_missing_keys, unexpected_keys, err_msg
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append("unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_state_dict_submodule(module, state_dict, submodule_name="", strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    submodule_state_dict = OrderedDict()
    # state_dict = state_dict.copy()
    for k, v in state_dict.items():
        if k.startswith(submodule_name):
            # in python3.9 we can use k.removeprefix(submodule_name)
            submodule_state_dict[k[len(submodule_name) :]] = v

    if metadata is not None:
        submodule_state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            submodule_state_dict, prefix, local_metadata, True, all_missing_keys, unexpected_keys, err_msg
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append("unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_checkpoint(
    model,
    filename,
    map_location=None,
    strict=False,
    hack_dict=None,
    logger=None,
    only_model=False,
    prefix="module.",
):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch_load(filename, map_location=map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith(prefix):
        state_dict = {k[len(prefix) :]: v for k, v in checkpoint["state_dict"].items()}

    if hack_dict is not None:
        for pattern, new_value in hack_dict.items():
            state_dict = {re.sub(pattern, new_value, k): v for k, v in checkpoint["state_dict"].items()}

    # load state_dict
    if only_model:
        load_state_dict(model, state_dict, strict, logger)
    else:
        load_state_dict(model.state_module(), state_dict, strict, logger)
    return checkpoint
