import collections
import collections.abc
import itertools
import numbers
from collections import abc
from itertools import repeat

import numpy as np
import torch

# From PyTorch internals


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def is_str(x):
    """Whether the input is an string instance.
    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def iter_cast(inputs, dst_type, return_type=None):
    """Cast elements of an iterable object into some type.
    Args:
        inputs (Iterable): The input object.
        dst_type (type): Destination type.
        return_type (type, optional): If specified, the output object will be
            converted to this type, otherwise an iterator.
    Returns:
        iterator or specified type: The converted object.
    """
    if not isinstance(inputs, abc.Iterable):
        raise TypeError("inputs must be an iterable object")
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')

    out_iterable = map(dst_type, inputs)

    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)


def list_cast(inputs, dst_type):
    """Cast elements of an iterable object into a list of some type.
    A partial method of :func:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=list)


def tuple_cast(inputs, dst_type):
    """Cast elements of an iterable object into a tuple of some type.
    A partial method of :func:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=tuple)


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.
    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.
    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def is_dict_of_dict(inputs_dict):
    is_dict_of_dict = True
    for key, input_cfg in inputs_dict.items():
        if not isinstance(input_cfg, dict):
            is_dict_of_dict = False
            break
    return is_dict_of_dict


def slice_list(in_list, lens):
    """Slice a list into several sub lists by a list of given length.
    Args:
        in_list (list): The list to be sliced.
        lens(int or list): The expected length of each out list.
    Returns:
        list: A list of sliced list.
    """
    if isinstance(lens, int):
        assert len(in_list) % lens == 0
        lens = [lens] * int(len(in_list) / lens)
    if not isinstance(lens, list):
        raise TypeError('"indices" must be an integer or a list of integers')
    elif sum(lens) != len(in_list):
        raise ValueError("sum of lens and list length does not " f"match: {sum(lens)} != {len(in_list)}")
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx : idx + lens[i]])
        idx += lens[i]
    return out_list


def concat_list(in_list):
    """Concatenate a list of list into a single list.
    Args:
        in_list (list): The list of list to be merged.
    Returns:
        list: The concatenated flat list.
    """
    return list(itertools.chain(*in_list))


def index_list(in_list, indices):
    out_list = [in_list[idx] for idx in indices]
    return out_list


class DictClass(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        self[attr] = value


def is_scalar(val, include_np=True, include_torch=True):
    """Tell the input variable is a scalar or not.

    Args:
        val: Input variable.
        include_np (bool): Whether include 0-d np.ndarray as a scalar.
        include_torch (bool): Whether include 0-d torch.Tensor as a scalar.

    Returns:
        bool: True or False.
    """
    if isinstance(val, numbers.Number):
        return True
    elif include_np and isinstance(val, np.ndarray) and val.ndim == 0:
        return True
    elif include_torch and isinstance(val, torch.Tensor) and ((val.dim() > 0 and val.numel() == 1) or (val.dim() == 0)):
        return True
    else:
        return False
