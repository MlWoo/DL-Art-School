import logging
from itertools import islice

import torch


def split_with_n(iterable, n):
    i = iter(iterable)
    piece = islice(i, n)
    return piece


def split_list(lst, n, keep_tail=True):
    if keep_tail:
        return [lst[i : i + n] for i in range(0, len(lst), n)]
    else:
        k = len(lst) // n  # 计算完整子列表的数量
        return [lst[i * n : (i + 1) * n] for i in range(k)]


def chunk(iterable, chunk_size, rank=0):
    ret = []
    i = 0
    for record in iterable:
        if record is None or record is False:
            print("invalid", i, record)
        else:
            if isinstance(record, torch.Tensor):
                record = record[rank].tolist()
            else:
                if isinstance(record, (list, tuple)) and isinstance(record[0], torch.Tensor):
                    record = record[rank].tolist()
                else:
                    assert rank == 1, "rank should be 0 when record is not a tensor"
            ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
        i += 1
    if ret:
        yield ret


def chunk_warmup(iterable, chunk_size, magnification, warm_factor=2):
    ret = []
    i = 0
    for record in iterable:
        if len(ret) == 0:
            _chunk_size = min(warm_factor + i, chunk_size)
        if record is None or record is False:
            logging.warning("invalid", i, record)
        else:
            ret.append(record)

        if len(ret) == _chunk_size * magnification:
            yield ret
            ret = []
            i += _chunk_size
    if ret:
        yield ret


def find_surrounding_elements(boundaries, value):
    left = 0
    right = len(boundaries) - 1

    # 处理特殊情况
    if not boundaries:
        return (None, None)
    if value < boundaries[0]:
        return (None, boundaries[0])
    if value > boundaries[-1]:
        return (boundaries[-1], None)

    # 二分查找
    while left <= right:
        mid = (left + right) // 2
        if boundaries[mid] == value:
            return (boundaries[mid], boundaries[mid])
        elif boundaries[mid] < value:
            left = mid + 1
        else:
            right = mid - 1

    # 此时 left 是第一个大于 value 的元素的索引
    # right 是最后一个小于 value 的元素的索引
    return (boundaries[right], boundaries[left])
