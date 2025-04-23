import importlib
import inspect
import re
from functools import partial
from multiprocessing import cpu_count, get_context
from typing import Any, Callable, Dict, List, Tuple

from utils import logging

logger = logging.getLogger("base")


def get_func_from_name(process_func):
    mod_name, func_name = process_func.rsplit(".", 1)
    try:
        mod = importlib.import_module(mod_name)
        cls_obj = None
    except ModuleNotFoundError:
        mod_name, cls_name = mod_name.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        cls_obj = getattr(mod, cls_name)
    return mod, cls_obj, func_name


def get_enc_func_from_name(process_func, default_module=["datasets", "dataset_ext"]):
    mod_name, func_name = process_func.rsplit(".", 1)
    pass


def get_whole_line(lines, starts="def", ends="):"):
    whole_line = ""
    start = -1
    end = -1
    for i, line in enumerate(lines):
        line = re.sub(r"[\n\t\s]*", "", line)
        if line.startswith(starts):
            start = i
        if start >= 0:
            whole_line += line
            if line.endswith(ends):
                end = i
                break
    return whole_line, (start, end)


def parse_params(whole_line):
    params_re = re.search(r"\((.)*\)", whole_line)
    if params_re:
        params_str = params_re.group()[1:-1]
        params = params_str.split(",")
        if "" in params:
            params.remove("")
    else:
        params = None

    return params


def parse_inout_params(func):
    source_code_line, _ = inspect.getsourcelines(func)
    input_whole_line, _ = get_whole_line(source_code_line)
    input_params = parse_params(input_whole_line)
    output_whole_line, _ = get_whole_line(source_code_line, starts="return", ends=")")
    output_params = parse_params(output_whole_line)
    return input_params, output_params


def func_generators(func, options, cfg):
    for options_ in zip(*options):
        results = func(options=options_, cfg=cfg)
        yield results


def naive_func_generators(func, data, cfg):
    for data_ in data:
        results = func(data_, cfg=cfg)
        yield results


def mp_executor(
    pool,
    func: Callable,
    options: Tuple[List, ...],
    cfg: Dict[str, Any],
    mp: bool = False,
    chunksize: int = 1,
    jobs: int = 1,
    ordered: bool = False,
):
    result_list = []
    if mp:
        items = len(options[0])
        if pool is None:
            max_jobs = 8  # cpu_count() // 2
            if jobs <= 0 or jobs > max_jobs:
                jobs = max_jobs
            if jobs > items:
                jobs = items

            if chunksize <= 0:
                if items % jobs == 0:
                    chunksize = items // jobs
                else:
                    chunksize = items // jobs + 1
            pool = get_context("spawn").Pool(max_jobs)
            local_pool = True
        else:
            if chunksize <= 0:
                jobs = pool._processes
                if items % jobs == 0:
                    chunksize = items // jobs
                else:
                    chunksize = items // jobs + 1
            local_pool = False
        mp_imap = pool.imap if ordered else pool.imap_unordered
        for info_metas in mp_imap(partial(func, cfg=cfg), zip(*options), chunksize=chunksize):
            info, metas = info_metas
            if metas is None:
                if info is None:
                    logger.warning("executor exception")
                else:
                    logger.warning(f"executor exception at {info}")
            else:
                # iterable
                result_list.append(metas)
        if local_pool:
            pool.close()
    else:
        for info_metas in func_generators(func, options, cfg):
            info, metas = info_metas
            if metas is None:
                if info is None:
                    logger.warning("executor exception")
                else:
                    logger.warning(f"executor exception at {info}")
            else:
                # iterable
                result_list.append(metas)

    return result_list


def naive_mp_executor(
    pool,
    func: Callable,
    data: List,
    cfg: Dict[str, Any],
    mp: bool = False,
    chunksize: int = 1,
    jobs: int = 1,
    ordered: bool = False,
):
    result_list = []
    if mp:
        items = len(data)
        if pool is None:
            max_jobs = cpu_count() // 2
            if jobs <= 0 or jobs > max_jobs:
                jobs = max_jobs
            if jobs > items:
                jobs = items

            if chunksize <= 0:
                if items % jobs == 0:
                    chunksize = items // jobs
                else:
                    chunksize = items // jobs + 1
            pool = get_context("spawn").Pool(max_jobs)
            local_pool = True
        else:
            if chunksize <= 0:
                jobs = pool._processes
                if items % jobs == 0:
                    chunksize = items // jobs
                else:
                    chunksize = items // jobs + 1
            local_pool = False
        mp_imap = pool.imap if ordered else pool.imap_unordered
        for info_metas in mp_imap(partial(func, cfg=cfg), data, chunksize=chunksize):
            info = None
            metas = info_metas
            if metas is None:
                if info is None:
                    logger.warning("executor exception")
                else:
                    logger.warning(f"executor exception at {info}")
            else:
                # iterable
                result_list.append(metas)
        if local_pool:
            pool.close()
    else:
        for info_metas in naive_func_generators(func, data, cfg):
            info = None
            metas = info_metas
            if metas is None:
                if info is None:
                    logger.warning("executor exception")
                else:
                    logger.warning(f"executor exception at {info}")
            else:
                # iterable
                result_list.append(metas)

    return result_list
