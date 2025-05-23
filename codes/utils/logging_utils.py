import os

import torch.distributed as dist
from utils import logging

from .misc import get_timestamp

logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    if name in logger_initialized:
        return logger_initialized[name]

    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger_initialized[logger_name]

    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, "w")
        handlers.append(file_handler)

    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = logger

    return logger


def print_log(msg, logger=None, level=logging.INFO, flush_right_now=True):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg, flush_right_now=flush_right_now)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            "logger should be either a logging.Logger object, str, " f'"silent" or None, but got {type(logger)}'
        )


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name="base", log_file=log_file, log_level=log_level)

    return logger


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    lg.setLevel(level)

    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(get_timestamp(), phase))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

    logger_initialized[logger_name] = lg
