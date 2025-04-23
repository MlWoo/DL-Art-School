# Copyright (c) Open-MMLab. All rights reserved.
import logging
import os
import os.path as osp
from pathlib import Path

import paramiko
import scp

from .io import mfs
from .misc import get_timestamp
from .xtype import is_str


def is_filepath(x):
    return is_str(x) or isinstance(x, Path)


def fopen(filepath, *args, **kwargs):
    if is_str(filepath):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)
    raise ValueError("`filepath` should be a string or a Path")


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not mfs.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name):
    if dir_name == "" or osp.exists(dir_name):
        return

    dir_name = osp.expanduser(dir_name)
    mfs.makedirs(dir_name, exist_ok=True)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir_or_exist(paths)
    else:
        for path in paths:
            mkdir_or_exist(path)


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def scandir(dir_path, suffix=None, recursive=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def find_vcs_root(path, markers=(".git",)):
    """Finds the root directory (including itself) of specified markers.

    Args:
        path (str): Path of directory or file.
        markers (list[str], optional): List of file or directory names.

    Returns:
        The directory contained one of the markers or None if not found.
    """
    if osp.isfile(path):
        path = osp.dirname(path)

    prev, cur = None, osp.abspath(osp.expanduser(path))
    while cur != prev:
        if any(osp.exists(osp.join(cur, marker)) for marker in markers):
            return cur
        prev, cur = cur, osp.split(cur)[0]
    return None


def list_dir(path, extension=None, sort=True):
    def remove_self_and_parent(file_dir_list):
        if "." in file_dir_list:
            file_dir_list.remove(".")
        if ".." in file_dir_list:
            file_dir_list.remove("..")

    fn_list = mfs.listdir(path)
    remove_self_and_parent(fn_list)
    if extension is None:
        return fn_list
    new_fn_list = []
    if isinstance(extension, str):
        for fn in fn_list:
            if fn.endswith(extension):
                new_fn_list.append(fn)
    elif isinstance(extension, (tuple, list)):
        for fn in fn_list:
            fn_ext = osp.splitext(fn)[1]
            if fn_ext in extension:
                new_fn_list.append(fn)
    if sort:
        new_fn_list = sorted(new_fn_list)
    return new_fn_list


def copy_files_to_server(host, user, password, files, remote_path):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password)
    scpclient = scp.SCPClient(client.get_transport())
    scpclient.put(files, remote_path)


def get_files_from_server(host, user, password, remote_path, local_path):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password)
    scpclient = scp.SCPClient(client.get_transport())
    scpclient.get(remote_path, local_path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + "_archived_" + get_timestamp()
        print("Path already exists. Rename it to [{:s}]".format(new_name))
        logger = logging.getLogger("base")
        logger.info("Path already exists. Rename it to [{:s}]".format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)
