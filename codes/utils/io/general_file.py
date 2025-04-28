import io
import os
import os.path as osp
import re

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.io import wavfile
from torchvision.utils import save_image

from .fileio import PathManager
from .mfs import GFile, isdir, listdir

_torch_save = torch.save
_torch_load = torch.load


def gfs(mode):
    def decorator(decorated):
        def wrapper(*args, **kwargs):
            if len(args) > 0:
                raise "positional arguments is forbidden"
            elif "path" not in (kwargs):
                raise "`path` argument is compulsory"
            gfs_path = kwargs["path"]
            if mode == "w":
                with GFile(gfs_path, "wb") as f:
                    vio = io.BytesIO()
                    kwargs["path"] = vio
                    result = decorated(**kwargs)
                    f.write(vio.read())
                    vio.close()
                    return result
            elif mode == "r":
                with GFile(gfs_path, "rb") as f:
                    data = f.read()
                    vio = io.BytesIO(data)
                    kwargs["path"] = vio
                    result = decorated(**kwargs)
                    vio.close()
                    return result
            else:
                raise "Other mode is not supported"

        return wrapper

    return decorator


@gfs(mode="r")
def wav_load(path, sr=None, mono=False):
    data, samplerate = sf.read(path)
    signed_int16_max = 2**15
    if data.dtype == np.int16:
        data = data.astype(np.float32) / signed_int16_max
    if sr is not None:
        if sr != samplerate:
            data = librosa.resample(data, samplerate, sr)
    if mono:
        data = librosa.to_mono(data)
    data = np.clip(data, -1.0, 1.0)
    return data


@gfs(mode="w")
def wav_save(path, sr, wav):
    wav = (wav * 32767) / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


def image_save(tensor, path, **kwargs):
    fn = osp.basename(path)
    _format = osp.splitext(fn)[1][1:]
    with GFile(path, "wb") as f:
        save_image(tensor, f, format=_format, **kwargs)


@gfs(mode="r")
def np_load(path, dtype=np.float32):
    data = np.load(path)
    data = data.astype(dtype)
    return data


def np_save(path, arr, **kwargs):
    with GFile(path, "wb") as f:
        np.save(f, arr, **kwargs)


def torch_load(path, map_location=torch.device("cpu"), default_restore_dir=None, endswith=(".pth.tar", ".pth")):
    restore = path if path != "" and path is not None else default_restore_dir
    if not isinstance(endswith, (tuple, list)):
        endswith = [endswith]
    if restore is None:
        return None
    if isdir(restore):
        file_list = listdir(restore)
        last_step = -1
        last_file_name = None
        for _file_ in file_list:
            for suffix in endswith:
                if _file_.endswith(suffix):
                    step_str_list = re.findall(r"\d+", _file_)
                    if len(step_str_list) > 0:
                        step = int(step_str_list[-1])
                        if last_step < step:
                            last_step = step
                            last_file_name = _file_
        if last_file_name is None:
            if path.endswith("checkpoints"):
                return None
            else:
                new_path = osp.join(path, "checkpoints")
                return torch_load(path=new_path, map_location=map_location, default_restore_dir=None, endswith=endswith)
        else:
            restore = os.path.join(restore, last_file_name)

    with GFile(restore, "rb") as f:
        vio = io.BytesIO(f.read())
        checkpoint = _torch_load(vio, map_location=map_location, weights_only=True)
        vio.close()
    return checkpoint


def torch_save(dicts, checkpoint_path, async_file=False):
    if async_file:
        with PathManager.opena(f"{checkpoint_path}", "wb") as f:
            vio = io.BytesIO()
            _torch_save(dicts, vio)
            f.write(vio.getvalue())
            vio.close()
    else:
        with GFile(f"{checkpoint_path}", "wb") as f:
            vio = io.BytesIO()
            _torch_save(dicts, vio)
            f.write(vio.getvalue())
            vio.close()
