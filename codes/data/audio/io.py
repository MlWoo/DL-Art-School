import json
import math
import os
import os.path as osp
import re
import time
from subprocess import PIPE, run

import av
import numpy as np
import pandas as pd
import s3fs
import soundfile as sf
import soxr
import torch


def get_duration_sec(file, cache=False):
    try:
        with open(file + ".dur", "r") as f:
            duration = float(f.readline().strip("\n"))
        return duration
    except:  # noqa: E722
        container = av.open(file)
        audio = container.streams.get(audio=0)[0]
        duration = audio.duration * float(audio.time_base)
        if cache:
            with open(file + ".dur", "w") as f:
                f.write(str(duration) + "\n")
        return duration


def read_audio_section(file, sr, offset, duration, resample=False, time_base="samples"):
    if time_base == "sec":
        offset = int(offset * sr)
        duration = int(duration * sr)
    if file.endswith("opus") or file.endswith("m4a"):
        if file.startswith("s3://"):
            if "PROFILE" in os.environ:
                profile = os.environ["PROFILE"]
            else:
                profile = "b-yarn"
            if "S3_URL" in os.environ:
                S3_URL = os.environ["S3_URL"]
            else:
                S3_URL = None
            storage_options = {"profile": profile, "endpoint_url": S3_URL}
            s3fs_ins = s3fs.S3FileSystem(**storage_options)
            file_handler = s3fs_ins.open(file, "rb")
            container = av.open(file_handler, buffer_size=4096)
        else:
            try:
                container = av.open(file)
            except:  # noqa: E722
                return None, None
        audio = container.streams.get(audio=0)[0]  # Only first audio stream
        audio_duration = audio.duration * float(audio.time_base)
        ori_sr = audio.sample_rate
        audio_duration_sr = int(math.ceil(audio_duration * sr))
        if offset + duration > audio_duration_sr:
            # Move back one window. Cap at audio_duration
            offset = np.min(audio_duration_sr - duration, offset - duration)

        ori_duration = int(ori_sr * (duration / sr))

        sig = np.zeros((ori_duration,), dtype=np.float32)

        container.seek(int(offset * ori_sr), stream=audio)
        total_read = 0

        for frame in container.decode(audio=0):  # Only first audio stream
            frame = frame.to_ndarray()  # Convert to floats and not int16
            # Get chunk
            read = frame.shape[-1]
            if total_read + read > ori_duration:
                read = ori_duration - total_read
            sig[total_read : total_read + read] = frame[0, :read]
            total_read += read
            if total_read == ori_duration:
                break
    else:
        if not osp.exists(file):
            print(f"No such file or directory: {file}")
            return None, None
        track = sf.SoundFile(file)
        ori_sr = track.samplerate
        channels = track.channels
        start_frame = offset
        frames_to_read = int(ori_sr * (duration / sr))
        track.seek(start_frame)
        audio_section = track.read(frames_to_read, dtype="float32")
        if channels == 1:
            sig = audio_section
        else:
            sig = audio_section[:, 0]

    if ori_sr == sr:
        rs = sig
    else:
        rs = soxr.resample(sig, ori_sr, sr, quality="HQ")  # input samplerate  # target samplerate
    rs = rs.reshape(1, -1)
    return rs, sr


def load_audio(file, sr, offset, duration=-1, resample=True, approx=True, time_base="samples", check_duration=True):
    if time_base == "sec":
        offset = int(math.ceil(offset * sr))
        duration = int(math.ceil(duration * sr))
    # Loads at target sr, stereo channels, seeks from offset, and stops after duration

    try:
        if file.startswith("s3://"):
            if "PROFILE" in os.environ:
                profile = os.environ["PROFILE"]
            else:
                profile = "b-yarn"
            if "S3_URL" in os.environ:
                S3_URL = os.environ["S3_URL"]
            else:
                S3_URL = None
            storage_options = {"profile": profile, "endpoint_url": S3_URL}
            start_time = time.time()
            s3fs_ins = s3fs.S3FileSystem(**storage_options)
            end_time = time.time()
            print(f"init s3fs time: {end_time - start_time}")
            start_time = time.time()
            file = s3fs_ins.open(file, "rb")
            end_time = time.time()
            print(f"open file time: {end_time - start_time}")
        container = av.open(file)
    except:  # noqa: E722
        return None, None

    audio = container.streams.get(audio=0)[0]  # Only first audio stream
    try:
        audio_duration = audio.duration * float(audio.time_base)
    except:  # noqa: E722
        return None, None

    ori_sr = audio.sample_rate
    audio_duration_sr = int(math.ceil(audio_duration * sr))
    if duration < 0:
        duration = audio_duration_sr

    if approx:
        if offset + duration > audio_duration_sr:
            # Move back one window. Cap at audio_duration
            offset = min(audio_duration_sr - duration, offset - duration)
        offset = max(0, offset)
    else:
        if check_duration:
            try:
                assert (
                    offset + duration <= audio_duration_sr
                ), f"End {offset + duration} beyond duration {audio_duration_sr}"
            except:  # noqa: E722
                return None, None

    if resample or sr != ori_sr:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
        """
        global resampler_dict
        resampler_key = f"{mono}_{dtype}_{ori_sr}"
        resampler = resampler_dict.get(resampler_key, None)
        if resampler is None:
            resampler = av.AudioResampler(format='fltp',layout='mono', rate=sr)
            resampler_dict[resampler_key] = resampler
            with open("./resampler.txt", "a+", encoding="utf-8") as f:
                f.write(f"file: {file} {mono}_{dtype}_{ori_sr}  |  mono: {mono}  |  ori_sr : {ori_sr}\n")
        """
    else:
        assert sr == ori_sr
    offset = int(
        offset / sr / float(audio.time_base)
    )  # int(offset / float(audio.time_base)) # Use units of time_base for seeking
    duration = int(duration)  # duration = int(duration * sr) # Use units of time_out ie 1/sr for returning
    sig = np.zeros((1, duration), dtype=np.float32)

    total_read = 0
    try:
        container.seek(offset, stream=audio)
        if resample or sr != ori_sr:
            for frame in container.decode(audio=0):  # Only first audio stream
                frame.pts = None
                re_frame = resampler.resample(frame)
                frame = re_frame[0]
                frame = frame.to_ndarray()  # Convert to floats and not int16
                read = frame.shape[-1]

                if total_read + read > duration:
                    read = duration - total_read
                sig[:, total_read : total_read + read] = frame[:1, :read]
                total_read += read
                if total_read == duration:
                    break
        else:
            for frame in container.decode(audio=0):  # Only first audio stream
                frame = frame.to_ndarray()  # Convert to floats and not int16
                read = frame.shape[-1]
                if total_read + read > duration:
                    read = duration - total_read
                sig[:, total_read : total_read + read] = frame[:1, :read]
                total_read += read
                if total_read == duration:
                    break
    except:  # noqa: E722
        return None, None

    assert total_read <= duration, f"Expected {duration} frames, got {total_read}"

    return sig, sr


def load_audiox(file, sr, offset, duration, resample=True, approx=False, time_base="samples", check_duration=True):
    file = re.sub(
        r"/mnt/shared-storage/groups/tts/fcl/zh_data/gennis/raw_data/",
        "/mnt/shared-storage/groups/tts/fcl/zh_data/gennis/raw_data/wav/",
        file,
    )
    # Loads at target sr, stereo channels, seeks from offset, and stops after duration
    try:
        container = av.open(file)
    except:  # noqa: E722
        return None, None
    audio = container.streams.get(audio=0)[0]  # Only first audio stream
    audio_duration = audio.duration * float(audio.time_base)
    ori_sr = audio.sample_rate
    mono = audio.layout.name
    dtype = audio.format.name
    audio_duration_sr = int(math.ceil(audio_duration * sr))

    if offset + duration > audio_duration_sr:
        # Move back one window. Cap at audio_duration
        offset = np.min(audio_duration_sr - duration, offset - duration)

    if resample or sr != ori_sr:
        global resampler_dict
        resampler_key = f"{mono}_{dtype}_{ori_sr}"
        resampler = resampler_dict.get(resampler_key, None)
        if resampler is None:
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
            resampler_dict[resampler_key] = resampler
            with open("./resampler.txt", "a+", encoding="utf-8") as f:
                f.write(f"file: {file} {mono}_{dtype}_{ori_sr}  |  mono: {mono}  |  ori_sr : {ori_sr}\n")
    else:
        assert sr == ori_sr

    offset = int(
        offset / sr / float(audio.time_base)
    )  # int(offset / float(audio.time_base)) # Use units of time_base for seeking
    duration = int(duration)  # duration = int(duration * sr) # Use units of time_out ie 1/sr for returning

    sig = np.zeros((1, duration), dtype=np.float32)
    container.seek(offset, stream=audio)
    total_read = 0
    try:
        if resample or sr != ori_sr:
            for frame in container.decode(audio=0):  # Only first audio stream
                frame.pts = None
                try:
                    re_frame = resampler.resample(frame)
                except:  # noqa: E722
                    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
                    with open("./error_file.txt", "a+", encoding="utf-8") as f:
                        f.write(f"file: {file}   |  mono: {mono}  |  ori_sr : {ori_sr}\n")
                    return None, None

                frame = re_frame[0]
                frame = frame.to_ndarray(format="fltp")  # Convert to floats and not int16
                read = frame.shape[-1]
                if total_read + read > duration:
                    read = duration - total_read
                sig[:, total_read : total_read + read] = frame[:1, :read]
                total_read += read
                if total_read == duration:
                    break
        else:
            for frame in container.decode(audio=0):  # Only first audio stream
                frame = frame.to_ndarray(format="fltp")  # Convert to floats and not int16
                read = frame.shape[-1]
                if total_read + read > duration:
                    read = duration - total_read
                sig[:, total_read : total_read + read] = frame[:1, :read]
                total_read += read
                if total_read == duration:
                    break
    except:  # noqa: E722
        return None, None

    assert total_read <= duration, f"Expected {duration} frames, got {total_read}"

    return sig, sr


def load_audio_segment(
    file, container, sr, offset, duration, resample=True, approx=False, time_base="samples", check_duration=True
):
    if time_base == "sec":
        offset = int(math.ceil(offset * sr))
        duration = int(math.ceil(duration * sr))
    # Loads at target sr, stereo channels, seeks from offset, and stops after duration
    audio = container.streams.get(audio=0)[0]  # Only first audio stream
    audio_duration = audio.duration * float(audio.time_base)
    ori_sr = audio.sample_rate
    mono = audio.layout.name
    dtype = audio.format.name
    audio_duration_sr = int(math.ceil(audio_duration * sr))
    if approx:
        if offset + duration > audio_duration_sr:
            # Move back one window. Cap at audio_duration
            offset = np.min(audio_duration_sr - duration, offset - duration)
    else:
        if check_duration:
            try:
                assert (
                    offset + duration <= audio_duration_sr
                ), f"End {offset + duration} beyond duration {audio_duration_sr}"
            except:  # noqa: E722
                return None, None

    if resample or sr != ori_sr:
        global resampler_dict
        resampler_key = f"{mono}_{dtype}_{ori_sr}"
        resampler = resampler_dict.get(resampler_key, None)
        if resampler is None:
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
            resampler_dict[resampler_key] = resampler
    else:
        assert sr == ori_sr
    offset = int(
        offset / sr / float(audio.time_base)
    )  # int(offset / float(audio.time_base)) # Use units of time_base for seeking
    duration = int(duration)  # duration = int(duration * sr) # Use units of time_out ie 1/sr for returning
    sig = np.zeros((1, duration), dtype=np.float32)
    container.seek(offset, stream=audio)
    total_read = 0
    if resample or sr != ori_sr:
        for frame in container.decode(audio=0):  # Only first audio stream
            frame.pts = None
            try:
                re_frame = resampler.resample(frame)
            except:  # noqa: E722
                return None, None

            frame = re_frame[0]
            frame = frame.to_ndarray(format="fltp")  # Convert to floats and not int16
            read = frame.shape[-1]
            if total_read + read > duration:
                read = duration - total_read
            sig[:, total_read : total_read + read] = frame[:, :read]
            total_read += read
            if total_read == duration:
                break
    else:
        for frame in container.decode(audio=0):  # Only first audio stream
            frame = frame.to_ndarray(format="fltp")  # Convert to floats and not int16
            read = frame.shape[-1]
            if total_read + read > duration:
                read = duration - total_read
            sig[:, total_read : total_read + read] = frame[:, :read]
            total_read += read
            if total_read == duration:
                break
    assert total_read <= duration, f"Expected {duration} frames, got {total_read}"
    return sig, sr


def test_simple_loader():
    import librosa
    from tqdm import tqdm

    def collate_fn(batch):
        return torch.stack([torch.from_numpy(b) for b in batch], dim=0)

    def get_batch(file, loader):
        y1, sr = loader(file, sr=44100, offset=0.0, duration=6.0, time_base="sec")
        y2, sr = loader(file, sr=44100, offset=20.0, duration=6.0, time_base="sec")
        return [y1, y2]

    def load(file, loader):
        batch = get_batch(file, loader)  # np
        x = collate_fn(batch)  # torch cpu
        x = x.to("cuda", non_blocking=True)  # torch gpu
        return x

    files = librosa.util.find_files("/root/data/", ["mp3", "m4a", "opus"])
    print(files[:10])
    loader = load_audio
    print("Loader", loader.__name__)
    for i, file in enumerate(tqdm(files)):
        _ = load(file, loader)
        if i == 100:
            break


def store_panda(meta_dict):
    df = pd.DataFrame(meta_dict)

    df.to_parquet(
        "test.parquet.gz",
        # 需要 pip install pyarrow
        engine="pyarrow",
        # 压缩方式，可选择：'snappy', 'gzip', 'brotli', None
        # 默认是 'snappy'
        compression="gzip",
        # 是否把 DataFrame 自带的索引写进去，默认写入
        # 但要注意的是，索引会以 range 对象的形式写入到元数据中
        # 因此不会占用太多空间，并且速度还更快
        index=True,
    )


def parse_channel_from_ffmpeg_output(ffmpeg_stderr: bytes) -> str:
    # ffmpeg will output line such as the following, amongst others:
    # "Stream #0:0: Audio: pcm_f32le, 16000 Hz, mono, flt, 512 kb/s"
    # but sometimes it can be "Stream #0:0(eng):", which we handle with regexp
    # pattern = re.compile(r"^\s*Stream #0:0.*: Audio: pcm_f32le.+(mono|stereo).+\s*$")
    pattern = re.compile(r"^\s*Stream #0:0.*: Audio: pcm_s16le.+(mono|stereo).+\s*$")
    for line in ffmpeg_stderr.splitlines():
        try:
            line = line.decode()
        except UnicodeDecodeError:
            # Why can we get UnicodeDecoderError from ffmpeg output?
            # Because some files may contain the metadata, including a short description of the recording,
            # which may be encoded in arbitrarily encoding different than ASCII/UTF-8, such as latin-1,
            # and Python will not automatically recognize that.
            # We simply ignore these lines as they won't have any relevant information for us.
            continue
        match = pattern.match(line)
        if match is not None:
            return match.group(1)
    raise ValueError(
        f"Could not determine the number of channels for OPUS file from the following ffmpeg output "
        f"(shown as bytestring due to avoid possible encoding issues):\n{str(ffmpeg_stderr)}"
    )


def read_opus_ffmpeg(
    path: str,
    offset=0.0,
    duration=None,
    force_opus_sampling_rate=None,
):
    """
    Reads OPUS files using ffmpeg in a shell subprocess.
    Unlike audioread, correctly supports offsets and durations for reading short chunks.
    Optionally, we can force ffmpeg to resample to the true sampling rate (if we know it up-front).

    :return: a tuple of audio samples and the sampling rate.
    """
    # Construct the ffmpeg command depending on the arguments passed.
    cmd = "ffmpeg -threads 1"
    sampling_rate = 48000
    # Note: we have to add offset and duration options (-ss and -t) BEFORE specifying the input
    #       (-i), otherwise ffmpeg will decode everything and trim afterwards...
    if offset > 0:
        cmd += f" -ss {offset}"
    if duration is not None:
        cmd += f" -t {duration}"
    # Add the input specifier after offset and duration.
    if "'" in path:
        cmd += f' -i "{path}"'
    else:
        cmd += f" -i '{path}'"
    # Optionally resample the output.
    if force_opus_sampling_rate is not None:
        cmd += f" -ar {force_opus_sampling_rate}"
        sampling_rate = force_opus_sampling_rate
    # Read audio samples directly as float32.
    # cmd += " -f f32le -threads 1 pipe:1"
    cmd += " -f s16le -acodec pcm_s16le -threads 1 pipe:1"
    # Actual audio reading.
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    raw_audio = proc.stdout
    # audio = np.frombuffer(raw_audio, dtype=np.float32)
    audio = np.frombuffer(raw_audio, np.int16).astype(np.float32) / 32768.0
    # Determine if the recording is mono or stereo and decode accordingly.
    try:
        channel_string = parse_channel_from_ffmpeg_output(proc.stderr)
        if channel_string == "stereo":
            new_audio = np.empty((2, audio.shape[0] // 2), dtype=np.float32)
            new_audio[0, :] = audio[::2]
            new_audio[1, :] = audio[1::2]
            audio = new_audio
        elif channel_string == "mono":
            audio = audio.reshape(1, -1)
        else:
            raise NotImplementedError(f"Unknown channel description from ffmpeg: {channel_string}")
    except ValueError as e:
        raise f"{e}\nThe ffmpeg command for which the program failed is: '{cmd}', error code: {proc.returncode}"
    return audio, sampling_rate


if __name__ == "__main__":
    # from jukebox.utils.dist_utils import setup_dist_from_mpi
    # setup_dist_from_mpi(port=29500)
    # test_dataset_loader()
    fp = "/bilibili/2020_uploader/历史调研室_519872016/士为知己者死！文在寅的复仇之路_BV1hK4y1b7TJ/audio.wav"
    # fp = '/mnt/shared-storage/groups/tts/wenetspeech_denoised/audio/train/youtube/B00078/Y0000020044_ppc0KeOTKyQ.opus'
    fp = "s3://bilibili/我是兜兜飞_386044336/6296d9dbc3cf61c786e60c27ce9bbcd8_BV1tm411y7vA/BV1tm411y7vA_audio.m4s"
    # fp = '/data/tts-data/ZH/ningyuguang/ningyuguang-zc/音频/hls-zc-nyg-1.wav'
    # start_time = time()
    # sig, sr = read_audio_section(fp, sr=16000, offset=1.5, duration=2.0, time_base='sec')
    # print(time() - start_time)
    fp = "s3://ytb/UCcRmQGLg1zx6SQFsoNBHHtQ/NlUeVqrfyNQ/NlUeVqrfyNQ.opus"

    start_time = time.time()
    # read_opus_ffmpeg(fp, offset=1.5, duration=2.0, force_opus_sampling_rate=16000)
    # sig, sr = load_audio(fp, sr=16000, offset=120.8, duration=1600000, resample=True)  # , time_base='sec')
    storage_options = {}
    s3fs_ins = s3fs.S3FileSystem(**storage_options)
    # fp = "/home/wumenglin/data/test.m4a"
    start_time = time.time()
    print("======= long opus 1 hour =========")
    # s3fs_ins.download(fp, "temp.opus")
    # fp = "temp.opus"
    sig, sr = read_audio_section(
        fp, sr=16000, offset=3350.8, duration=120, resample=True, time_base="sec"
    )  # , time_base='sec')
    # import torchaudio
    # torchaudio.save(torch.from_numpy(sig), "temp.wav", sample_rate=24000)

    print("sig shape", sig.shape)
    print("opus s3 time cost", time.time() - start_time)
    fp = "s3://ytb/UCh0eq9O0BXwfEdXMpbQfbbA/eQQ8lB5jInc/eQQ8lB5jInc.opus"
    start_time = time.time()
    print("======= short opus 6min =========")
    sig, sr = read_audio_section(
        fp, sr=16000, offset=50.8, duration=120, resample=True, time_base="sec"
    )  # , time_base='sec')
    print("sig shape", sig.shape)
    print("opus s3 time cost", time.time() - start_time)

    print("======= long m4a 1 hour =========")
    start_time = time.time()
    fp = "s3://ytb/UCh0eq9O0BXwfEdXMpbQfbbA/eQQ8lB5jInc/eQQ8lB5jInc.m4a"
    sig, sr = read_audio_section(
        fp, sr=16000, offset=10.8, duration=120, resample=True, time_base="sec"
    )  # , time_base='sec')
    print("sig shape", sig.shape)
    print("m4a s3 time cost", time.time() - start_time)

    print("======= short m4a 6min =========")
    start_time = time.time()
    fp = "s3://ytb/KhanAcademy/yeiroTpHFmg/yeiroTpHFmg.m4a"
    sig, sr = read_audio_section(
        fp, sr=16000, offset=10.8, duration=120, resample=True, time_base="sec"
    )  # , time_base='sec')
    print("sig shape", sig.shape)
    print("m4a s3 time cost", time.time() - start_time)

    exit(0)

    meta_json = "/bilibili/bilibili-20231124/metadata_concatenated_step3.json"
    meta_list = []
    paths = []
    begin_times = []
    durations = []
    texts = []
    with open(meta_json, "r") as f:
        for line in f:
            meta_subs = json.loads(line)
            audios = meta_subs["audios"]
            for audio in audios:
                path = audio["path"]
                segments = audio["segments"]
                for segment in segments:
                    begin_time = segment["begin_time"]
                    duration = segment["duration"]
                    text = segment["text"]
                    begin_times.append(begin_time)
                    durations.append(duration)
                    texts.append(text)
                    meta_list.append(
                        dict(path=path, begin_time=begin_time, duration=duration, text=text, speaker="OLLEH_SPEAKER")
                    )
                paths.extend(
                    [
                        path,
                    ]
                    * len(segments)
                )

    meta_dict = dict(path=paths, begin_time=begin_times, duration=durations, text=texts)
    # store_panda(meta_dict)
    # import pdb; pdb.set_trace()
    # 0.02s/sentence
    with open("temp.json", "w") as f:
        for meta in meta_list:
            line = json.dumps(meta, ensure_ascii=False)
            f.write(line + "\n")
