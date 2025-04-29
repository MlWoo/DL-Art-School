import argparse
import functools
import glob
import logging
import os
from io import BytesIO
from multiprocessing import Pool

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Dataset, Features, Value, load_dataset
from tqdm import tqdm

features = Features(
    {
        "audio_id": Value("string"),
        "audio": Audio(),
        "duration": Value("float32"),
    }
)

ROUND_PRECISION = 5
GAP_THRESHOLD = 1


def merge_intervals(intervals, min_gap):
    if not intervals:
        return []
    merged = [intervals[0].copy()]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] - last[1] < min_gap:
            merged[-1][1] = current[1]
        else:
            merged.append(current.copy())
    return merged


def process_short_intervals(intervals):
    changed = True
    while changed:
        changed = False
        new_intervals = []
        i = 0
        while i < len(intervals):
            current = intervals[i]
            current_length = current[1] - current[0]
            if current_length < 2:
                # 获取前一个和后一个区间
                prev = new_intervals[-1] if new_intervals else None
                next_idx = i + 1
                next = intervals[next_idx] if next_idx < len(intervals) else None

                # 计算前后区间长度（若存在）
                prev_length = float("inf")
                if prev is not None:
                    prev_length = prev[1] - prev[0]
                next_length = float("inf")
                if next is not None:
                    next_length = next[1] - next[0]

                # 选择合并方向
                if prev_length <= next_length and prev is not None:
                    # 合并到前一个区间
                    new_intervals[-1][1] = current[1]
                    changed = True
                    i += 1  # 跳过当前区间
                elif next is not None:
                    # 合并到后一个区间，同时处理可能的连续合并
                    merged = [current[0], next[1]]
                    new_intervals.append(merged)
                    changed = True
                    i += 2  # 跳过当前和后一个
                else:
                    # 无法合并，保留当前区间
                    new_intervals.append(current)
                    i += 1
            else:
                new_intervals.append(current)
                i += 1
        intervals = new_intervals
    return intervals


def process_intervals(intervals):
    # Step 1: 合并间隔小于0.3的相邻区间
    merged_step1 = merge_intervals(intervals, 0.2)

    # Step 2: 合并相邻间隔小于等于0.8的区间，同时处理长度超过15的独立段
    processed = []
    for interval in merged_step1:
        if not processed:
            processed.append(interval.copy())
        else:
            last = processed[-1]
            last_length = last[1] - last[0]
            if last_length >= 14:
                processed.append(interval.copy())
            else:
                gap = interval[0] - last[1]
                if gap <= 0.5:
                    new_interval = [last[0], interval[1]]
                    new_length = new_interval[1] - new_interval[0]
                    if new_length > 14:
                        processed.append(interval.copy())
                    else:
                        processed[-1] = new_interval
                else:
                    processed.append(interval.copy())
    # Step 3: 处理长度低于2的区间
    # final_processed = process_short_intervals(processed)
    final_processed = processed
    return final_processed


def process_file(file, input_dir, output_dir):
    filename = file.split(".")[0]
    webdataset_path = os.path.join(input_dir, file)
    # if os.path.exists(f"{output_dir}/{filename}.parquet"):
    #     return
    try:
        dataset = load_dataset(
            "webdataset",
            data_files=webdataset_path,
            cache_dir="/home/wumenglin/data/hf/cache/librilight-webdataset",
            streaming=True,
        )
    except Exception as e:
        logging.error(f"Error loading dataset: {webdataset_path}")
        return

    dataset = dataset["train"]
    # duration_list = []
    # for i, data in tqdm(enumerate(dataset)):
    #     audio_info = data['flac']
    #     audio = audio_info['array']
    #     duration = audio.shape[0] / audio_info['sampling_rate']
    #     duration_list.append(duration)
    # total_duration = sum(duration_list)
    # import pdb; pdb.set_trace()
    # print(total_duration)
    # exit()

    audio_items = []
    slice_indices = []
    acc_duration = 0
    for data in tqdm(dataset):
        audio_id = data["__key__"]
        audio_info = data["flac"]
        json_info = data["json"]
        audio = audio_info["array"].astype(np.float32)
        sampling_rate = audio_info["sampling_rate"]
        voice_activity = json_info["voice_activity"]
        processed_intervals = process_intervals(voice_activity)
        for part, (part_start_duration, part_end_duration) in enumerate(processed_intervals):
            part_start_duration = round(part_start_duration, ROUND_PRECISION)
            part_end_duration = round(part_end_duration, ROUND_PRECISION)
            part_start = int(part_start_duration * sampling_rate)
            part_end = min(int(part_end_duration * sampling_rate), len(audio))
            part_end_duration = round(part_end / sampling_rate, ROUND_PRECISION)
            duration = round(part_end_duration - part_start_duration, ROUND_PRECISION)
            if abs(int(duration * sampling_rate) - len(audio[part_start:part_end])) > 160:
                logging.error(f"{webdataset_path} {part} {duration} {len(audio[part_start:part_end])}")
                break

            if abs(duration * sampling_rate - len(audio[part_start:part_end])) > 0:
                new_part_end = part_start + int(duration * sampling_rate)
                if part_end == new_part_end:
                    new_part_start = part_end - int(duration * sampling_rate)
                else:
                    new_part_start = part_start
                part_start = new_part_start
                part_end = new_part_end

            part_audio = audio[part_start:part_end]
            part_audio = np.ascontiguousarray(part_audio)
            memory_file = BytesIO()
            memory_file.name = f"{audio_id}_{part:03d}.flac"
            sf.write(memory_file, part_audio, sampling_rate)
            memory_file.seek(0)

            if int(duration * sampling_rate) != len(part_audio):
                logging.warning(
                    f"{webdataset_path} {part} {duration} {len(audio[part_start:part_end])} is not equal to {duration * sampling_rate}"
                )
            assert int(duration * sampling_rate) == len(part_audio)

            audio_items.append(
                {
                    "audio_id": audio_id,
                    "audio": {
                        "path": f"{audio_id}_{part:03d}.flac",
                        "bytes": memory_file.read(),
                    },
                    "duration": duration,
                }
            )
            acc_duration += duration
            if acc_duration > 28500:
                acc_duration = duration
                slice_indices.append(len(audio_items) - 1)

    # batch_size = 800
    # iter_params = []
    # sum_index = (len(audio_items) - 1) // batch_size + 1
    # for index in range(0, sum_index):
    #     iter_params.append(
    #         (
    #             audio_items[index * batch_size : (index + 1) * batch_size],
    #             os.path.join(output_dir, f"{filename}-{index:05d}-of-{sum_index:05d}.parquet"),
    #         )
    #     )
    slice_indices.append(len(audio_items))
    for i in range(len(slice_indices) - 1):
        params = audio_items[slice_indices[i] : slice_indices[i + 1]]
        if len(params) == 0:
            continue
        new_dataset = Dataset.from_list(params, features)
        table = new_dataset.with_format("arrow")[:]
        pq.write_table(table, f"{output_dir}/{filename}-{i:05d}-of-{len(slice_indices)-1:05d}.parquet")


def save_corrupted_file(file, input_dir, output_dir):
    filename = file.split(".")[0]
    parquet_path = f"{input_dir}/{filename}.parquet"
    assert os.path.exists(parquet_path)
    df = pl.read_parquet(parquet_path)
    audio_items = []
    for i, row in tqdm(enumerate(df.iter_rows(named=True))):
        audio_id = row["audio_id"]
        duration = row["duration"]
        audio_path = row["audio"]["path"]
        audio_bytes = row["audio"]["bytes"]
        audio_bytes = BytesIO(audio_bytes)
        audio, sampling_rate = sf.read(audio_bytes)
        audio = audio.astype(np.float32)
        audio = audio.copy()
        memory_file = BytesIO()
        memory_file.name = f"{audio_id}_{i:03d}.flac"
        sf.write(memory_file, audio, sampling_rate)
        memory_file.seek(0)

        audio_items.append(
            {
                "audio_id": audio_id,
                "audio": {
                    "path": f"{audio_id}_{i:03d}.flac",
                    "bytes": memory_file.read(),
                },
                "duration": duration,
            }
        )

    new_dataset = Dataset.from_list(audio_items, features)
    table = new_dataset.with_format("arrow")[:]
    pq.write_table(table, f"{output_dir}/{filename}.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="/data0/.cache/huggingface/hub/datasets--collabora--librilight-webdataset/snapshots/d384c1aebb63f1be4100dd844147e816213d1440",
        help="Path to the data file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="/home/wumenglin/data/hf/librilight-webdataset",
        help="Path to the output file.",
    )
    parser.add_argument("-c", "--save-dir", type=str, default=None, help="Path to the corrupted file.")
    parser.add_argument("-p", "--num-threads", type=int, default=48, help="Number of threads to use.")
    args = parser.parse_args()
    func = process_file if args.save_dir is None else save_corrupted_file

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    input_dir = args.input_dir
    file_list = []
    cnt = 0

    # for file in os.listdir(args.output_dir):
    #     cnt += 1
    #     if file.endswith(".parquet"):
    #         file_list.append(file)
    #         # import pdb; pdb.set_trace()
    #         # break
    #         # if os.path.exists(f"{args.output_dir}/{file.split('.')[0]}.parquet"):
    #         #     break
    #         # file_list.append(file)
    # print(file_list[0])
    # func(file_list[0], args.output_dir, args.save_dir)
    # exit()

    for file in os.listdir(input_dir):
        cnt += 1
        if file.endswith(".tar"):
            if len(glob.glob(f"{args.output_dir}/{file.split('.')[0]}*.parquet")) > 0:
                continue
            file_list.append(file)
    # print(file_list[0])
    # func(file_list[0], input_dir, args.output_dir)
    # exit()

    # print(file_list, cnt, len(file_list))

    with Pool(args.num_threads) as pool:
        list(
            tqdm(
                pool.imap(
                    functools.partial(
                        func,
                        input_dir=input_dir,
                        output_dir=args.output_dir,
                    ),
                    file_list,
                ),
                total=len(file_list),
            )
        )
