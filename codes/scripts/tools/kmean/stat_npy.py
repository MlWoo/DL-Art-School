import argparse
import json
import os
import os.path as osp
from multiprocessing.pool import ThreadPool

import numpy as np
from tqdm import tqdm

from babble.utils import list_dir


def process_data(in_params):
    out_json, sub_abs_dir = in_params
    files = list_dir(sub_abs_dir, extension=".npy")
    lines = []
    for fn in tqdm(files):
        fp = osp.join(sub_abs_dir, fn)
        try:
            data = np.load(fp)
            frames = data.shape[1]
            info = dict(
                code_path=fp,
                frames=frames,
            )
            line = json.dumps(info)
            lines.append(line)
        except:
            print("accident error.")
    with open(out_json, "w") as f:
        for line in lines:
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i",
        "--in_dir",
        type=str,
        default="/mnt/bn/wml-lq-8t/repo/DL-Art-School/temp/dvae-2048-64hz/",
        help="Output path.",
    )
    parser.add_argument(
        "-o", "--out_dir", type=str, required=True, help="Number of concurrent workers processing files."
    )
    parser.add_argument(
        "-p", "--num_threads", type=int, help="Number of concurrent workers processing files.", default=1
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    in_dir = list_dir(args.in_dir)
    data_info = []
    for i, sub_dir in enumerate(in_dir):
        sub_abs_dir = osp.join(args.in_dir, sub_dir)
        out_json = osp.join(args.out_dir, f"part_{i}")
        data_info.append((out_json, sub_abs_dir))

    if args.num_threads > 1:
        with ThreadPool(args.num_threads) as pool:
            tqdm(list(pool.imap(process_data, data_info, chunksize=1)), total=len(data_info))
    else:
        for subdir in tqdm(data_info):
            process_data(subdir)
