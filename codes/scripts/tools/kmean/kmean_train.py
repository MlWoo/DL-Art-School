import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{cur_path}/")
import argparse
from kmeans_monster import KMeanReservoir
from utils import options as option

from data import create_dataloader, create_dataset


def get_data(data, key="audio"):
    return data[key]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-opt_data", type=str, help="Path to option YAML file.", default="../conf/repre_learner7/gen_feat.yml"
    )
    parser.add_argument(
        "-km_path",
        type=str,
        help="Path to option YAML file.",
    )
    parser.add_argument(
        "-out_dir",
        type=str,
        help="Path to option YAML file.",
    )

    args = parser.parse_args()
    opt_data = option.parse(args.opt_data, is_train=False)
    dt_params = opt_data["datasets"]["train"]

    ds, collate_fn = create_dataset(dt_params, return_collate=True)

    max_iter_epoch = 20
    kmean = KMeanReservoir(
        n_clusters=1024,
        max_iter_epoch=200000,
        estimation_reservoir_size=8192 * 16 * max_iter_epoch,
        estimation_iters=50,
        minibatch=1024 * 16,
        tol=0.001,
        verbose=2,
        mode="euclidean",
        init_method="kmeans++",
        km_path=args.km_path,
    )
    os.makedirs(args.out_dir, exist_ok=True)

    kmean.fit_dataset_kmean(ds, km_dir=args.out_dir, data_func=get_data, device=0, save_interval=50)
