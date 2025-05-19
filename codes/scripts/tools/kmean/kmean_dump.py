import random
import numpy as np
import torch


def kmean_cluster(samples, means):
    dists = torch.cdist(samples, means)
    indices = dists.argmin(dim=1).cpu().numpy()
    return indices.tolist()


def dump_label(samples, mean):
    dims = samples[0].shape[-1]
    x_lens = [x.shape[1] for x in samples]
    total_len = sum(x_lens)
    x_sel = torch.FloatTensor(1, total_len, dims)
    start_len = 0
    for sample in samples:
        sample_len = sample.shape[1]
        end_len = start_len + sample_len
        x_sel[:, start_len:end_len] = sample
        start_len = end_len
    dense_x = x_sel.cuda().squeeze(0)
    mean = mean.cuda()
    indices = kmean_cluster(dense_x, mean)
    indices_list = []
    start_len = 0
    for x_len in x_lens:
        end_len = start_len + end_len
        indices_list.append(indices[start_len:end_len])
    return indices_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("km_path")
    args = parser.parse_args()
    samples = []
    for i in range(4):
        length = random.randint(4, 10)
        samples.append(torch.randn(1, length, 512))
    if args.km_path is None:
        mean = torch.randn(8192, 512)
    else:
        km_np = np.load(args.km_path)
        mean = torch.from_numpy(km_np)
    indices_list = dump_label(samples, mean)
    print(indices_list)
