import os
import queue
import random
import time
from functools import partial
from multiprocessing.pool import ThreadPool
from threading import Thread
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.distributed import get_dist_info
from utils.seed import set_random_seed

from data.sampler import LengthChunkSampler

# Object used by _background_consumer to signal the source is exhausted
# to the main thread.
_sentinel = object()


class BackgroundConsumer(Thread):
    def __init__(self, queue, source, max_len, cuda_device, post_func):
        Thread.__init__(self)

        self._queue = queue
        self._source = source
        self._max_len = max_len
        self.count = 0
        self.cuda_device = cuda_device
        self.post_func = post_func

    def run(self):
        # set_device to avoid creation of GPU0 context when using pin_memory
        if self.cuda_device is not None:
            torch.cuda.set_device(self.cuda_device)
        try:
            for item in self._source:
                if self.post_func is not None:
                    if self.cuda_device is None:
                        new_item = item
                        item = self.post_func(new_item, force_inter=True)
                    else:
                        new_item = dict()
                        for k, v in item.items():
                            if isinstance(v, torch.Tensor):
                                new_item[k] = v.to(self.cuda_device)
                            else:
                                new_item[k] = v
                        new_item = self.post_func(new_item, force_inter=True)
                        new_item_x = dict()
                        for k, v in new_item.items():
                            v = v.cpu()
                            new_item_x[k] = v
                        item = new_item_x

                self._queue.put(item)

                # Stop if we reached the maximum length
                self.count += 1
                if self._max_len is not None and self.count >= self._max_len:
                    break

            # Signal the consumer we are done.
            self._queue.put(_sentinel)
        except Exception as e:
            self._queue.put(e)


class BufferedIteratorDataloader(object):
    def __init__(self, size, iterable, collate_fn=None, process_device=-1, predefine_len=False):
        self._queue = queue.Queue(size)
        self._iterable = iterable
        self._consumer = None

        self.start_time = time.time()
        self.warning_time = None

        if collate_fn is not None and hasattr(collate_fn, "_postprocess_tensor_"):
            self.post_func = collate_fn._postprocess_tensor_
        else:
            self.post_func = None
        self.process_device = process_device
        if predefine_len:
            self.total = len(iterable)
        else:
            self.total = None
        self.predefine_len = predefine_len

    def _create_consumer(self):
        self._consumer = BackgroundConsumer(
            self._queue,
            self._iterable,
            self.total if self.predefine_len else None,
            torch.cuda.current_device() if torch.cuda.is_available() and self.process_device >= 0 else None,
            self.post_func,
        )
        self._consumer.daemon = True
        self._consumer.start()

    def __iter__(self):
        return self

    def __len__(self):
        return self.total if self.predefine_len else len(self._iterable)

    def set_epoch(self, epoch):
        del self._consumer
        self._consumer = None
        self._iterable.set_epoch(epoch)

    def take(self, n):
        self.total = min(self.total, n)
        # Propagate this change to the underlying iterator
        if hasattr(self._iterable, "take"):
            self._iterable.take(n)
        return self

    def __next__(self):
        # Create consumer if not created yet
        if self._consumer is None:
            self._create_consumer()

        # Notify the user if there is a data loading bottleneck
        if self._queue.qsize() < min(2, max(1, self._queue.maxsize // 2)):
            if time.time() - self.start_time > 5 * 60:
                if self.warning_time is None or time.time() - self.warning_time > 15 * 60:
                    print(
                        "Data loading buffer is empty or nearly empty. This may "
                        "indicate a data loading bottleneck, and increasing the "
                        "number of workers (--num-workers) may help."
                    )
                    self.warning_time = time.time()

        # Get next example
        item = self._queue.get(True)
        if isinstance(item, Exception):
            raise item
        if item is _sentinel:
            raise StopIteration()
        return item

    @property
    def dataset(self):
        return self._iterable.dataset

    @property
    def sampler(self):
        return self._iterable.sampler

    @property
    def collate_fn(self):
        return self._iterable.collate_fn


def file_worker_init_fn(worker_id, num_workers, rank, seed, cpu_affinity=False):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    print(f"file worker init ====== worker_id: {worker_id} num_workers: {num_workers}, rank: {rank} ======")
    # worker_seed = num_workers * rank + worker_id + seed
    # np.random.seed(worker_seed)
    # random.seed(worker_seed)
    set_random_seed(seed)
    if False and cpu_affinity:
        cpu = num_workers * rank + worker_id
        os.sched_setaffinity(0, [cpu])


def pool_worker_init_fn(worker_id, num_workers, rank, seed, cpu_affinity=False):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker_info = torch.utils.data.get_worker_info()
    worker_dataset = worker_info.dataset
    if worker_dataset.pool_threads > 0:
        thread_pool = ThreadPool(worker_dataset.pool_threads)
        pool_name = "pool"
        setattr(worker_dataset, pool_name, thread_pool)


class GeneralDataloader(DataLoader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.outer_iter: int = 0
        self.inner_iter: int = 0
        self.total_iter: int = 0

    def __inner_iter__(self):
        self.inner_iter = 0
        # StopIteration will be raised if ends is reached
        batches = next(self.outer_iterator)
        if isinstance(batches, Iterable) and not isinstance(batches, dict):
            batch_iterator = iter(batches)
        else:
            batch_iterator = iter(
                [
                    batches,
                ]
            )
        self.outer_iter + 1
        return batch_iterator

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __len__(self):
        return self.sampler.num_batch

    def __iter__(self):
        self.outer_iterator: Iterable = super().__iter__()
        self.inner_iterator: Iterable = self.__inner_iter__()
        return self

    def __next__(self):
        while True:
            try:
                data = next(self.inner_iterator)
                self.inner_iter += 1
                self.total_iter += 1
                return data
            except StopIteration:
                # StopIteration will be raised if the ends outer iterator is reached,
                # and the whole dataloader will be over.
                self.inner_iterator = self.__inner_iter__()

    @property
    def progress(self):
        return self.total_iter


class SubDataloaders(object):
    def __init__(self, dataloader: GeneralDataloader, num_subdataloader: int = 1, debug: bool = False):
        """
        Args:
        dataloader (GeneralDataloader): A PyTorch genertal dataset.
        num_subdataloader (int): Number of sub dataloaders
        Returns:
        DataLoader (Iterable(GeneralDataloader)): PyTorch dataloaders.
        """
        self.dataloader = dataloader
        self.num_subdataloader = num_subdataloader
        self.sampler_iterable = None
        self.inner_loader = None
        self.debug = debug

    @property
    def num_batch(self):
        return self.sampler.num_batch

    @property
    def dataset(self):
        return self.dataloader.dataset

    @property
    def sampler(self):
        return self.dataloader.sampler

    @property
    def collate_fn(self):
        return self.dataloader.collate_fn

    @property
    def iter_num(self):
        if self.sampler.num_batch % self.num_subdataloader == 0:
            self.iter_num = self.sampler.num_batch // self.num_subdataloader
        else:
            self.iter_num = self.sampler.num_batch // self.num_subdataloader + 1

    def __len__(self):
        return self.num_batch

    def __inner_iter__(self, reset=False):
        if self.debug:
            if reset:
                i_r = list(self.sampler_iterable)
                remainder_len = len(i_r)
                self.sampler_iterable = iter(i_r)
            else:
                remainder_len = -1
            print(f"[Debug] ---> reset: {reset}, remainder len {remainder_len}")
        if reset:
            self.sampler.iter_is_null = True
        else:
            self.sampler.load_iter(self.sampler_iterable)

        self.sampler_iterable = self.sampler.truncate_iter(self.iter_num)
        self.inner_loader = iter(self.dataloader)

    def __iter__(self):
        self.sampler_iterable = self.sampler.iterable
        self.__inner_iter__(reset=self.sampler.iter_is_null)
        return self

    def __next__(self):
        while True:
            try:
                data = next(self.inner_loader)
                return data
            except StopIteration:
                self.__inner_iter__(reset=False)


class GeneralSeqDataloader:
    def __init__(
        self,
        dataset,
        collate_fn,
        workers_per_gpu,
        pin_memory=False,
        seed=0,
        multiprocessing_context="forkserver",
        persistent_workers=True,
        buffer_background=False,
        buffer_background_size=4,
        process_background=False,
        process_device=-1,
        debug=False,
        sampler_opt=None,
        **kwargs,
    ):
        """Build PyTorch DataLoader.
        In distributed training, each GPU/process has a dataloader.
        In non-distributed training, there is only one dataloader for all GPUs.
        Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training sample on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        buffer_batch_group (int): KVDataset buffer size to increase loading efficiency
        phases (list[str]): KVDataset will be splited into different parts to be used in respected phases.
        split_ratios (list[float]): KVDataset will be splited into different parts according to these ratios
        last_samples (str): How to handle with the tailfing samples in the last batch.
        num_range (tuple(int)): Limited the number of per class in this dataloader. The dataset of one class will be
            ignored if the number is lower than left boundary. The redundant samples will be discarded if the number
            exceeds the right boundary.
        num_classes_limit (int): The upper limit of classes.
        shuffle (bool): Whether to shuffle the data at every epoch. Default: True.
        seed (int): The random seed of dataloaders in  distributed training.
        copies(int): How many epoches to release dataloader resources. The option will increase dataloader efficiency.
        multiprocessing_context (context or str): The mothod to spawn the multiprocessing, `spawn` is forbidden in
            KVDataset when connected HDFS storage.
        persistent_workers (bool): whether the workers are realeased or not when reloading the dataloaders,
            Default: True.
        kwargs: any keyword argument to be used to initialize samplers
        Returns:
        DataLoader: PyTorch dataloaders.
        """
        self.phase_named_dataloaders = dict()
        if kwargs.get("world_size", None) is None:
            rank, world_size = get_dist_info()
        else:
            world_size = kwargs.get("world_size")
            rank = kwargs.get("rank")

        samples_per_gpu = kwargs.get("batch_size", 2) if sampler_opt is None else sampler_opt.get("batch_size", 2)
        buffer_batch_group = (
            kwargs.get("buffer_batch_group", 1) if sampler_opt is None else sampler_opt.get("buffer_batch_group", 1)
        )
        bucket_batch_volume = (
            kwargs.get("bucket_batch_volume", 32) if sampler_opt is None else sampler_opt.get("bucket_batch_volume", 32)
        )
        length_range = (
            kwargs.get("length_range", (0, -1)) if sampler_opt is None else sampler_opt.get("length_range", (0, -1))
        )
        limited_type = (
            kwargs.get("limited_type", None) if sampler_opt is None else sampler_opt.get("limited_type", None)
        )
        similar_type = (
            kwargs.get("similar_type", None) if sampler_opt is None else sampler_opt.get("similar_type", None)
        )
        bucket_boundaries = (
            kwargs.get("bucket_boundaries", None) if sampler_opt is None else sampler_opt.get("bucket_boundaries", None)
        )
        num_buckets = kwargs.get("num_buckets", None) if sampler_opt is None else sampler_opt.get("num_buckets", None)
        bucket_padding_noise = (
            kwargs.get("bucket_padding_noise", 0.0)
            if sampler_opt is None
            else sampler_opt.get("bucket_padding_noise", 0.0)
        )
        batch_mode = (
            kwargs.get("batch_mode", "dynamical") if sampler_opt is None else sampler_opt.get("batch_mode", "dynamical")
        )
        acc_coeffs = (
            kwargs.get("acc_coeffs", ((1.0, 1.0)))
            if sampler_opt is None
            else sampler_opt.get("acc_coeffs", ((1.0, 1.0)))
        )
        max_tokens = kwargs.get("max_tokens", 1024) if sampler_opt is None else sampler_opt.get("max_tokens", 1024)
        max_samples = kwargs.get("max_samples", 16) if sampler_opt is None else sampler_opt.get("max_samples", 16)
        copies = kwargs.get("copies", 1) if sampler_opt is None else sampler_opt.get("copies", 1)
        last_samples = (
            kwargs.get("last_samples", "pad") if sampler_opt is None else sampler_opt.get("last_samples", "pad")
        )
        shuffle = kwargs.get("shuffle", True) if sampler_opt is None else sampler_opt.get("shuffle", True)
        partition_record_path = (
            kwargs.get("partition_record_path", None)
            if sampler_opt is None
            else sampler_opt.get("partition_record_path", None)
        )
        bucket_max_batch_tokens = (
            kwargs.get("bucket_max_batch_tokens", 16384)
            if sampler_opt is None
            else sampler_opt.get("bucket_max_batch_tokens", 16384)
        )
        bucket_min_samples = (
            kwargs.get("bucket_min_samples", 1) if sampler_opt is None else sampler_opt.get("bucket_min_samples", 1)
        )
        bucket_max_samples = (
            kwargs.get("bucket_max_samples", 64) if sampler_opt is None else sampler_opt.get("bucket_max_samples", 64)
        )

        for phase, map_dataset_indice in dataset.phases_indice_dict.items():
            if phase == "train":
                if copies > 1:
                    _copies = round(copies)
                    persistent_workers = True
                elif copies == 1:
                    _copies = 1
                else:
                    persistent_workers = False
                    _copies = 1
                i_workers_per_gpu = workers_per_gpu
            else:
                _copies = 1
                last_samples = "keep"
                shuffle = False
                i_workers_per_gpu = 1

            sampler = LengthChunkSampler(
                batch_size=samples_per_gpu,
                buffer_batch_group=buffer_batch_group,
                shuffle=shuffle,
                last_samples=last_samples,
                copies=_copies,
                seed=seed,
                num_replicas=world_size,
                rank=rank,
                limited_type=limited_type,
                similar_type=similar_type,
                length_range=length_range,
                phase=phase,
                bucket_batch_volume=bucket_batch_volume,
                bucket_boundaries=bucket_boundaries,
                num_buckets=num_buckets,
                bucket_padding_noise=bucket_padding_noise,
                batch_mode=batch_mode,
                acc_coeffs=acc_coeffs,
                max_tokens=max_tokens,
                max_samples=max_samples,
                partition_record_path=partition_record_path,
                bucket_max_batch_tokens=bucket_max_batch_tokens,
                bucket_min_samples=bucket_min_samples,
                bucket_max_samples=bucket_max_samples,
            )
            sampler.finalize_dataset(dataset=dataset, indices=map_dataset_indice, verbose=True)
            if sampler.batch_mode == "bucketed":
                collate_fn.set_bucketed_batch_size(
                    key=sampler.similar_type, bucket_boundaries_batch_size_map=sampler.bucket_boundaries_batch_size_map
                )

            if hasattr(sampler, "seed"):
                seed = sampler.seed
            else:
                seed = 0
            if hasattr(dataset, "pool"):
                worker_init_fn = pool_worker_init_fn
            else:
                worker_init_fn = file_worker_init_fn
            init_fn = partial(worker_init_fn, num_workers=i_workers_per_gpu, rank=rank, seed=seed, cpu_affinity=True)

            if i_workers_per_gpu == 0:
                multiprocessing_context = None
                persistent_workers = False

            if buffer_background and collate_fn is not None and process_background:
                collate_fn.inter_post = False

            _dataloader = GeneralDataloader(
                dataset,
                batch_size=None,
                num_workers=i_workers_per_gpu,
                sampler=sampler,
                pin_memory=pin_memory,
                drop_last=False,
                collate_fn=collate_fn,
                worker_init_fn=init_fn,
                multiprocessing_context=multiprocessing_context,
                persistent_workers=persistent_workers,
            )
            if phase == "train" and copies < 1:
                fraction = round(1.0 / copies)
                dataloader = SubDataloaders(dataloader=_dataloader, num_subdataloader=fraction, debug=debug)
            else:
                dataloader = _dataloader

            self.phase_named_dataloaders[phase] = (
                BufferedIteratorDataloader(
                    buffer_background_size,
                    dataloader,
                    collate_fn,
                    process_device,
                    predefine_len=batch_mode == "fixed",
                )
                if buffer_background
                else dataloader
            )

        self._dataset = dataset

    def dataloader(self, name="default"):
        if len(self.phase_named_dataloaders) == 1 and (name == "default" or name is None):
            default = list(self.phase_named_dataloaders.keys())[0]
            return self.phase_named_dataloaders[default]
        else:
            return self.phase_named_dataloaders[name]

    @property
    def named_dataloaders(self):
        return self.phase_named_dataloaders

    @property
    def dataset(self):
        return self._dataset
