import os
from torch.utils.data.sampler import Sampler
import math
import torch
import threading
from torch.utils.data import DataLoader
import queue as Queue
import numpy as np


from .dataset_rgb import (
    DataLoaderTrain,
    DataLoaderTrainSR,
    DataLoaderVal,
    DataLoaderValSR,
    DataLoaderPredictSR,
)


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)


def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options, None)


def get_training_data_SR(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrainSR(rgb_dir, img_options, None)


def get_validation_data_SR(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderValSR(rgb_dir, img_options, None)


def get_predict_data_SR(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderPredictSR(rgb_dir, None, None)


class DistSampler(Sampler):
    """Distributed sampler that loads data from a subset of the dataset.
    Actually just generate idxs.
    Why enlarge? We only shuffle the dataloader before each epoch.
        Enlarging dataset can save the shuffling time.
    Support we have im00, im01, im02. We set ratio=3 and we have 2 workers.
        Enlarged ds: im00 01 02 00 01 02 00 01 02
        Worker 0: im00, im02, im01, im00, im02
        Worker 1: im01, im00, im02, im01, (im00)
    Enlargement is compatible with augmentation.
        Each sampling is different due to the random augmentation.
    Modified from torch.utils.data.distributed.DistributedSampler.
    Args:
        dataset size.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio.
    """

    def __init__(self, ds_size, num_replicas=None, rank=None, ratio=1):
        self.ds_size = ds_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # enlarged by ratio, and then divided by num_replicas
        self.num_samples = math.ceil(ds_size * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        """
        For distributed training, shuffle the subset of each dataloader.
        For single-gpu training, no shuffling.
        """
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)  # shuffle based on self.epoch
        idxs = torch.randperm(self.total_size, generator=g).tolist()
        idxs = [idx % self.ds_size for idx in idxs]
        idxs = idxs[self.rank : self.total_size : self.num_replicas]
        return iter(idxs)

    def __len__(self):
        return self.num_samples  # for one rank


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.

    Ref:
    https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """Prefetch version of dataloader.

    Ref:
    https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    TODO:
    Need to test on single gpu and ddp (multi-gpu). There is a known issue in
    ddp.

    Args:
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.
    """

    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher:
    """CPU prefetcher."""

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher:
    """CUDA prefetcher.

    Ref:
    https://github.com/NVIDIA/apex/issues/304#

    It may consums more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, device):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(self.device)
        self.preload()

    def preload(self):
        try:
            self.batch = {}
            for i, b in enumerate(next(self.loader)):
                self.batch[i] = b
        except StopIteration:
            self.batch = None
            return None

        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()


def split_image_into_overlapping_patches(image_array, patch_size):
    step = patch_size

    if image_array.ndim == 2:
        h, w = image_array.shape
    elif image_array.ndim == 3:
        h, w, c = image_array.shape
    else:
        raise ValueError(f"Image ndim should be 2 or 3, but got {image_array.ndim}")

    # [  0 256 464]
    h_space = np.arange(0, h - patch_size + 1, step)
    h_space = np.append(h_space, h - patch_size)

    w_space = np.arange(0, w - patch_size + 1, step)
    w_space = np.append(w_space, w - patch_size)

    index = 0
    patches = []
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = image_array[x : x + patch_size, y : y + patch_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            patches.append(cropped_img)

    return np.array(patches), image_array.shape


def stich_together(patches, padded_image_shape, scale=2):
    h, w, c = padded_image_shape
    h = h * scale
    w = w * scale

    patch_size = patches.shape[1]
    step = patch_size
    complete_image = np.zeros((h, w, 3))

    # [  0 256 464]
    h_space = np.arange(0, h - patch_size + 1, step)
    h_space = np.append(h_space, h - patch_size)

    w_space = np.arange(0, w - patch_size + 1, step)
    w_space = np.append(w_space, w - patch_size)

    index = 0
    for x in h_space:
        for y in w_space:
            complete_image[x : x + patch_size, y : y + patch_size, ...] = patches[index]
            index += 1

    return complete_image