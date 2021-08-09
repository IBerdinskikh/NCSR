import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils.data_augmentation import *
import pyvips
import random
import cv2

# from .data_rgb import split_image_into_overlapping_patches


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".PNG"])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, "groundtruth")))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, "input")))

        self.clean_filenames = [
            os.path.join(rgb_dir, "groundtruth", x)
            for x in clean_files
            if is_png_file(x)
        ]
        self.noisy_filenames = [
            os.path.join(rgb_dir, "input", x) for x in noisy_files if is_png_file(x)
        ]

        self.img_options = img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        noisy = pyvips.Image.new_from_file(
            self.noisy_filenames[tar_index], access="sequential"
        )

        clean = pyvips.Image.new_from_file(
            self.clean_filenames[tar_index], access="sequential"
        )

        # Crop Input and Target
        ps = self.img_options["patch_size"]
        H = noisy.height
        W = noisy.width
        B = noisy.bands
        r = random.randint(0, H - ps)
        c = random.randint(0, W - ps)

        noisy = noisy.crop(c, r, ps, ps).write_to_memory()
        clean = clean.crop(c, r, ps, ps).write_to_memory()

        noisy = np.ndarray(buffer=noisy, dtype=np.uint8, shape=[ps, ps, B])
        clean = np.ndarray(buffer=clean, dtype=np.uint8, shape=[ps, ps, B])

        noisy = torch.from_numpy(np.float32(noisy / 255.0))
        clean = torch.from_numpy(np.float32(clean / 255.0))

        noisy = noisy.permute(2, 0, 1)
        clean = clean.permute(2, 0, 1)

        return clean, noisy  # , clean_filename, noisy_filename


# ##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, "groundtruth")))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, "input")))

        self.clean_filenames = [
            os.path.join(rgb_dir, "groundtruth", x)
            for x in clean_files
            if is_png_file(x)
        ]
        self.noisy_filenames = [
            os.path.join(rgb_dir, "input", x) for x in noisy_files if is_png_file(x)
        ]

        self.img_options = img_options

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = pyvips.Image.new_from_file(
            self.clean_filenames[tar_index], access="sequential"
        )
        noisy = pyvips.Image.new_from_file(
            self.noisy_filenames[tar_index], access="sequential"
        )

        # Crop Input and Target
        ps = self.img_options["patch_size"]
        H = clean.height
        W = clean.width
        B = clean.bands
        r = random.randint(0, H - ps)
        c = random.randint(0, W - ps)
        clean = clean.crop(c, r, ps, ps).write_to_memory()
        noisy = noisy.crop(c, r, ps, ps).write_to_memory()

        clean = np.ndarray(buffer=clean, dtype=np.uint8, shape=[ps, ps, B])
        noisy = np.ndarray(buffer=noisy, dtype=np.uint8, shape=[ps, ps, B])

        clean = torch.from_numpy(np.float32(clean / 255.0))
        noisy = torch.from_numpy(np.float32(noisy / 255.0))

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################


class DataLoaderTrainSR(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrainSR, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, "groundtruth")))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, "input")))

        self.clean_filenames = [
            os.path.join(rgb_dir, "groundtruth", x)
            for x in clean_files
            if is_png_file(x)
        ]
        self.noisy_filenames = [
            os.path.join(rgb_dir, "input", x) for x in noisy_files if is_png_file(x)
        ]

        self.img_options = img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        noisy = pyvips.Image.new_from_file(
            self.noisy_filenames[tar_index], access="sequential"
        )

        clean = pyvips.Image.new_from_file(
            self.clean_filenames[tar_index], access="sequential"
        )

        # Crop Input and Target
        scale = 2
        ps = self.img_options["patch_size"]
        lq_ps = self.img_options["patch_size"] // scale
        H = noisy.height
        W = noisy.width
        B = noisy.bands
        r = random.randint(0, H - lq_ps)
        c = random.randint(0, W - lq_ps)

        noisy = noisy.crop(c, r, lq_ps, lq_ps).write_to_memory()
        clean = clean.crop(c * scale, r * scale, ps, ps).write_to_memory()

        noisy = np.ndarray(buffer=noisy, dtype=np.uint8, shape=[lq_ps, lq_ps, B])
        clean = np.ndarray(buffer=clean, dtype=np.uint8, shape=[ps, ps, B])

        noisy = torch.from_numpy(np.float32(noisy / 255.0))
        clean = torch.from_numpy(np.float32(clean / 255.0))

        noisy = noisy.permute(2, 0, 1)
        clean = clean.permute(2, 0, 1)

        # if random.random() < 0.5:
        # noisy, clean = adjust_brightness(noisy, clean)
        # noisy, clean = adjust_contrast(noisy, clean)
        # noisy, clean = adjust_saturation(noisy, clean)

        noisy, clean = random_horizontally_flip(noisy, clean)

        noisy = torch.clamp(noisy, 0, 1)
        clean = torch.clamp(clean, 0, 1)

        return clean, noisy


##################################################################################################


class DataLoaderValSR(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderValSR, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, "groundtruth")))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, "input")))

        self.clean_filenames = [
            os.path.join(rgb_dir, "groundtruth", x)
            for x in clean_files
            if is_png_file(x)
        ]
        self.noisy_filenames = [
            os.path.join(rgb_dir, "input", x) for x in noisy_files if is_png_file(x)
        ]

        self.img_options = img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        noisy = pyvips.Image.new_from_file(
            self.noisy_filenames[tar_index], access="sequential"
        )

        clean = pyvips.Image.new_from_file(
            self.clean_filenames[tar_index], access="sequential"
        )

        # Crop Input and Target
        scale = 2
        ps = self.img_options["patch_size"]
        lq_ps = self.img_options["patch_size"] // scale
        H = noisy.height
        W = noisy.width
        B = noisy.bands
        r = random.randint(0, H - lq_ps)
        c = random.randint(0, W - lq_ps)

        noisy = noisy.crop(c, r, lq_ps, lq_ps).write_to_memory()
        clean = clean.crop(c * scale, r * scale, ps, ps).write_to_memory()

        noisy = np.ndarray(buffer=noisy, dtype=np.uint8, shape=[lq_ps, lq_ps, B])
        clean = np.ndarray(buffer=clean, dtype=np.uint8, shape=[ps, ps, B])

        noisy = torch.from_numpy(np.float32(noisy / 255.0))
        clean = torch.from_numpy(np.float32(clean / 255.0))

        noisy = noisy.permute(2, 0, 1)
        clean = clean.permute(2, 0, 1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################


class DataLoaderPredictSR(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderPredictSR, self).__init__()

        self.target_transform = target_transform

        noisy_files = sorted(os.listdir(rgb_dir))
        self.noisy_filenames = [
            os.path.join(rgb_dir, x) for x in noisy_files if is_png_file(x)
        ]

        self.img_options = img_options

        self.tar_size = len(self.noisy_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        noisy = cv2.cvtColor(
            cv2.imread(self.noisy_filenames[tar_index]), cv2.COLOR_BGR2RGB
        )

        noisy = np.float32(noisy / 255.0)

        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        return noisy, noisy_filename
