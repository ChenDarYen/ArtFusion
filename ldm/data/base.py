import bisect
import random

import numpy as np
import albumentations
from PIL import Image
import torch
from torch.utils.data import Dataset

from main import instantiate_from_config


class ConcatDataset(Dataset):
    def __init__(self, data_cfgs):
        self.datasets = torch.utils.data.ConcatDataset([instantiate_from_config(cfg) for cfg in data_cfgs])

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        return self.datasets.__getitem__(item)


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=384, crop_size=256, random_augment=False, smallest_max_size=True, labels=None):
        self.size = size
        self.crop_size = crop_size or size
        self.random_augment = random_augment

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size) if smallest_max_size else \
                albumentations.Resize(height=self.size, width=self.size)
            self.cropper = \
                albumentations.RandomCrop(height=self.crop_size, width=self.crop_size) if random_augment else \
                    albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
            transforms = [self.rescaler, self.cropper]
            if random_augment:
                self.flipper = albumentations.HorizontalFlip(p=0.5)
                transforms.append(self.flipper)

            self.preprocessor = albumentations.Compose(transforms=transforms)
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
