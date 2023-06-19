import os
import random
import warnings

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from ldm.data.base import ImagePaths

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def test_images(root, images):
    passed_images = list()
    for fname in tqdm(images):
        with warnings.catch_warnings(record=True) as caught_warnings:
            image = np.array(Image.open(os.path.join(root, fname)))
            if len(caught_warnings) > 0:
                continue
        if image.ndim == 3 and image.shape[-1] not in [1, 3, 4]:
            continue
        passed_images.append(fname)

    return passed_images


def get_all_images(root, split_ratio=0.):
    train_file = os.path.join(root, 'train.npy')
    val_file = os.path.join(root, 'val.npy')

    if not os.path.isfile(train_file) or not os.path.isfile(val_file):
        images = list()
        for category in [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]:
            category_dir = os.path.join(root, category)
            images.extend([os.path.join(category, fname) for fname in os.listdir(category_dir)
                           if os.path.splitext(fname)[1].lower() in ['.jpg', '.jpeg', '.png']])
        # passed_images = test_images(root, images)
        passed_images = images

        random.shuffle(passed_images)
        num_train_images = int(len(passed_images) * (1 - split_ratio))
        images_train = np.array(passed_images[:num_train_images])
        images_val = np.array(passed_images[num_train_images:])

        np.save(train_file, images_train)
        np.save(val_file, images_val)

    else:
        images_train = np.load(train_file)
        images_val = np.load(val_file)

    return {'train': images_train, 'val': images_val}


class Base(Dataset):
    def __init__(self, content_drop=0.5, style_drop=0.1, *args, **kwargs):
        super().__init__()
        self.content_drop = content_drop
        self.style_drop = style_drop
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        r = random.random()
        if r < self.style_drop:
            example['content_flag'] = True
            example['style_flag'] = False
        elif r < self.style_drop + self.content_drop:
            example['content_flag'] = False
            example['style_flag'] = True
        else:
            example['content_flag'] = True
            example['style_flag'] = True

        return example


class WikiArtTrain(Base):
    def __init__(self, size=384, crop_size=256, random_augment=True, root='datasets/wiki-art/',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        relpaths = get_all_images(root)['train']
        abspaths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=abspaths,
                               size=size,
                               crop_size=crop_size,
                               random_augment=random_augment)

        print(f'total {len(self)} wikiart training data.')


class WikiArtValidation(Base):
    def __init__(self, size=384, crop_size=256, random_augment=False,
                 root='datasets/wiki-art/', base=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        relpaths = get_all_images(root)['val']
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, crop_size=crop_size, random_augment=random_augment)

        print(f'total {len(self)} wikiart validation data.')


class HybridTrain(Base):
    def __init__(self,
                 style_size=384, style_crop_size=256, content_size=384, content_crop_size=256, random_augment=True,
                 style_root='datasets/wiki-art/', content_root='datasets/ms-coco/',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        style_relpaths = get_all_images(style_root)['train']
        style_abspaths = [os.path.join(style_root, relpath) for relpath in style_relpaths]
        self.style_data = ImagePaths(paths=style_abspaths,
                                     size=style_size,
                                     crop_size=style_crop_size,
                                     random_augment=random_augment)

        content_relpaths = os.listdir(os.path.join(content_root, 'train2017'))
        content_abspaths = [os.path.join(content_root, 'train2017', relpath) for relpath in content_relpaths]
        self.content_data = ImagePaths(paths=content_abspaths,
                                       size=content_size,
                                       crop_size=content_crop_size,
                                       random_augment=random_augment)

        print(f'total {len(self.style_data)} hybrid training data, {len(self.content_data)} content data.')

    def __len__(self):
        return len(self.style_data)

    def __getitem__(self, i):
        example = dict()
        r = random.random()
        if r < self.style_drop:
            example['content_flag'] = True
            example['style_flag'] = False
        elif r < self.style_drop + self.content_drop:
            example['content_flag'] = False
            example['style_flag'] = True
        else:
            example['content_flag'] = True
            example['style_flag'] = True

        if example['style_flag']:
            example.update(self.style_data[i])
        else:
            example.update(self.content_data[random.randint(0, len(self.content_data) - 1)])

        return example
