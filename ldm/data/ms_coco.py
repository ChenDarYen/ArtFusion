import os

from torch.utils.data import Dataset

from ldm.data.base import ImagePaths


class MSCOCOBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        example['style_flag'] = False
        example['content_flag'] = True
        return example


class MSCOCOTrain(MSCOCOBase):
    def __init__(self, size=384, crop_size=256, random_crop=True, root='/ssd_data/datasets/ms_coco_384/',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        relpaths = os.listdir(os.path.join(root, 'train2017'))
        abspaths = [os.path.join(root, 'train2017', relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=abspaths,
                               size=size,
                               crop_size=crop_size,
                               random_crop=random_crop)

        print(f'total {len(self.data)} MS-COCO training data.')


class MSCOCOValidation(MSCOCOBase):
    def __init__(self, size=384, crop_size=256, random_crop=False,
                 root='/ssd_data/datasets/ms_coco_384/', base=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        relpaths = os.listdir(os.path.join(root, 'val2017'))
        abspaths = [os.path.join(root, 'val2017', relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=abspaths, size=size, crop_size=crop_size, random_crop=random_crop)

        print(f'total {len(self.data)} MS-COCO validation data.')
