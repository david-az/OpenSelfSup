import torch
from torch.utils.data import Dataset

from openselfsup.utils import build_from_cfg

from .registry import DATASETS, PIPELINES
from .builder import build_datasource
from .utils import to_numpy
import json
import numpy as np
import cv2
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image

class ComposeCoord(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        coord = None
        for t in self.transforms:
            if 'RandomResizedCropCoord' in t.__class__.__name__:
                img, coord = t(img)
            elif 'FlipCoord' in t.__class__.__name__:
                img, coord = t(img, coord)
            else:
                img = t(img)
        return img, coord

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

@DATASETS.register_module
class PixProDataset(Dataset):
    def __init__(self, ann_file, transform, in_channels=3, prefetch=True, prefetch_size=300):
        super().__init__()

        self.prefetch = prefetch
        self.in_channels = in_channels
        transform = [build_from_cfg(p, PIPELINES) for p in transform]
        self.transform = ComposeCoord(transform)
        
        with open(ann_file, 'r') as f:
            anns = json.load(f)

        self.paths = [ann['file_name'] for ann in anns['images']]
        self.resize = T.Resize(prefetch_size)

        if self.prefetch:
            self.prefetch_images()

    def prefetch_images(self):
        self.torch_images = []
        print(f'Loading {len(self.paths)} images\n')
        for path in tqdm(self.paths):
            try:
                img = cv2.imread(path, -1).astype(np.float32)
            except:
                continue
            img = torch.tensor(img).unsqueeze(0)
            img = self.resize(img)
            self.torch_images.append(img)
        print(f'Loaded {len(self.torch_images)} images\n')
        

    def __len__(self):
        return len(self.torch_images) if self.prefetch else len(self.paths)

    def __getitem__(self, index):
        if self.prefetch:
            img = self.torch_images[index]
            img = img.expand(1, self.in_channels, -1, -1)

        else:
            path = self.paths[index]
            with open(path, 'rb') as f:
                img = Image.open(f).convert('F')
            # img = cv2.imread(path, -1).astype(np.float32)
            # img = torch.tensor(img).unsqueeze(0)
            # img = img.expand(self.in_channels, -1, -1).unsqueeze(0)

        view1, coord1 = self.transform(img)
        view2, coord2 = self.transform(img)

        return dict(im_1=view1, im_2=view2, coord1=coord1, coord2=coord2)