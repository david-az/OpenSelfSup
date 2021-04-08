import torch
from torch.utils.data import Dataset

from openselfsup.utils import build_from_cfg

from torchvision.transforms import Compose

from .registry import DATASETS, PIPELINES
from .builder import build_datasource
from .utils import to_numpy
import json
import numpy as np
import cv2
from tqdm import tqdm
import torchvision.transforms as T

@DATASETS.register_module
class BYOLDataset(Dataset):
    """Dataset for BYOL.
    """

    def __init__(self, data_source, pipeline1, pipeline2, prefetch=False):
        self.data_source = build_datasource(data_source)
        pipeline1 = [build_from_cfg(p, PIPELINES) for p in pipeline1]
        self.pipeline1 = Compose(pipeline1)
        pipeline2 = [build_from_cfg(p, PIPELINES) for p in pipeline2]
        self.pipeline2 = Compose(pipeline2)
        self.prefetch = prefetch

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        img1 = self.pipeline1(img)
        img2 = self.pipeline2(img)
        if self.prefetch:
            img1 = torch.from_numpy(to_numpy(img1))
            img2 = torch.from_numpy(to_numpy(img2))

        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented

@DATASETS.register_module
class ImagesDataset(Dataset):
    def __init__(self, ann_file, transform):
        super().__init__()
        with open(ann_file, 'r') as f:
            anns = json.load(f)

        self.paths = [ann['file_name'] for ann in anns['images']]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = cv2.imread(path, -1).astype(np.float32)
        img = torch.tensor(img).unsqueeze(0)
        img = img.expand(3, -1, -1).unsqueeze(0)
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1.squeeze(), view2.squeeze()

@DATASETS.register_module
class PrefetchImagesDataset(Dataset):
    def __init__(self, ann_file, transform, prefetch=True, prefetch_size=300):
        super().__init__()

        self.prefetch = prefetch
        transform = [build_from_cfg(p, PIPELINES) for p in transform]
        self.transform = Compose(transform)
        
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
            img = img.expand(3, -1, -1).unsqueeze(0)
            img = self.resize(img)
            self.torch_images.append(img)
        print(f'Loaded {len(self.torch_images)} images\n')
        

    def __len__(self):
        return len(self.torch_images) if self.prefetch else len(self.paths)

    def __getitem__(self, index):
        if self.prefetch:
            img = self.torch_images[index]
        else:
            path = self.paths[index]
            img = cv2.imread(path, -1).astype(np.float32)
            img = torch.tensor(img).unsqueeze(0)
            img = img.expand(3, -1, -1).unsqueeze(0)

        view1 = self.transform(img)
        view2 = self.transform(img)
        img_cat = torch.cat([view1, view2], dim=0)
        return dict(img=img_cat)