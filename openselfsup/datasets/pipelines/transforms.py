from __future__ import division

import cv2
import inspect
import numpy as np
from PIL import Image, ImageFilter
import torch
import math
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import warnings

from torchvision.transforms import functional as F


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


import torch
from torchvision import transforms as _transforms

from openselfsup.utils import build_from_cfg

from ..registry import PIPELINES

# register all existing transforms in torchvision
_EXCLUDED_TRANSFORMS = ['GaussianBlur']
for m in inspect.getmembers(_transforms, inspect.isclass):
    if m[0] not in _EXCLUDED_TRANSFORMS:
        PIPELINES.register_module(m[1])

def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

@PIPELINES.register_module
class RandomHorizontalFlipCoord(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[0] = coord[2]
            coord_new[2] = coord[0]
            return F.hflip(img), coord_new
        return img, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


@PIPELINES.register_module
class RandomVerticalFlipCoord(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[1] = coord[3]
            coord_new[3] = coord[1]
            return F.vflip(img), coord_new
        return img, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

@PIPELINES.register_module
class RandomResizedCropCoord(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w, height, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w, height, width = self.get_params(img, self.scale, self.ratio)
        coord = torch.Tensor([float(j) / (width - 1), float(i) / (height - 1),
                              float(j + w - 1) / (width - 1), float(i + h - 1) / (height - 1)])
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), coord

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


@PIPELINES.register_module
class RandomAppliedTrans(object):
    """Randomly applied transformations.

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


# custom transforms
@PIPELINES.register_module
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)."""

    _IMAGENET_PCA = {
        'eigval':
        torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec':
        torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }

    def __init__(self):
        self.alphastd = 0.1
        self.eigval = self._IMAGENET_PCA['eigval']
        self.eigvec = self._IMAGENET_PCA['eigvec']

    def __call__(self, img):
        assert isinstance(img, torch.Tensor), \
            "Expect torch.Tensor, got {}".format(type(img))
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class Solarization(object):
    """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733."""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 -img)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module
class NormalizeMeanVar(object):
    def __call__(self, img):
        return (img - img.mean([1, 2, 3], True)) / img.std([1, 2, 3], keepdim=True)


@PIPELINES.register_module
class NormalizeMeanVar2(object):
    def __call__(self, img):
        return (img - img.mean()) / img.std()

@PIPELINES.register_module
class NormalizeMinMax(object):
    def __call__(self, img):
        return (img - img.min()) / (img.max() - img.min())