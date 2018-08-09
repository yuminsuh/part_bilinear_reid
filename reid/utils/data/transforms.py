from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math

class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)

class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(0.9, 1.1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)

class RGB_to_BGR(object):
    def __call__(self, tensor):
        tensor = tensor[[2,1,0],:,:]
        return tensor

class NormalizeBy(object):
    def __init__(self, max_val=255):
        self.max_val = max_val

    def __call__(self, tensor):
        if self.max_val == 1:
            mean = [0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
        elif self.max_val == 255:
            mean = [104.0/255.0, 117.0/255.0, 123.0/255.0]
            std = [1.0/255.0, 1.0/255.0, 1.0/255.0]
        else:
            raise ValueError('unknown')

        return Normalize(mean=mean, std=std)(tensor)
