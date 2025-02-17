from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import random
import torchvision.transforms as transforms
random.seed(1)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __ToRGB(img):
    return torch.repeat_interleave(img, repeats=3, dim=0)

def get_transform(params=None, grayscale=True, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    crop_size = 128
    
    transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], crop_size)))

    if convert:
        transform_list += [transforms.ToTensor()]
    transform_list.append(transforms.Lambda(lambda img: __ToRGB(img)))
    return transforms.Compose(transform_list)

def get_params(size, crop_size = 128):
    w, h = size
    new_h = h
    new_w = w
    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))
    return {'crop_pos': (x, y)}

class CropA(object):
    def __init__(self) -> None:
        self.input_nc = 1

    def __call__(self, A):
        transform_params = get_params(A.size)
        A_transform = get_transform(transform_params, grayscale=(self.input_nc == 1))
        return A_transform(A)


class CropA_defined(object):
    def __init__(self, crop_x) -> None:
        self.input_nc = 1
        self.crop_x = crop_x

    def __call__(self, A):
        x = self.crop_x
        y = 0
        transform_params = {'crop_pos': (x, y)}
        A_transform = get_transform(transform_params, grayscale=(self.input_nc == 1))
        return A_transform(A)

class ToTensor(object):
    """Convert numpy array to tensor format. For vhist.
    """
    def __init__(self):
        pass

    def __call__(self, vhist):
        return torch.from_numpy(vhist)

### Label Transforms 
class LabelMap(object):

    def __init__(self, label_type='pattern', ymap=None):
        self.label_type = label_type
        self.ymap = ymap
        
    def __call__(self, des):
        """
        Args:
        des (Dictionary): the descriptor information for the sample
        """
        if self.label_type in des.keys():
            label = des[self.label_type]
        else:
            print('Error! Label type {label_type} not in the label dictionary.')
        if self.ymap is not None:
            label = self.ymap[label]
        return label


class ToOneHot(object):
    """Change an integer label into one-hot label
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def __call__(self, label):
        return F.one_hot(torch.tensor(label, dtype=torch.int64), self.num_classes)

