"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
# from torch import from_numpy
import numpy as np
from skimage import exposure, img_as_ubyte
import cv2
import os.path
from normalizer import normalize, freibeg_crop_ir, freibeg_crop_rgb

def default_loader(path):
    try:
        # print("RGB")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image {path}")
            return None
    except Exception as e:
        print(f"Failed to load image {path}: {e}")
        return None
    image = freibeg_crop_rgb(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(f"Image shape new: {image.shape}")
    return Image.fromarray(image)

# Added by me to load 16 bit IR images
def ir_loader(path):
    try:
        # print("IR")
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Failed to load image {path}")
            return None
    except Exception as e:
        print(f"Failed to load image {path}: {e}")
        return None
    # image = img_as_ubyte(exposure.rescale_intensity(image))
    # image = cv2.equalizeHist(image)
    image = normalize(freibeg_crop_ir(image))
    image = cv2.merge((image, image, image))
    return Image.fromarray(image).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None and img is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None and img is not None:
            img = self.transform(img)
        if img is None:
            print(f"Failed to load image {impath}")
            return None, None
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]

        img = self.loader(path)
        if isinstance(img, type(None)):
            return None
        # print(f"Image shape: {img.size}")    
        if self.transform is not None and img is not None:
            # print("Image shape: ", img.size)
            # if self.loader == ir_loader:
            #     print("IR image")
            img = self.transform(img)

        if self.return_paths and img is not None:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

