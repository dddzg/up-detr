# ------------------------------------------------------------------------
# UP-DETR
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
pre-training dataset which implements random query patch detection.
"""
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np
import datasets.transforms as T
from torchvision.transforms import transforms
from PIL import ImageFilter
import random

def get_random_patch_from_img(img, min_pixel=8):
    """
    :param img: original image
    :param min_pixel: min pixels of the query patch
    :return: query_patch,x,y,w,h
    """
    w, h = img.size
    min_w, max_w = min_pixel, w - min_pixel
    min_h, max_h = min_pixel, h - min_pixel
    sw, sh = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
    x, y = np.random.randint(w - sw) if sw != w else 0, np.random.randint(h - sh) if sh != h else 0
    patch = img.crop((x, y, x + sw, y + sh))
    return patch, x, y, sw, sh


class SelfDet(Dataset):
    """
    SelfDet is a dataset class which implements random query patch detection.
    It randomly crops patches as queries from the given image with the corresponding bounding box.
    The format of the bounding box is same to COCO.
    """
    def __init__(self, root, detection_transform, query_transform, num_patches=10):
        super(SelfDet, self).__init__()
        self.root = root
        self.detection_transform = detection_transform
        self.query_transform = query_transform
        self.files = []
        self.num_patches = num_patches
        for (troot, _, files) in os.walk(root, followlinks=True):
            for f in files:
                path = os.path.join(troot, f)
                self.files.append(path)
        print(f'num of files:{len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img_path = self.files[item]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if w<=16 or h<=16:
            return self[(item+1)%len(self)]
        # the format of the dataset is same with COCO.
        target = {'orig_size': torch.as_tensor([int(h), int(w)]), 'size': torch.as_tensor([int(h), int(w)])}
        iscrowd = []
        labels = []
        boxes = []
        area = []
        patches = []
        while len(area) < self.num_patches:
            patch, x, y, sw, sh = get_random_patch_from_img(img)
            boxes.append([x, y, x + sw, y + sh])
            area.append(sw * sh)
            iscrowd.append(0)
            labels.append(1)
            patches.append(self.query_transform(patch))
        target['iscrowd'] = torch.tensor(iscrowd)
        target['labels'] = torch.tensor(labels)
        target['boxes'] = torch.tensor(boxes)
        target['area'] = torch.tensor(area)
        img, target = self.detection_transform(img, target)
        return img, torch.stack(patches, dim=0), target


def make_self_det_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # The image of ImageNet is relatively small.
    scales = [320, 336, 352, 368, 400, 416, 432, 448, 464, 480]

    if image_set == 'train':
        return T.Compose([
            # T.RandomHorizontalFlip(), HorizontalFlip may cause the pretext too difficult, so we remove it
            T.RandomResize(scales, max_size=600),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([480], max_size=600),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')




class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_query_transforms(image_set):
    if image_set == 'train':
        # SimCLR style augmentation
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    if image_set == 'val':
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    raise ValueError(f'unknown {image_set}')


def build_selfdet(image_set, args):
    return SelfDet(args.imagenet_path, detection_transform=make_self_det_transforms(image_set),
                   query_transform=get_query_transforms(image_set), num_patches=args.num_patches)
