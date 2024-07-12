#!/usr/bin/env python
import sys
sys.path.append("..")

from utils import cv2_trans as transforms
from termcolor import cprint
import cv2
import torchvision
import torch.utils.data as data
import torch
import random
import numpy as np
import os
import warnings


class MagTrainDataset(data.Dataset):
    def __init__(self, ann_file, transform=None):
        self.ann_file = ann_file
        self.transform = transform
        self.init()

    def init(self):
        self.weight = {}
        self.im_names = []
        self.targets = []
        self.pre_types = []
        with open(self.ann_file) as f:
            for line in f.readlines():
                data = line.strip().split(' ')
                self.im_names.append(data[0])
                self.targets.append(int(data[2]))

    def __getitem__(self, index):
        im_name = self.im_names[index]
        target = self.targets[index]
        # Check for the range and unique target values
        #unique_targets = set(self.targets)
        #print(f"Number of unique targets: {len(unique_targets)}")
        #print(f"Max target value: {max(unique_targets)}")
        #print(f"Min target value: {min(unique_targets)}")
        #print("TARGETS!!", target)
        img = cv2.imread(im_name)
        
        ## NEW ----
        # Check if the image was loaded correctly
        if img is None:
            print(f"Warning: Image at path {im_name} could not be loaded.")
            # Handle the error as you see fit, for example, return a dummy image
            img = np.zeros((112, 112, 3), dtype=np.uint8)  # Assuming your images are of size 224x224

        if self.transform:
            img = self.transform(img)
        #print("TARGET RETURNED", target)
        return img, target
        ## ----

        # img = self.transform(img)
        # return img, target

    def __len__(self):
        return len(self.im_names)


def train_loader(args):
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        
    ])
    train_dataset = MagTrainDataset(
        args.train_list,
        transform=train_trans
    )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=(train_sampler is None))

    return train_loader
