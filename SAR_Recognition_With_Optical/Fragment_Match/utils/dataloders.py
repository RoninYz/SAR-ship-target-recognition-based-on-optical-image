# -*- coding: utf-8 -*-
# @Time    : 2024/3/10 0010 12:55
# @Author  : Ronin
# @File    : dataloders.py
# @Software: PyCharm
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os


class LoadImagesAndLabels(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((100, 100))])
        self.OIMG_dir = os.path.join(root_dir, 'Optical')
        self.SIMG_dir = os.path.join(root_dir, 'SAR')
        self.Label_dir = os.path.join(root_dir, 'Label')
        self.OIMG_names = os.listdir(self.OIMG_dir)
        self.SIMG_names = os.listdir(self.SIMG_dir)
        self.Label_names = os.listdir(self.Label_dir)
        lis = list(zip(self.Label_names, self.OIMG_names, self.SIMG_names))
        random.shuffle(lis)
        self.Label_names, self.OIMG_names, self.SIMG_names = zip(*lis)

    def __getitem__(self, index):
        OIMG_name = self.OIMG_names[index]
        SIMG_name = self.SIMG_names[index]
        Label_name = self.Label_names[index]
        OIMG_path = os.path.join(self.OIMG_dir, OIMG_name)
        SIMG_path = os.path.join(self.SIMG_dir, SIMG_name)
        Label_path = os.path.join(self.Label_dir, Label_name)
        OIMG = self.transform(Image.open(OIMG_path).convert('L'))
        SIMG = self.transform(Image.open(SIMG_path).convert('L'))
        with open(Label_path, 'r') as f:
            fit = torch.tensor(int(f.readline().strip()))
        Label = torch.tensor([0, 1]) if fit == 1 else torch.tensor([1, 0])
        return OIMG, SIMG, Label, fit

    def __len__(self):
        return len(self.OIMG_names)
