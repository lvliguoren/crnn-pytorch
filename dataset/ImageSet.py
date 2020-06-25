import torch.utils.data as data
import cv2
import os
from torchvision import transforms
import torch
import math
import numpy as np


def default_loader(path):
    img = cv2.imread(path)
    H, W, C = img.shape
    scale = 32 / H
    img = cv2.resize(img, dsize=(math.floor(W * scale), 32), fx=1, fy=1)
    # 转为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return transforms.ToTensor()(img)


class CustomData(data.Dataset):
    def __init__(self, root, char_dict,is_train=True, loader=default_loader):
        super(CustomData, self).__init__()
        self.root = root
        self.loader = loader
        self.img_name = []
        self.labels = []
        self.char_dict = char_dict

        if is_train:
            with open('dataset/txt/train.txt', 'r') as file:
                for line in file.readlines():
                    self.img_name.append(line.split()[0])
                    label = line.split()[1:]
                    self.labels.append(','.join(label))
        else:
            with open('dataset/txt/test.txt', 'r') as file:
                for line in file.readlines():
                    self.img_name.append(line.split()[0])
                    label = line.split()[1:]
                    self.labels.append(','.join(label))


    def __len__(self):
        return len(self.img_name)


    def __getitem__(self, index):
        path = os.path.join(self.root, self.img_name[index])
        img = self.loader(path)
        label = self.labels[index]

        return img, label