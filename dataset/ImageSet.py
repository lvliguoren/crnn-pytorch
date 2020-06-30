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
    img = transforms.ToTensor()(img)
    img = img.sub(0.5).div(0.5)
    return img


class CustomData(data.Dataset):
    def __init__(self, root, char_dict,is_train=True, loader=default_loader):
        super(CustomData, self).__init__()
        self.root = root
        self.loader = loader
        self.img_name = []
        self.labels = []
        self.char_dict = char_dict

        if is_train:
            self.__getlabels('train.txt')
        else:
            self.__getlabels('test.txt')


    def __len__(self):
        return len(self.img_name)


    def __getitem__(self, index):
        path = os.path.join(self.root, self.img_name[index])
        img = self.loader(path)
        label = self.labels[index]

        return img, label

    def __getlabels(self, file_name):
        with open('dataset/txt/'+ file_name, 'r', encoding='UTF-8') as file:
            for line in file.readlines():
                self.img_name.append(line.split()[0])
                label = line.split()[1]
                word_index = []
                for word in label:
                    word_index.append(str(self.char_dict[word]))
                self.labels.append(','.join(word_index))

    @staticmethod
    def get_file(file_path):
        img = default_loader(file_path)
        C, H, W = img.shape
        img = img.view(1, C, H, W)
        return img


if __name__ == '__main__':
    root = 'E:/TEST/Synthetic Chinese String Dataset/images'
    img_name = '72687140_2765922188.jpg'
    path = os.path.join(root,img_name)
    cv2.imshow("test", default_loader(path))
    cv2.waitKey()