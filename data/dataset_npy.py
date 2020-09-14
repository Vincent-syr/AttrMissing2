# API design
import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os


class MiniImgDataset:
    def __init__(self, split, transform, aux):
        # split: train/val/test
        data_dir = '/test/0Dataset_others/Dataset/mini-imagenet-am3'
        img_path = os.path.join(data_dir, 'few-shot-{:}.npz'.format(split))
        attr_path = os.path.join(data_dir, 'few-shot-wordemb-{:}.npz'.format(split))
        self.attr_all = None
        self.aux = aux
        if aux:
            attr_all = np.load(attr_path)['features'].astype('float32')
            self.attr_all = torch.from_numpy(attr_all)

        self.img_all, self.label = np.load(img_path)['features'], np.load(img_path)['targets']
        # self.img_all = self.img_all.astype('float32')
        self.transform = transform

    def __getitem__(self, i):
        # img = np.transpose(self.img_all[i], (2,0,1))  # (3, 84,84,84)
        img = Image.fromarray(self.img_all[i], mode='RGB')
        img = self.transform(img)
        target = self.label[i]
        if self.aux:
            attr = self.attr_all[self.label[i]]
            x = (img, attr)
        else:
            x = img
        
        return x, target
        

    def __len__(self):
        return len(self.label)