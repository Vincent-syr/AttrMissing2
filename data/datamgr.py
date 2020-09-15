# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, MultiModalDataset, SetDataset, EpisodicBatchSampler, EpisodicMultiModalSampler
from data.dataset_npy import MiniImgDataset
from abc import abstractmethod

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)

        if 'miniImagenet' in data_file:
            # trans_map = {"base":'train', 'val':'val', "novel":'test'}
            if 'base' in data_file:
                split = 'base'
            else:
                split = 'val' if 'val' in data_file else 'novel'
            # print('split = ', split)
            dataset = MiniImgDataset(split, transform, aux=False)
        else:
            dataset = SimpleDataset(data_file, transform)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, aux=False,  n_episode =100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.trans_loader = TransformLoader(image_size)
        self.aux = aux   # use attribute
        
    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        # print('data_file = ', data_file)
        if not self.aux:     # only image feature
            # read from npy according to am3, only for mini-imagenet
            if 'miniImagenet' in data_file:
                # trans_map = {"base":'train', 'val':'val'}
                if 'base' in data_file:
                    split = 'base'
                else:
                    split = 'val' if 'val' in data_file else 'novel'
                dataset = MiniImgDataset(split, transform, self.aux)
                sampler = EpisodicMultiModalSampler(dataset.label, self.n_way, self.batch_size, self.n_episode)
            else:
                dataset = SimpleDataset(data_file, transform)
                sampler = EpisodicMultiModalSampler(dataset.label, self.n_way, self.batch_size, self.n_episode)
            data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
            # data_loader_params = dict(batch_sampler = sampler,  num_workers = 1, pin_memory = True)       
            data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        else: # use attribute word vector
            if 'miniImagenet' in data_file[0]:
                # trans_map = {"base":'train', 'val':'val'}
                if 'base' in data_file[0]:
                    split = 'base'
                else:
                    split = 'val' if 'val' in data_file[0] else 'novel'

                dataset = MiniImgDataset(split, transform, self.aux)
                sampler = EpisodicMultiModalSampler(dataset.label, self.n_way, self.batch_size, self.n_episode)

            else:
                img_file, attr_file = data_file
                dataset = MultiModalDataset(img_file, attr_file, transform)
                sampler = EpisodicMultiModalSampler(dataset.label, self.n_way, self.batch_size, self.n_episode)

            data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
            data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


