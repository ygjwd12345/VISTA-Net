###########################################################################
# Created by: Dan Xu
# Email: danxu@robots.ox.ac.uk
# Copyright (c) 2019
###########################################################################

import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data

class KITTIDataset(data.Dataset):
    NUM_CLASS = 1
    def __init__(self, **db_params):
        '''Initialization'''
        self.data_root = db_params['data_root']
        self.list_IDs = db_params['list_IDs']
        self.db_params = db_params
        self.train = db_params['train']
        self.output_size = db_params['output_size']
        self.data_aug_ = db_params['data_augmentation']
        self.mean_values = db_params['image_mean']

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        # Select sample
        sample_data = self.create_one_sample(index, self.list_IDs)
        return sample_data

    def create_one_sample(self, index, list_IDs):
        #read one line from the list
        rgb_file_pth = list_IDs[index].strip().split(' ')[0]

        #load data
        rgb_img = cv2.imread(osp.join(self.data_root, rgb_file_pth))
        rgb_img = np.float32(rgb_img)

        #processing
        img = cv2.resize(rgb_img, (self.output_size[1], self.output_size[0]))
        img_ori = img / 255.0
        img -= self.mean_values
        img = img[...,::-1] / 255.0
        img = np.transpose(img, (2, 0, 1))
        
        depth = []
        if self.train:
            gt_file_pth = list_IDs[index].strip().split(' ')[1]
            gt_depth = cv2.imread(osp.join(self.data_root, gt_file_pth), -1)
            gt_depth = np.float32(gt_depth) / 100.
            depth = cv2.resize(gt_depth, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)
            depth = depth[np.newaxis, ...]

        if self.data_aug_ and self.train:
            flip = np.random.choice(2)*2-1
            img = img[:, ::flip]
            depth = depth[:, ::flip]

        rgb_out = torch.from_numpy(img.copy())
        #depth_out = torch.from_numpy(depth.copy())
        
        if self.train:
            depth_out = torch.from_numpy(depth.copy())
            return rgb_out, depth_out
        #return rgb_out, torch.from_numpy(img_ori.copy())
        return rgb_out
