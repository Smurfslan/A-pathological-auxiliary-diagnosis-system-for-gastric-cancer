import glob
import torch
import skimage.io
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
import os
import ast
import re


class MyDataset(Dataset):
    def __init__(self, img_list, mask_list, transform=None):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = self.img_list[idx]
        image_array = ast.literal_eval(re.sub(r'(\d?)(?:\r\n)*\s+', r'\1, ', image))
        image_array = np.array(image_array)
        image_array = cv2.cvtColor(image_array.astype(np.float32), cv2.COLOR_GRAY2BGR)
        # image_array = image_array[:, :, np.newaxis]
        mask = np.array(self.mask_list[idx])
        imidx = np.array([idx])

        if self.transform:
            img_mask = self.transform(image=image_array)
            image_array = img_mask['image']
            # mask = img_mask['mask']

        sample = {'imidx': imidx, 'image': image_array, 'label': torch.from_numpy(mask)}

        return sample


class MyDataset_test(Dataset):
    def __init__(self, img_list, mask_list, test_ori, img_name_list, transform=None):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.img_list = img_list
        self.mask_list = mask_list
        self.test_ori = test_ori
        self.img_name_list = img_name_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = self.img_list[idx]
        image_array = ast.literal_eval(re.sub(r'(\d?)(?:\r\n)*\s+', r'\1, ', image))
        image_array = np.array(image_array)
        image_array = cv2.cvtColor(image_array.astype(np.float32), cv2.COLOR_GRAY2BGR)
        # image_array = image_array[:, :, np.newaxis]
        mask = int(self.mask_list[idx])
        imidx = np.array([idx])
        pre_ori = int(self.test_ori[idx])
        image_name = str(self.img_name_list[idx])

        if self.transform:
            img_mask = self.transform(image=image_array)
            image_array = img_mask['image']
            # mask = img_mask['mask']

        sample = {'imidx': imidx, 'image': image_array, 'label': mask, 'pre_ori': pre_ori, 'image_name': image_name}

        return sample
