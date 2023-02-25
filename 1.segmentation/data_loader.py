# data loader
from __future__ import print_function, division
from utils import *
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
from tools import myTransforms
import cv2
import os


def augment_HE(img, GaussBlur=True, HED=True):
    GaussBlur = GaussBlur and random.random() < 0.5
    HED = HED and random.random() < 0.5
    preprocess1 = myTransforms.HEDJitter(theta=0.05)
    preprocess2 = myTransforms.RandomGaussBlur(radius=[0.5, 1.5])

    if GaussBlur:
        img = Image.fromarray(img)
        img = preprocess1(img)
        img = np.array(img)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = img.astype(np.float32) / 255.

    if HED:
        img = Image.fromarray(img)
        img = preprocess2(img)
        img = np.array(img)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = img.astype(np.float32) / 255.

    return img

    # return [_augment(img) for img in img_list]


# ==========================dataset load==========================
class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0,
                               preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl}


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        img = transform.resize(image, (new_h, new_w), mode='constant')
        lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl}


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'imidx': imidx, 'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)

        image = image / np.max(image)
        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpLbl = np.zeros(label.shape)

        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                    np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                    np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                    np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                    np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                    np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                    np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                    np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                    np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                    np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpImg = tmpImg[:, :, ::-1].copy()
        tmpLbl = label.transpose((2, 0, 1))
        tmpLbl = tmpLbl[:, :, ::-1].copy()

        return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, augment_HE=False, transform=None):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform
        self.augment_HE = augment_HE

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):

        # image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
        # label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

        # image = io.imread(self.image_name_list[idx])
        image = cv2.imread(self.image_name_list[idx])

        imname = self.image_name_list[idx]
        imidx = np.array([idx])

        # io.imshow(image)
        # plt.show()

        if self.augment_HE:
            image = augment_HE(image)

        # io.imshow(image)
        # plt.show()

        if (0 == len(self.label_name_list)):
            label_3 = np.zeros(image.shape)
        else:
            # label_3 = io.imread(self.label_name_list[idx])
            label_3 = cv2.imread(self.label_name_list[idx])

        label = np.zeros(label_3.shape[0:2])
        if (3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif (2 == len(label_3.shape)):
            label = label_3

        if (3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(label.shape)):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        # tmpLbl = np.zeros(label.shape)
        # if (np.max(label) < 1e-6):
        #     label = label
        # else:
        #     label = label / np.max(label)
        # tmpLbl[:, :, 0] = label[:, :, 0]
        # tmpLbl = label.transpose((2, 0, 1))
        # tmpLbl = tmpLbl[:, :, ::-1].copy()

        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        # label = normalization(label)

        label = label.transpose((2, 0, 1)).astype(np.uint8)

        sample = {'imidx': imidx, 'image': image, 'label': label}

        if self.transform:
            augmented = self.transform(**sample)
            imidx, image, label = augmented['imidx'], augmented['image'], augmented['label']
            sample = {'imidx': imidx, 'image': image, 'label': label}

        return sample


class MyDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_name_list[idx])[:, :, ::-1]
        label = cv2.imread(self.label_name_list[idx])[:, :, 0]
        imidx = np.array([idx])

        if self.transform:
            img_mask = self.transform(image=image, mask=label)
            image = img_mask['image']
            label = img_mask['mask']

        sample = {'imidx': imidx, 'image': image, 'label': (label/255).long()}

        return sample


class MyDataset_mean_teacher(Dataset):
    def __init__(self, img_name_list, lbl_name_list, img_name_list_ema, train_transform=None, test_transform=None):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.img_name_list_ema = img_name_list_ema

        self.train_transform = train_transform
        self.test_transform = test_transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_name_list[idx])[:, :, ::-1]
        label = cv2.imread(self.label_name_list[idx])[:, :, 0]
        len_ema = len(self.img_name_list_ema)
        idx_ema = random.randint(0, len_ema - 1)
        img_ema = cv2.imread(self.img_name_list_ema[idx_ema])[:, :, ::-1]
        imidx = np.array([idx])

        # if self.transform:
        #     img_mask = self.transform(image=image, mask=label)
        #     image = img_mask['image']
        #     label = img_mask['mask']
        if random.random()>0.5:
            image = image[:, ::-1, :]
            img_ema = img_ema[:, ::-1, :]
            label = label[:, ::-1]
        if random.random()>0.5:
            image = image[::-1, :, :]
            img_ema = img_ema[::-1, :, :]
            label = label[::-1, :]
        if random.random()>0.5:
            image = image.transpose(1, 0, 2)
            img_ema = img_ema.transpose(1, 0, 2)
            label = label.transpose(1, 0)
        img_mask = self.train_transform(image=image.copy(), mask=label.copy())
        img_mask2 = self.train_transform(image=img_ema.copy())
        sample = {'imidx': imidx, 'image': img_mask['image'], 'label': (img_mask['mask']/255).long(), 'img_ema': img_mask2['image']}

        return sample



class MyDataset_Server(Dataset):
    def __init__(self, img_list, mask_list, transform=None):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        image = self.img_list[idx]
        label = self.mask_list[idx]
        imidx = np.array([idx])

        if self.transform:
            img_mask = self.transform(image=image, mask=label)
            image = img_mask['image']
            label = img_mask['mask']

        sample = {'imidx': imidx, 'image': image, 'label': (label/255).long()}

        return sample


if __name__ == '__main__':
    image_path = "/mnt/ai2019/ljl/data/gastric/total/2048/dyl/train/16-20076-5.6zhong_46.jpg"
    image = io.imread(image_path)
    save_path = "/mnt/ai2019/ljl/code/software_platform/train/torch_framework/experiments/U2NetP_Augument_HE_Test/val_img"
    imagename = os.path.basename(image_path)
    io.imsave(os.path.join(save_path, imagename), image)
    image = augment_HE(image)
    io.imsave(os.path.join(save_path, imagename.split(".jpg")[0] + "_trans.jpg"), image)
