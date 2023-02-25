import os
from openslide import OpenSlide
import cv2
import glob
import csv
import sys
from libtiff import TIFF
from skimage import color, io, transform
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
import segmentation_models_pytorch as smp
import pandas as pd
import heapq
import argparse


def net_load(model_path, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    net_load_time = time.time()
    net = smp.DeepLabV3Plus(encoder_name='efficientnet-b3', encoder_weights=None, classes=2)
    # net = smp.UnetPlusPlus(encoder_name='efficientnet-b3', encoder_weights=None, classes=2)
    # net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    print('net_load_time:', time.time() - net_load_time)
    return net


def Calculate_pre_score(pre_np_0_255, rescale_size):
    pix_total = rescale_size * rescale_size
    number_tumor_pixel = np.sum(pre_np_0_255==255)

    return number_tumor_pixel/pix_total


def cancer(i):
    # (mean, stddv) = cv2.meanStdDev(i[:,:,0])
    # mean = np.mean(i[:,:,0])/255
    # stddv = np.std(i[:,:,0])/255

    mean2 = np.mean(i) / 255
    stddv2 = np.std(i) / 255

    # mean2 = np.mean(i)
    # stddv2 = np.std(i)

    if float(mean2) > 0.001 and float(stddv2) > 0.015:
        return 1
    else:
        return 0


def cancer2(mask, flag, pix_num):
    number = np.sum(mask ==flag)
    if number > pix_num:
        return 1
    else:
        return 0


def wsi_pre_list(wsi_path, net, rescale_size, matrix_size, top_pixel=1000):

    try:
        wsi_img = OpenSlide(wsi_path)

        image_name = os.path.basename(wsi_path)
        # pre_list = []
        # pre_array = np.zeros((30, 30), dtype=np.float64)
        pre_array = np.zeros((matrix_size, matrix_size), dtype=np.float64)
        cancer_bool_list = []
        cancer_flag_list = [0 for _ in range(top_pixel)]

        for i in range(0, wsi_img.level_dimensions[0][1] - 4096, 4096):
            for j in range(0, wsi_img.level_dimensions[0][0] - 4096, 4096):
                patch = wsi_img.read_region((j, i), 1, (2048, 2048))
                patch = np.array(patch)
                patch = patch[:, :, :3]

                # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                re_image = cv2.resize(patch, (rescale_size, rescale_size), interpolation=cv2.INTER_AREA) / 255

                tmpImg = np.zeros((re_image.shape[0], re_image.shape[1], 3))
                re_image = re_image / np.max(re_image)
                tmpImg[:, :, 0] = (re_image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (re_image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (re_image[:, :, 2] - 0.406) / 0.225
                tmpImg = tmpImg.transpose((2, 0, 1))
                tmpImg = tmpImg[:, :, ::-1].copy()

                std = np.std(re_image[:, :, 0])

                if std < (5 / 255):
                    cancer_bool = 0
                else:
                    img = torch.from_numpy(tmpImg).type(torch.FloatTensor).unsqueeze(0)

                    if torch.cuda.is_available():
                        inputs_test = Variable(img.cuda())
                    else:
                        inputs_test = Variable(img)

                    d1 = net(inputs_test)

                    d1_heat = F.softmax(torch.squeeze(d1), dim=0)[1].to('cpu').detach().numpy()
                    d1_list = sum(d1_heat.tolist(), [])
                    max_1000 = heapq.nlargest(top_pixel, d1_list)
                    list_tmap = max_1000 + cancer_flag_list
                    cancer_flag_list = heapq.nlargest(top_pixel, list_tmap)

                    d1_divide_0_1 = d1.flip([-1]).argmax(1).squeeze(0)
                    # d1_divide_0_1=d1_divide_0_1.squeeze(0)
                    img_np_0_1 = d1_divide_0_1.cpu().numpy()
                    img_np_0_255 = (img_np_0_1 * 255).astype('uint8')

                    pre_score = Calculate_pre_score(img_np_0_255, rescale_size)
                    # pre_list.append(pre_score)
                    w = int(i/4096)
                    h = int(j/4096)
                    pre_array[w][h] = pre_score

                    cancer_bool = cancer(img_np_0_255)
                    cancer_bool_list.append(cancer_bool)

        # if 1 in cancer_bool_list:
        #     ori_pre_wsi = 1
        # else:
        #     ori_pre_wsi = 0

        pixel_mean = np.mean(cancer_flag_list)
        if pixel_mean >= 0.5:
            ori_pre_wsi = 1
        else:
            ori_pre_wsi = 0

        return image_name, pre_array, ori_pre_wsi

    except:
        return wsi_path, None, None


def wsi_pre_list_6(wsi_path, net, rescale_size, matrix_size, top_pixel=1000):

    try:
        wsi_img = OpenSlide(wsi_path)

        image_name = os.path.basename(wsi_path)
        # pre_list = []
        # pre_array = np.zeros((30, 30), dtype=np.float64)
        pre_array = np.zeros((matrix_size, matrix_size), dtype=np.float64)
        cancer_bool_list = []
        cancer_flag_list = [0 for _ in range(top_pixel)]

        for i in range(0, wsi_img.level_dimensions[0][1]//2 - 4096, 4096):
            for j in range(0, wsi_img.level_dimensions[0][0]//3 - 4096, 4096):
                patch = wsi_img.read_region((j, i), 1, (2048, 2048))
                patch = np.array(patch)
                patch = patch[:, :, :3]

                # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                re_image = cv2.resize(patch, (rescale_size, rescale_size), interpolation=cv2.INTER_AREA) / 255

                tmpImg = np.zeros((re_image.shape[0], re_image.shape[1], 3))
                re_image = re_image / np.max(re_image)
                tmpImg[:, :, 0] = (re_image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (re_image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (re_image[:, :, 2] - 0.406) / 0.225
                tmpImg = tmpImg.transpose((2, 0, 1))
                tmpImg = tmpImg[:, :, ::-1].copy()

                std = np.std(re_image[:, :, 0])

                if std < (5 / 255):
                    cancer_bool = 0
                else:
                    img = torch.from_numpy(tmpImg).type(torch.FloatTensor).unsqueeze(0)

                    if torch.cuda.is_available():
                        inputs_test = Variable(img.cuda())
                    else:
                        inputs_test = Variable(img)

                    d1 = net(inputs_test)

                    d1_heat = F.softmax(torch.squeeze(d1), dim=0)[1].to('cpu').detach().numpy()
                    d1_list = sum(d1_heat.tolist(), [])
                    max_1000 = heapq.nlargest(top_pixel, d1_list)
                    list_tmap = max_1000 + cancer_flag_list
                    cancer_flag_list = heapq.nlargest(top_pixel, list_tmap)

                    d1_divide_0_1 = d1.flip([-1]).argmax(1).squeeze(0)
                    # d1_divide_0_1=d1_divide_0_1.squeeze(0)
                    img_np_0_1 = d1_divide_0_1.cpu().numpy()
                    img_np_0_255 = (img_np_0_1 * 255).astype('uint8')

                    pre_score = Calculate_pre_score(img_np_0_255, rescale_size)
                    # pre_list.append(pre_score)
                    w = int(i/4096)
                    h = int(j/4096)
                    pre_array[w][h] = pre_score

                    cancer_bool = cancer(img_np_0_255)
                    cancer_bool_list.append(cancer_bool)

        # if 1 in cancer_bool_list:
        #     ori_pre_wsi = 1
        # else:
        #     ori_pre_wsi = 0

        pixel_mean = np.mean(cancer_flag_list)
        if pixel_mean >= 0.5:
            ori_pre_wsi = 1
        else:
            ori_pre_wsi = 0

        return image_name, pre_array, ori_pre_wsi

    except:
        return wsi_path, None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch extract array Framework')
    parser.add_argument('--cuda_device', '-cuda', type=str, default='4', help='cuda_device_number')
    parser.add_argument('--seg_model', type=str,
                        default='/mnt/ai2019/ljl/code/software_platform/train/torch_framework/paper/patch/Test1_Semi_ts_0.3_DeepLabV3Plus_b3_epoch300_bs16_lr0.0005_optAdamW_schedulerCosineAnnealingLR_mixloss/model/Test1_Semi_ts_0.3_DeepLabV3Plus_b3_epoch300_bs16_lr0.0005_optAdamW_schedulerCosineAnnealingLR_mixloss_itr_39000.pth',
                        help='trained_seg_pth')
    parser.add_argument('--test_file', type=str,
                        default='/mnt/ai2020/ljl/data/gastric/paper_test/3.7_small/normal_tif/',
                        help='tif_image_file')
    parser.add_argument('--csv_flie', type=str,
                        default='./matrix/external_test/CycleGAN_aug_shen_qian_230104_model/CycleGAN_aug_shen_qian_230104_classification_small_model221229_depv3+_ef3.csv',
                        help='save_csv_path')
    parser.add_argument('--tumor_flag', type=int, default=0, help='normal:0, tumor:1')
    parser.add_argument('--image_num', type=int, default=1, help='the number of tissue per slide. 1 or 6')
    parser.add_argument('--big_or_small', type=str, default='small', help='big_or_small')

    args = parser.parse_args()

    net = net_load(args.seg_model, args)
    # test_file = '/mnt/ai2020/ljl/data/gastric/paper_test/3.7_small/normal_tif/'
    # csv_flie = './matrix/small/CycleGAN_aug_quan_221229_classification_small_model221229_depv3+_ef3.csv'
    # tumor_flag = 0
    # image_num = 1

    if args.big_or_small == 'big':
        top_pixel = 5000
        matrix_size = 100
    elif args.big_or_small == 'small':
        top_pixel = 1000
        matrix_size = 30

    test_img_path = sorted(glob.glob(args.test_file + '*.tif'))

    image_name1 = []
    pre_array1 = []
    gt1 = []
    ori_pre_wsi1 = []
    np.set_printoptions(threshold=np.inf)

    with tqdm(total=len(test_img_path)) as t:
        for wsi_path in test_img_path:
            if args.image_num == 6:
                image_name, pre_array, ori_pre_wsi = wsi_pre_list_6(wsi_path, net, rescale_size=512, matrix_size=matrix_size, top_pixel=top_pixel)
            else :
                image_name, pre_array, ori_pre_wsi = wsi_pre_list(wsi_path, net, rescale_size=512,
                                                                  matrix_size=matrix_size, top_pixel=top_pixel)
            image_name1.append(image_name)
            pre_array1.append(pre_array)
            gt1.append(args.tumor_flag)
            ori_pre_wsi1.append(ori_pre_wsi)
            t.update()

    pre_info = pd.DataFrame()
    pre_info['image_name'] = image_name1
    pre_info['pre_array'] = pre_array1
    pre_info['gt'] = gt1
    pre_info['ori_pre_wsi'] = ori_pre_wsi1

    pre_info.to_csv(args.csv_flie, mode='a', index=False)
