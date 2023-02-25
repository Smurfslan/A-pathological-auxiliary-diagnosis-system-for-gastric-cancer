import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
import glob
import os
from shutil import move, copy


source_dir = '/mnt/ai2019/deng/Data/gastric_s2_s4/train/new_train_img/'
source_img_list = sorted(glob.glob(source_dir + '*' + '.png'))
copy_path = '/mnt/ai2019/deng/Data/gastric_s2_s4/train/cro_new_mask_512/'
copy_to_path = '/mnt/ai2019/deng/Data/gastric_s2_s4/train/new_train_label/'
for i in range(len(source_img_list)):
    img_name = os.path.basename(source_img_list[i]).replace('.png', '_mask.png')
    # img = os.path.basename(test_mask[i])

    copy(os.path.join(copy_path, img_name), copy_to_path)