import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
import glob
import os
import sys
def image_hist(image, i):
    color = ('blue', 'green', 'red')
    for ii, color in enumerate(color):
        hist = cv2.calcHist([image], [ii], None, [256], [0,256],accumulate=True)

        plt.plot(hist, color=color)
        plt.xlim([0,256])
        plt.title(os.path.basename(i))
        save_path = '/mnt/ai2019/deng/project/experiments/hist_test/' + os.path.splitext(os.path.basename(i))[0]+'_hist.png'
        plt.savefig(save_path)
    # plt.show()


train_dir = '/mnt/ai2019/deng/Data/gastric_s2_s4/train/clean_data_new/train_img_512/'
save_dir = '//mnt/ai2019/deng/Data/gastric_s2_s4/train/clean_data_new/hue_141_img/'

tra_img_name_list = sorted(glob.glob(train_dir + '*' + '.png'))
for i, ll in enumerate(tra_img_name_list):

    sys.stdout.write('\r%d/%s' % (i, len(tra_img_name_list)))

    image = cv2.imread(ll)
    out_img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    a = out_img_HSV.copy()

    a[:, :, 0] = 141  # change here to alter image Hue value


    a_img = cv2.cvtColor(a, cv2.COLOR_HSV2BGR)
    # cv2.imshow('1', a_img)
    # cv2.waitKey(0)


    save_path = save_dir + '/' + os.path.splitext(os.path.basename(ll))[0] + '.png'
    cv2.imwrite(save_path, a_img)
