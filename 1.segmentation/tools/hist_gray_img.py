import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import os

from utils import makedir



# train_dir = '/mnt/ai2019/deng/Data/gastric_s2_s4/train/image_matching/image_16/'
# tra_img_name_list = sorted(glob.glob(train_dir + '*' + '.png'))
# n=0
# for i in tra_img_name_list:
#     img1 = cv2.imread(i, 0)
#     img_name = os.path.splitext(os.path.basename(i))[0]
#     # 别忘了中括号 [img],[0],None,[256],[0,256]，只有 mask 没有中括号
#     hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
#
#     # img2 = cv2.imread('test6.jpg')
#     # color = ('b', 'g', 'r')
#     # for i, col in enumerate(color):
#     #     histr = cv2.calcHist([img2], [i], None, [256], [0, 256])
#     #     plt.subplot(224), plt.plot(histr, color=col),
#     #     plt.xlim([0, 256]), plt.title('Histogram')
#
#     # plt.subplot(211)
#     # plt.imshow(img1, 'gray')
#     # plt.title(img_name)
#     #
#     # plt.subplot(212)
#     plt.hist(img1.ravel(), 256, [0, 256])
#     save_path = '/mnt/ai2019/deng/project/experiments/hist_gray/' + os.path.splitext(os.path.basename(i))[0] + '_hist.png'
#     makedir(save_path.rsplit('/', 1)[0])
#     plt.savefig(save_path)
#     sys.stdout.write('\r%d/%d' % (n, len(tra_img_name_list)))
#     # sys.stdout.flush()
#     n+=1
#     # plt.title('Histogram'), plt.xlim([0, 256])
#     # plt.subplot(223), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title('Image2')
#     # plt.show()


n=0
train_dir = '/mnt/ai2019/deng/Data/gastric_s2_s4/train/image_matching/image_16/'
tra_img_name_list = sorted(glob.glob(train_dir + '*' + '.png'))
for i in tra_img_name_list:
    img1 = cv2.imread(i, 0)
    # 别忘了中括号 [img],[0],None,[256],[0,256]，只有 mask 没有中括号
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])

    img2 = cv2.imread(i)

    plt.subplot(221), plt.imshow(img1, 'gray'), plt.title('Image1')
    plt.subplot(222), plt.hist(img1.ravel(), 256, [0, 256]),
    plt.title('Histogram'), plt.xlim([0, 256])
    plt.subplot(223), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title('Image2')
    color = ('b', 'g', 'r')
    plt.subplot(224)
    for ii, col in enumerate(color):
        histr = cv2.calcHist([img2], [ii], None, [256], [0, 256], accumulate=True)
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
        plt.title('Histogram')
    # plt.show()
    save_path = '/mnt/ai2019/deng/project/experiments/hist_new/' + os.path.splitext(os.path.basename(i))[0] + '_hist.png'
    makedir(save_path.rsplit('/', 1)[0])
    plt.savefig(save_path)
    sys.stdout.write('\r%d/%d' % (n, len(tra_img_name_list)))
    n+=1
    plt.close()