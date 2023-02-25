import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
import glob
import os
import sys
import matplotlib

# image_RGB_hist_show
def image_RGB_hist_show(ll):
    im = plt.imread(ll)
    plt.figure(os.path.basename(ll))
    plt.subplot(121)
    plt.title(os.path.basename(ll))
    plt.imshow(im)
    plt.subplot(122)
    r = im[:,:,0].flatten()
    plt.hist(r, bins=20, edgecolor='r', facecolor='r')
    g = im[:, :, 1].flatten()
    plt.hist(g, bins=20, edgecolor='g', facecolor='g')
    b = im[:, :, 2].flatten()
    plt.hist(b, bins=20, edgecolor='b', facecolor='b')
    plt.show()


# HSV_transfer
def HSV_transfer(ll, save_dir):
    image = cv2.imread(ll)
    out_img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    a = out_img_HSV.copy()
    # turn_green_hsv[:, :, 0] = (turn_green_hsv[:, :, 0] - 30) % 180
    a[:, :, 0] = 140
    a_img = cv2.cvtColor(a, cv2.COLOR_HSV2BGR)
    save_path = save_dir + '/' + os.path.splitext(os.path.basename(ll))[0] + '.png'
    cv2.imwrite(save_path, a_img)


# HSV_hist_image + Hue_image + Saturation_image + origin_image
def HSV_hist_image(ll, save_dir):
    image = cv2.imread(ll)
    out_img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsvChannels = cv2.split(out_img_HSV)# 将HSV格式的图片分解为3个通道
    hue_np = np.array(hsvChannels[0])
    hue_onehot = hue_np.ravel()
    hue_list = hue_onehot.tolist()
    hue_max = max(set(hue_list), key=hue_list.count)
    s_np = np.array(hsvChannels[1])

    ax1 = plt.subplot(221)
    plt.title(os.path.basename(ll))
    ax1.imshow(image)
    ax2 = plt.subplot(222)
    plt.title('Hue_hist')
    n, bins, patchs = ax2.hist(hue_np.ravel(),  256, [0, 256], density=True)
    height = max(n)
    # n, bins, patchs = plt.hist(hue_np.ravel(), 256, [0, 256], density=True)
    ax2.text(hue_max, height-0.025, hue_max, fontsize=10, color="r", alpha=1)
    ax3 = plt.subplot(223)
    plt.title('Hue_img')
    plt.imshow(hue_np)
    ax4 = plt.subplot(224)
    plt.title('s_img')
    plt.imshow(s_np)
    # plt.show()

    save_path = save_dir + os.path.splitext(os.path.basename(ll))[
        0] + '.png'
    plt.savefig(save_path)
    plt.close()


def RGB_hist_image(image, i):
    color = ('blue', 'green', 'red')
    for ii, color in enumerate(color):
        hist = cv2.calcHist([image], [ii], None, [256], [0,256],accumulate=True)

        plt.plot(hist, color=color)
        plt.xlim([0,256])
        plt.title(os.path.basename(i))
        save_path = save_dir + os.path.splitext(os.path.basename(i))[0]+'_hist.png'
        plt.savefig(save_path)
    # plt.show()


def calculate_HUE_value(tra_img_name_list, hue_mean):
    for i, ll in enumerate(tra_img_name_list):
        sys.stdout.write('\r%d/%s' % (i, len(tra_img_name_list)))

        image = cv2.imread(ll)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_image = hsv_image[:, :, 0]
        # saturation_image = hsv_image[:, :, 1]
        # value_image = hsv_image[:, :, 2]

        hue_onehot = hue_image.ravel()
        hue_list = hue_onehot.tolist()
        hue_max = max(set(hue_list), key=hue_list.count)

        hue_mean.append(hue_max)
    print()
    print(sum(hue_mean) / len(tra_img_name_list))


def plt_hist_img(tra_img_name_list):
    for i, ll in enumerate(tra_img_name_list):
        sys.stdout.write('\r%d/%s' % (i, len(tra_img_name_list)))
        matplotlib.rcParams.update({'font.size': 8})
        plt.figure(figsize=(8, 4), dpi=150)
        image = cv2.imread(ll)

        ax1 = plt.subplot(241)
        plt.title(os.path.basename(ll), fontsize=8)
        ax1.imshow(image)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_image = hsv_image[:, :, 0]
        saturation_image = hsv_image[:, :, 1]
        value_image = hsv_image[:, :, 2]

        hue_onehot = hue_image.ravel()
        hue_list = hue_onehot.tolist()
        hue_max = max(set(hue_list), key=hue_list.count)

        sat_onehot = saturation_image.ravel()
        sat_list = sat_onehot.tolist()
        sat_max = max(set(sat_list), key=sat_list.count)

        val_onehot = value_image.ravel()
        val_list = val_onehot.tolist()
        val_max = max(set(val_list), key=val_list.count)

        ax2 = plt.subplot(242)
        plt.title('Hue_image', fontsize=8)
        plt.imshow(hue_image)

        ax3 = plt.subplot(243)
        plt.title('sat_image', fontsize=8)
        plt.imshow(saturation_image)

        ax4 = plt.subplot(244)
        plt.title('val_image', fontsize=8)
        plt.imshow(value_image)

        ax5 = plt.subplot(245)
        plt.title('RGB_hist', y=-0.01, fontsize=8)
        color = ('blue', 'green', 'red')
        for ii, color in enumerate(color):
            hist = cv2.calcHist([image], [ii], None, [256], [0, 256], accumulate=True)

            plt.plot(hist, color=color)
            plt.xlim([0, 256])

        ax6 = plt.subplot(246)
        plt.title('Hue_hist', y=-0.01, fontsize=8)
        n, bins, patchs = ax6.hist(hue_onehot, 256, [0, 256], density=True)
        height = max(n)
        ax6.text(hue_max, height - 0.025, hue_max, fontsize=10, color="r", alpha=1)

        ax7 = plt.subplot(247)
        plt.title('sat_hist', y=-0.01, fontsize=8)
        n, bins, patchs = ax7.hist(sat_onehot.ravel(), 256, [0, 256], density=True)
        height = max(n)
        ax7.text(sat_max, height, sat_max, fontsize=10, color="r", alpha=1)

        ax8 = plt.subplot(248)
        plt.title('val_hist', y=-0.01, fontsize=8)
        n, bins, patchs = ax8.hist(val_onehot.ravel(), 256, [0, 256], density=True)
        height = max(n)
        ax8.text(val_max, height, val_max, fontsize=10, color="r", alpha=1)

        # plt.show()
        # print()
        save_path = save_dir + os.path.splitext(os.path.basename(ll))[
            0] + '.png'
        plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    hue_mean=[]
    train_dir = '/mnt/ai2019/deng/Data/gastric_s2_s4/train/clean_data_new/train_img_512/'
    save_dir = '/mnt/ai2019/deng/Data/gastric_s2_s4/val/hist/'
    tra_img_name_list = sorted(glob.glob(train_dir + '*' + '.png'))

    calculate_HUE_value(tra_img_name_list, hue_mean)
    # plt_hist_img(tra_img_name_list)
