
import glob
import os
import numpy
import numpy as np
import math
import cv2
import torch
from tools.pytorch_ssim import ssim as p_ssim
from torch.autograd import Variable
import sys



# def ssim(img1, img2):
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
#
# def calculate_ssim(img1, img2):
#     '''calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 s = ssim(img1, img2)
#                 ssims.append(s)
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')



# 直接用uint8的图算出来的PSNR更高
def PSNR(img1, img2, shave_border=0):
    height, width = img1.shape[:2]
    img1 = img1[shave_border:height - shave_border, shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border, shave_border:width - shave_border]
    rmse = math.sqrt(np.mean((img1 - img2) ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

# 用float64的图算出来的PSNR更low
def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim_s(img1, img2):
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0) / 255.0
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0) / 255.0
    img1 = Variable(img1, requires_grad=False)  # torch.Size([256, 256, 3])
    img2 = Variable(img2, requires_grad=False)
    ssim_value = p_ssim(img1, img2).item()
    return ssim_value


# singal image caculate PSNR SSIM
# original = cv2.imread("1.png")  # numpy.adarray
# contrast = cv2.imread("2.png", 1)
# psnrValue = psnr(original, contrast)
# ssimValue = ssim(original, contrast)
# print(psnrValue)
# print(ssimValue)

HR_dir = '/mnt/ai2019/deng/project/experiments/cyclegan/ljl_gastric/output/test_img/B/'
LR_dir = '/mnt/ai2019/ljl/data/MITOS-ATYPIA-14/train/enddata/testB/'
img_name_list = sorted(glob.glob(HR_dir + '*' + '.png'))
# tra_lbl_name_list = sorted(glob.glob(LR_dir + '*' + '.jpg'))
ssim_sum=[]
p_sum=[]
ppp_sum=[]

for i, batch in enumerate(img_name_list):
    sys.stdout.write('\r%d/%s' % (i, len(img_name_list)))
    img1 = cv2.imread(batch, cv2.IMREAD_UNCHANGED)

    base_name = os.path.basename(batch)
    LR_jpeg = os.path.splitext(base_name)[0] + '.jpg'
    img2 = cv2.imread((LR_dir + LR_jpeg), cv2.IMREAD_UNCHANGED)
    # img1 = np.double(img1)
    # img2 = np.double(img2)

    s = ssim_s(img1, img2)
    p = PSNR(img1, img2)
    ssim_sum.append(s)
    p_sum.append(p)

print('\nssim:{}'.format(sum(ssim_sum)/len(img_name_list)))
print('psnr:{}'.format(sum(p_sum)/len(img_name_list)))


#28.218755456890623