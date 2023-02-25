import cv2
import glob
import os
import sys
import torch
from shutil import move, copy

m = []
nn = []
zero = []
one = []
img_dir = '/mnt/ai2019/deng/Data/gastric_s2_s4/train/clean_data_new/train_label_512/'
img_list = sorted(glob.glob(os.path.join(img_dir + '*' + '.png')))
for i, batch in enumerate(img_list):
    sys.stdout.write('\r%d/%s/%s' % (i, len(img_list), len(m)))
    img = cv2.imread(batch, 0)

    img_t = torch.tensor(img).cuda()
    # if (img==0).all():
    #
    #     zero.append(batch)
    # else:
    #     one.append(batch)

    n=0
    for x in range(img_t.shape[0]):
        for y in range(img_t.shape[1]):
            if img[x][y]==0:
                n+=1
    if n >=235930:
        # m.append(n)
        # nn.append(os.path.basename(batch))
        move(batch, '/mnt/ai2019/deng/Data/gastric_s2_s4/train/clean_data_new/1/' + os.path.basename(batch))
print(len(m))
print(nn)
