from random import shuffle
import glob
import os

img_path = '/mnt/ai2019/ljl/data/gastric/total/2048/orton/train_add_normal/train_image'
img = glob.glob(os.path.join(img_path, '*'))
shuffle(img)
with open('/mnt/ai2019/ljl/code/software_platform/train/torch_framework/data_id/500_add_normal.txt', 'w') as f:
    for i in range(500):
        img_name = os.path.basename(img[i])
        # img_name = img[i]
        f.writelines(img_name + ' ' + '\n')
