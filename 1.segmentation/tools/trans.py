import os
import glob
import shutil

a = '/mnt/ai2019/ljl/data/gastric/total/2048/dyl/mask'
c = glob.glob(os.path.join(a, '*'))


train = '/mnt/ai2019/ljl/data/gastric/total/2048/dyl/train_mask'
val = '/mnt/ai2019/ljl/data/gastric/total/2048/dyl/val_mask'
os.makedirs(train, exist_ok=True)
os.makedirs(val, exist_ok=True)

f1 = open("/mnt/ai2019/ljl/data/gastric/total/2048/dyl/train.txt")
date = f1.read().splitlines()
for tr in date:
    print(len(date))
    for n in c:
        if tr in n:
            print('train:', tr)
            shutil.copy(n, os.path.join(train, tr+'.png'))
f1.close()

f1 = open("/mnt/ai2019/ljl/data/gastric/total/2048/dyl/val.txt")
date = f1.read().splitlines()
for tr in date:
    print(len(date))
    for n in c:
        if tr in n:
            print('val:', tr)
            shutil.copy(n, os.path.join(val, tr+'.png'))
f1.close()