from models.loss.Loss_all import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from utils import *
import glob
from tqdm import tqdm
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset, MyDataset, MyDataset_Server
from models.Models import *
import os
from tensorboardX import SummaryWriter
from models.metrics.Miou import *
import argparse
import cv2
import torch
from torchstat import stat
from models.u2net import U2NET
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser(description='Pytorch Segmentation Framework')
parser.add_argument('--model_name', '-name', type=str,
                    default='DeepLabV3Plus_b3_epoch300_bs16_lr0.0005_optAdamW_schedulerCosineAnnealingLR_221103_augument_mixloss',
                    help='model_name')
parser.add_argument('--batch_size_train', '-batch', type=int, default=16, help='train batch size')
parser.add_argument('--cuda_device', '-cuda', type=str, default='0', help='cuda_device_number')
# parser.add_argument('--pretrained_model', '-pretrain', type=str, default='', help='pretrained_model')
# parser.add_argument('--state_dir', '-state', type=str, default='', help='state_dir')
parser.add_argument('--save_frq', '-save', type=int, default=5000, help='save_state_frequency')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = range(torch.cuda.device_count())  # torch.cuda.device_count()=2

# ------- 1. set the directory of training dataset --------
model_name = args.model_name  # 'u2netp'
pretrained_dir = ''  # pre_trained_model保存路径
state_dir = ''  # state_model保存路径s

epoch_num = 300
save_frq = args.save_frq  # save the model every 2000 iterations
batch_size_train = args.batch_size_train
batch_size_val = 1
ite_num = 0
current_step = 0
lr = 0.0005
best_miou = 0
resize_size = 512


train_dir = '/mnt/ai2019/ljl/data/gastric/total/2048/ljl/2048/clean/resize_512/train_image/'
train_label_dir = '/mnt/ai2019/ljl/data/gastric/total/2048/ljl/2048/clean/resize_512/train_mask/'
val_dir = '/mnt/ai2019/ljl/data/gastric/total/2048/ljl/2048/clean/resize_512/val_image/'
val_label_dir = '/mnt/ai2019/ljl/data/gastric/total/2048/ljl/2048/clean/resize_512/val_mask/'
# aug_dir = '/mnt/ai2019/ljl/data/gastric/total/2048/ljl/2048/clean/supplement/aug_img/'
# aug_label_dir = '/mnt/ai2019/ljl/data/gastric/total/2048/ljl/2048/clean/supplement/aug_mask/'

model_dir = os.path.join('/mnt/ai2019/ljl/code/software_platform/train/torch_framework/paper/patch',
                         model_name + os.sep)
mkdir_exists_chmod(model_dir)
writer = SummaryWriter(log_dir=model_dir + '/tb/')
tra_img_name_list = sorted(glob.glob(train_dir + '*'))
tra_lbl_name_list = [img.replace('/train_image', '/train_mask').replace('.png', '_mask.png') for img in
                     tra_img_name_list]
val_img_name_list = sorted(glob.glob(val_dir + '*'))
val_lbl_name_list = [img.replace('/val_image', '/val_mask').replace('.png', '_mask.png') for img in val_img_name_list]
# aug_img_name_list = sorted(glob.glob(aug_dir + '*'))
# aug_lbl_name_list = [img.replace('/aug_img', '/aug_mask').replace('.png', '_mask.png') for img in aug_img_name_list]

# imgs_train = [cv2.resize(cv2.imread(i), (resize_size, resize_size))[:, :, ::-1] for i in tra_img_name_list]
# masks_train = [cv2.resize(cv2.imread(i), (resize_size, resize_size))[:, :, 0] for i in tra_lbl_name_list]
# imgs_val = [cv2.resize(cv2.imread(i), (resize_size, resize_size))[:, :, ::-1] for i in val_img_name_list]
# masks_val = [cv2.resize(cv2.imread(i), (resize_size, resize_size))[:, :, 0] for i in val_lbl_name_list]
# imgs_aug = [cv2.resize(cv2.imread(i), (resize_size, resize_size))[:, :, ::-1] for i in aug_img_name_list]
# masks_aug = [cv2.resize(cv2.imread(i), (resize_size, resize_size))[:, :, 0] for i in aug_lbl_name_list]

# imgs_train = imgs_train + imgs_aug
# masks_train = masks_train + masks_aug

print("---")
print("train mode")
print("train_dir:{}, num:{}".format(train_dir, len(tra_img_name_list)))
print("train_label_dir:{}, num:{}".format(train_label_dir, len(tra_lbl_name_list)))
print("val_dir:{}, num:{}".format(val_dir, len(val_img_name_list)))
print("val_label_dir:{}, num:{}".format(val_label_dir, len(val_lbl_name_list)))
print("model_name:", model_name)
print("epoch_num:", epoch_num)
print("---")
print("save log_file to " + model_dir)
file = open(os.path.join(model_dir, 'log.txt'), 'w')
with open(os.path.abspath(__file__), 'r') as f:
    data = f.read()
    file.write(data)
file.close()
print("---")

# ------- 3. define model --------

# net = U_Net(3, 1).to(device)
net = smp.DeepLabV3Plus("efficientnet-b3", encoder_weights='imagenet', classes=2).to(device)
# net = smp.DeepLabV3Plus("efficientnet-b0", encoder_weights='imagenet', classes=2).to(device)
# net = smp.UnetPlusPlus("efficientnet-b3", encoder_weights='imagenet', classes=2).to(device)


print("param size = {} MB".format(count_parameters_in_MB(net)))
if len(device_ids) > 1:
    net = torch.nn.DataParallel(net)
# stat(net, (3, 512, 512))
# ------- 4. define optimizer --------
print("---define optimizer...")
# optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer = optim.AdamW(net.parameters(), lr=lr)
CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-8)
mix_loss = MixedLoss(alpha=10, gamma=2, classes=2)
# ------- 5. training process --------
print("---start training...")

# 数据增强
train_transform = A.Compose([
    A.Resize(512, 512),
    A.RandomRotate90(),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.1), rotate_limit=40, p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.5,
        contrast_limit=0.1,
        p=0.5
    ),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=100, val_shift_limit=80),
    # A.OneOf([
    #     A.CoarseDropout(max_holes=100,max_height=aug_size,max_width=aug_size,fill_value=[239, 234, 238]),
    #     A.GaussNoise()
    # ]),
    A.GaussNoise(),
    A.OneOf([
        A.ElasticTransform(),
        A.GridDistortion(),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0)
    ]),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()], p=1.)

# val_transform = A.Compose([A.Resize(512, 512), ToTensorV2()])
val_transform = A.Compose([
    A.Resize(512, 512),
    A.RandomRotate90(),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.1), rotate_limit=40, p=0.5),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()], p=1.)

# train_dataset = MyDataset_Server(img_list=imgs_train, mask_list=masks_train, transform=train_transform)
train_dataset = MyDataset(img_name_list=tra_img_name_list, lbl_name_list=tra_lbl_name_list, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=8, pin_memory=True,
                              drop_last=True)

# val_dataset = MyDataset_Server(img_list=imgs_val, mask_list=masks_val, transform=val_transform)
val_dataset = MyDataset(img_name_list=val_img_name_list, lbl_name_list=val_lbl_name_list, transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=8, pin_memory=False)

# 如果有保存的模型，则加载模型，并在其基础上继续训练
if os.path.exists(pretrained_dir):
    if os.path.exists(state_dir):
        checkpoint = torch.load(state_dir)
        # print(checkpoint.keys())
        # print(torch.load(log_dir))  # U2NET
        net.load_state_dict(torch.load(pretrained_dir))
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        ite_num = checkpoint['iter']
        best_miou = checkpoint['miou']
        print('---Loading epoch {}/ iter {}/ last best miou {} success！'.format(start_epoch, ite_num, best_miou))
else:
    start_epoch = 0
    print('---No exist module, start new training!')
    print('---')


def train_val():
    global ite_num, best_miou
    print('Train: epoch_num:{}, train_size:{}'.format(epoch_num, len(train_dataloader)))
    with tqdm(total=len(train_dataloader) * epoch_num) as t:
        scaler = GradScaler()
        for epoch in range(start_epoch, epoch_num):

            net.train()
            train_loss = 0
            train_miou = 0
            for i, data in enumerate(train_dataloader):
                t.set_description("iter:{} ".format(ite_num))
                inputs, labels = data['image'], data['label']  # b, c, h, w

                inputs_v = inputs.type(torch.FloatTensor).to(device)
                labels_v = labels.to(device)

                with autocast():
                    d0 = net(inputs_v)
                    loss = mix_loss(d0, labels_v)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                predicted = d0.argmax(1)

                train_miou += calculate_miou(predicted, labels_v, 2).item()

                # writer.add_scalar('step_Train_loss', loss, ite_num)
                writer.add_scalar('Train_loss', train_loss / (i + 1), global_step=ite_num)
                writer.add_scalar('Train_Miou', train_miou / (i + 1), global_step=ite_num)

                ite_num = ite_num + 1
                t.set_postfix(loss='{:.3f}'.format(train_loss / (i + 1)))
                t.update(1)
                # del temporary outputs and loss
                del loss, d0

                if ite_num % save_frq == 0:
                    net.eval()
                    val_loss = 0
                    val_miou = 0
                    val_loss_BCEloss = 0
                    with torch.no_grad():
                        with tqdm(total=len(val_dataloader), ncols=80, ascii=True) as p:
                            for b, v_data in enumerate((val_dataloader)):
                                imgname = val_img_name_list[b]
                                p.set_description('Val')
                                v_inputs, v_labels = v_data['image'], v_data['label']
                                v_inputs = v_inputs.type(torch.FloatTensor).to(device)
                                v_labels = v_labels.to(device)

                                d1 = net(v_inputs)
                                predicted_val = d1.argmax(1)

                                v_loss = mix_loss(d1, v_labels)
                                val_loss += v_loss.item()

                                val_miou += calculate_miou(predicted_val, v_labels, 2).item()

                                p.set_postfix(val_miou='{:.3f}'.format(val_miou / (b + 1)))
                                # # test_miou='{:best_miou .3f}'.format(test_miou / (b + 1)))
                                p.update(1)

                                if (val_miou / (b + 1)) > best_miou:
                                    pred_img = predicted_val
                                    pred_img_np = pred_img.detach().cpu().numpy()
                                    pred_img_np = pred_img_np * 255
                                    pred_img_np = pred_img_np.transpose((1, 2, 0))
                                    cv2.imwrite(model_dir + '/val_img/' + val_img_name_list[b].split(os.sep)[-1],
                                                pred_img_np)

                    writer.add_scalar('val_loss', val_loss / (b + 1), global_step=ite_num)
                    writer.add_scalar('val_Miou', val_miou / (b + 1), global_step=ite_num)
                    if (val_miou / (b + 1)) > best_miou:
                        state = {'optimizer': optimizer.state_dict(), 'epoch': epoch, 'iter': ite_num,
                                 'miou': val_miou / (b + 1)}
                        os.makedirs(model_dir + '/state/', exist_ok=True)
                        os.makedirs(model_dir + '/model/', exist_ok=True)
                        torch.save(state,
                                   model_dir + '/state/' + "%d.pth" % (ite_num))
                        torch.save(net.state_dict(),
                                   model_dir + '/model/' + model_name + "_itr_%d.pth" % (
                                       ite_num))
                        best_miou = (val_miou / (b + 1))
                        p.write("---save the state...best_miou:{:.3f}".format(best_miou))
                    file = open(os.path.join(model_dir, 'state.txt'), 'a')
                    file.write("epoch:%3d, epoch_num:%3d, ite: %d, " % (epoch, epoch_num, ite_num) + '\n')
                    file.write("train loss: %3f, train miou: %3f" % (train_loss / (i + 1), train_miou / (i + 1)) + '\n')
                    file.write("val loss: %3f, val miou: %3f, best_miou: %3f" % (
                        val_loss / (b + 1), val_miou / (b + 1), best_miou) + '\n' + '\n')
                    file.close()
                    net.train()
                    p.close()
                    del d1, v_loss

        CosineLR.step()
    # writer.add_graph(net, (inputs,))
    writer.close()


if __name__ == '__main__':
    train_val()
