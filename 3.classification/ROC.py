import os
import sys
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model_vgg import vgg
from model_me import Efficient_b3
import torchvision

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from dataset import MyDataset_test

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import time


def roc_method(net, data_loader):
    net.eval()
    pre_ori_list = []
    gt_list = []
    pre_list = []
    pre_score = []
    image_name_list = []
    with torch.no_grad():
        test_bar = tqdm(data_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, test_labels, pre_ori, image_name = test_data['image'], test_data['label'], test_data[
                'pre_ori'], \
                                                            test_data['image_name']
            outputs = net(test_images.to(device))

            # predict = outputs.argmax(1)
            # predict_value = predict.cpu().numpy()[0]
            predict = torch.softmax(outputs, 1).cpu().numpy().squeeze()[1]
            if predict > 0.5:
                predict_value = 1
            else:
                predict_value = 0

            test_labels_value = test_labels.cpu().numpy()[0]
            pre_ori_value = pre_ori.cpu().numpy()[0]

            pre_ori_list.append(pre_ori_value)
            gt_list.append(test_labels_value)
            pre_list.append(predict_value)
            pre_score.append(predict)
            image_name_list.append(image_name)

    fpr, tpr, thresholds = roc_curve(np.array(gt_list), np.array(pre_score))
    fpr_ori, tpr_ori, thresholds_ori = roc_curve(np.array(gt_list), np.array(pre_ori_list))
    AUC = auc(fpr, tpr)
    AUC_ori = auc(fpr_ori, tpr_ori)

    return fpr, tpr, fpr_ori, tpr_ori, AUC, AUC_ori


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_csv = '/mnt/ai2019/ljl/code/software_platform/patch_test1_classification_big_model20220225_depv3+_ef3.csv'
top_csv = '/mnt/ai2019/ljl/code/software_platform/infer/val_small_top1000.csv'


# model_weight_path1 = "./K_fold_model_small/shufflenet_v2_x0_5_best.pth"
# model_weight_path2 = "./K_fold_model_small/Efficient_b3_best.pth"
# model_weight_path3 = "./K_fold_model_small/mobilenet_v3_large_best.pth"
# model_weight_path4 = "./K_fold_model_small/resnet18_best.pth"
# model_weight_path5 = "./K_fold_model_small/resnet34_best.pth"
# model_weight_path6 = "./K_fold_model_small/resnet50_best.pth"

model_weight_path1 = "./K_fold_model_big_new/shufflenet_v2_x0_5.pth"
model_weight_path2 = "./K_fold_model_big_new/Efficient_b3.pth"
model_weight_path3 = "./K_fold_model_big_new/mobilenet_v3_large.pth"
model_weight_path4 = "./K_fold_model_big_new/resnet18.pth"
model_weight_path5 = "./K_fold_model_big_new/resnet34.pth"
model_weight_path6 = "./K_fold_model_big_new/resnet50.pth"


net1 = torchvision.models.shufflenet_v2_x0_5(pretrained=False)
net1.fc = nn.Linear(in_features=1024, out_features=2, bias=True)
net2 = Efficient_b3(3, 2)
net3 = torchvision.models.mobilenet_v3_large(pretrained=False)
net3.classifier[3] = nn.Linear(in_features=1280, out_features=2, bias=True)
net4 = torchvision.models.resnet18(pretrained=False)
net4.fc = nn.Linear(in_features=512, out_features=2, bias=True)
net5 = torchvision.models.resnet34(pretrained=False)
net5.fc = nn.Linear(in_features=512, out_features=2, bias=True)
net6 = torchvision.models.resnet50(pretrained=False)
net6.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

net1.to(device)
net1.load_state_dict(torch.load(model_weight_path1, map_location=device))
net2.to(device)
net2.load_state_dict(torch.load(model_weight_path2, map_location=device))
net3.to(device)
net3.load_state_dict(torch.load(model_weight_path3, map_location=device))
net4.to(device)
net4.load_state_dict(torch.load(model_weight_path4, map_location=device))
net5.to(device)
net5.load_state_dict(torch.load(model_weight_path5, map_location=device))
net6.to(device)
net6.load_state_dict(torch.load(model_weight_path6, map_location=device))

test_transform = A.Compose([
    A.Resize(30, 30),
    ToTensorV2()
])

test_data = pd.read_csv(test_csv)
test_name_list = [img_name for img_name in test_data['image_name'] if img_name != 'image_name']
test_imgs = [img for img in test_data['pre_array'] if img != 'pre_array']
test_masks = [mask for mask in test_data['gt'] if mask != 'gt']
test_ori = [mask for mask in test_data['ori_pre_wsi'] if mask != 'ori_pre_wsi']

# test_data = pd.read_csv(test_csv)
# test_name_list = [img_name for img_name in test_data['image_name'] if img_name != 'image_name']
# test_imgs = [img for img in test_data['pre_array'] if img != 'pre_array']
# test_masks = [int(mask) for mask in test_data['gt'] if mask != 'gt']
# test_ori = [int(mask) for mask in test_data['ori_pre_wsi'] if mask != 'ori_pre_wsi']

top_csv_data = pd.read_csv(top_csv)
top_gt_list = [mask for mask in top_csv_data['gt']]
top_pre_list = [mask for mask in top_csv_data['ori_pre_wsi']]
fpr_top, tpr_top, thresholds_top = roc_curve(np.array(top_gt_list), np.array(top_pre_list))

# fpr_top, tpr_top, thresholds_top = roc_curve(np.array(test_masks), np.array(test_ori))

test_dataset = MyDataset_test(test_imgs, test_masks, test_ori, test_name_list, test_transform)
test_num = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

fpr1, tpr1, fpr_ori1, tpr_ori1, AUC1, AUC_ori1 = roc_method(net1, test_loader)
fpr2, tpr2, fpr_ori2, tpr_ori2, AUC2, AUC_ori2 = roc_method(net2, test_loader)
fpr3, tpr3, fpr_ori3, tpr_ori3, AUC3, AUC_ori3 = roc_method(net3, test_loader)
fpr4, tpr4, fpr_ori4, tpr_ori4, AUC4, AUC_ori4 = roc_method(net4, test_loader)
fpr5, tpr5, fpr_ori5, tpr_ori5, AUC5, AUC_ori5 = roc_method(net5, test_loader)
fpr6, tpr6, fpr_ori6, tpr_ori6, AUC6, AUC_ori6 = roc_method(net6, test_loader)

# 指标
matrix = confusion_matrix(top_gt_list, top_pre_list)
TP = matrix[1][1]
FP = matrix[0][1]
FN = matrix[1][0]
TN = matrix[0][0]
accuracy = (TP + TN) / (TP + FP + FN + TN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
AUC = auc(fpr_top, tpr_top)
print('AUC', AUC)
print('Accuracy', accuracy)
print('Sensitivity', sensitivity)
print('Specificity', specificity)



# ROC曲线
# plt.figure()
figure, ax = plt.subplots()
plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_ori1, tpr_ori1, color='navy', linewidth=1, markersize=8, label=u'Original(AUC= %0.4f)' % AUC_ori1)
plt.plot(fpr_top, tpr_top, color='navy', linewidth=1, markersize=8, label=u'Top_1000_probabilities(AUC= %0.4f)' % AUC)
plt.plot(fpr1, tpr1, color='hotpink', linewidth=1, markersize=8, label=u'ShuffleNet_v2(AUC= %0.4f)' % AUC1)
plt.plot(fpr2, tpr2, color='paleturquoise', linewidth=1, markersize=8, label=u'Efficienet_b3(AUC= %0.4f)' % AUC2)
plt.plot(fpr3, tpr3, color='gold', linewidth=1, markersize=8, label=u'MobileNetV3(AUC= %0.4f)' % AUC3)
plt.plot(fpr4, tpr4, color='skyblue', linewidth=1, markersize=8, label=u'ResNet18(AUC= %0.4f)' % AUC4)
plt.plot(fpr5, tpr5, color='palegreen', linewidth=1, markersize=8, label=u'ResNet34(AUC= %0.4f)' % AUC5)
plt.plot(fpr6, tpr6, color='darkorange', linewidth=1, markersize=8, label=u'ResNet50(AUC= %0.4f)' % AUC6)


plt.legend(loc='lower right', prop={'family':'Times New Roman','size':10})
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.ylabel('True Positive Rate', fontfamily="Times New Roman")
plt.xlabel('False Positive Rate', fontfamily="Times New Roman")
plt.title('Performance of slide-level classification in biopsy specimen', fontfamily="Times New Roman")
plt.grid(linestyle='-.')
plt.grid(True)
# plt.savefig('./figure/ROC_surgical_specimen_new.png',dpi=1200, bbox_inches='tight')
plt.show()


# color = [
# 'lightcoral',
# 'coral',
# 'darkorange',
# 'gold',
# 'palegreen',
# 'paleturquoise',
# 'skyblue',
# 'plum',
# 'hotpink',
# 'pink']
