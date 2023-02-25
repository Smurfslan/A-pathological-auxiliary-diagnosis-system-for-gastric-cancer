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
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
import time



def roc_curve_me(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.show()


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_csv = '/mnt/ai2019/ljl/code/software_platform/infer/patch_test1_classification_big_model20220225_depv3+_ef3.csv'
top_csv = '/mnt/ai2019/ljl/code/software_platform/infer/val_small_top1000.csv'
model_weight_path = "./K_fold_model_big_new/Efficient_b3.pth"

# ---------------------model----------------------------------
# net = torchvision.models.resnet18(pretrained=False)
# net.fc = nn.Linear(in_features=512, out_features=2, bias=True)

# net = torchvision.models.resnet34(pretrained=False)
# net.fc = nn.Linear(in_features=512, out_features=2, bias=True)

# net = torchvision.models.resnet50(pretrained=False)
# net.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

# net = torchvision.models.mobilenet_v3_large(pretrained=False)
# net.classifier[3] = nn.Linear(in_features=1280, out_features=2, bias=True)

net = Efficient_b3(3, 2)

# net = torchvision.models.shufflenet_v2_x0_5(pretrained=False)
# net.fc = nn.Linear(in_features=1024, out_features=2, bias=True)


net.to(device)
net.load_state_dict(torch.load(model_weight_path, map_location=device))


test_transform = A.Compose([
    A.Resize(100, 100),
    ToTensorV2()
])

test_data = pd.read_csv(test_csv)
test_name_list = [img_name for img_name in test_data['image_name']]
test_imgs = [img for img in test_data['pre_array']]
test_masks = [mask for mask in test_data['gt']]
test_ori = [mask for mask in test_data['ori_pre_wsi']]

top_csv_data = pd.read_csv(top_csv)
top_gt_list = [mask for mask in top_csv_data['gt']]
top_pre_list = [mask for mask in top_csv_data['ori_pre_wsi']]

test_dataset = MyDataset_test(test_imgs, test_masks, test_ori, test_name_list, test_transform)
test_num = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

print("param size = {} MB".format(count_parameters_in_MB(net)))

net.eval()
pre_ori_list = []
gt_list = []
pre_list = []
pre_score = []
image_name_list = []

time_begin = time.time()
with torch.no_grad():
    test_bar = tqdm(test_loader, file=sys.stdout)
    for test_data in test_bar:
        # with tqdm(total=len(test_loader), ncols=80, ascii=True) as t:
        # for b, test_data in enumerate((test_loader)):
        test_images, test_labels, pre_ori, image_name = test_data['image'], test_data['label'], test_data['pre_ori'], \
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
        # t.update()

print('test_time', (time.time()-time_begin)/len(image_name_list))

# 错误数据分析
for j in range(len(pre_ori_list)):
    if pre_list[j] == 1 and gt_list[j] == 0:
        print('误诊： ', image_name_list[j])
    elif pre_list[j] == 0 and gt_list[j] == 1:
        print('漏诊： ', image_name_list[j])
    if pre_ori_list[j] == 0 and gt_list[j] == 1:
        print('原始漏诊： ', image_name_list[j])

matrix = confusion_matrix(gt_list, pre_list)
matrix1 = confusion_matrix(gt_list, pre_ori_list)
matrix2 = confusion_matrix(top_gt_list, top_pre_list)

TP = matrix[1][1]
FP = matrix[0][1]
FN = matrix[1][0]
TN = matrix[0][0]

TP1 = matrix2[1][1]
FP1 = matrix2[0][1]
FN1 = matrix2[1][0]
TN1 = matrix2[0][0]

accuracy = (TP + TN) / (TP + FP + FN + TN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

accuracy1 = (TP1 + TN1) / (TP1 + FP1 + FN1 + TN1)
sensitivity1 = TP1 / (TP1 + FN1)
specificity1 = TN1 / (TN1 + FP1)

fpr, tpr, thresholds = roc_curve(np.array(gt_list), np.array(pre_score))
fpr_ori, tpr_ori, thresholds_ori = roc_curve(np.array(top_gt_list), np.array(top_pre_list))


# AUC = auc(fpr, tpr)
# AUC_ori = auc(fpr_ori, tpr_ori)
#
# AUC_new = roc_auc_score(np.array(gt_list), np.array(pre_score))
# AUC_ori_new = roc_auc_score(np.array(gt_list), np.array(pre_ori_list))

print('Accuracy', accuracy)
print('Sensitivity', sensitivity)
print('Specificity', specificity)
# print('AUC', AUC)
# print('AUC_new', AUC_new)

print('Accuracy_ori', accuracy1)
print('Sensitivity_ori', sensitivity1)
print('Specificity_ori', specificity1)
# print('AUC_ori', AUC_ori)
# print('AUC_ori_new', AUC_ori_new)


# matrix
plt.matshow(matrix, cmap=plt.cm.Greens)
plt.colorbar()
for x in range(len(matrix)):
    for y in range(len(matrix)):
        plt.annotate(matrix.T[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
plt.ylabel('True label')# 坐标轴标签
plt.xlabel('Predicted label')# 坐标轴标签
# plt.savefig('./figure/matrix_small.png', format='png')

plt.matshow(matrix2, cmap=plt.cm.Greens)
plt.colorbar()
for x in range(len(matrix2)):
    for y in range(len(matrix2)):
        plt.annotate(matrix2.T[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
plt.ylabel('True label')# 坐标轴标签
plt.xlabel('Predicted label')# 坐标轴标签
# plt.savefig('./figure/matrix_small_top_1000.png', format='png')
plt.show()


# roc_curve_me(np.array(gt_list), np.array(pre_score))
# roc_curve_me(np.array(gt_list), np.array(pre_ori_list))


# ROC曲线
# plt.figure()
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, color='darkorange', linewidth=1, markersize=8, label=u'New(AUC= %0.4f)' % AUC)
# plt.plot(fpr_ori, tpr_ori, color='navy', linewidth=1, markersize=8, label=u'Original(AUC= %0.4f)' % AUC_ori)
# plt.legend(loc='lower right', prop={'family':'Times New Roman','size':10})
# plt.xlim([-0.1, 1.1])
# plt.ylim([-0.1, 1.1])
# plt.ylabel('True Positive Rate', fontfamily="Times New Roman")
# plt.xlabel('False Positive Rate', fontfamily="Times New Roman")
# plt.title('Receiver operating characteristic example', fontfamily="Times New Roman")
# plt.grid(linestyle='-.')
# plt.grid(True)
# # plt.savefig('roc/ROC4.png',dpi=1200, bbox_inches='tight')
# plt.show()
