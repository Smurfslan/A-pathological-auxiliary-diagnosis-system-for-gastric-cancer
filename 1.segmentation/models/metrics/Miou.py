import torch
import numpy as np

def iou_mean(pred, target, n_classes = 1):
# n_classes ：the number of classes in your dataset,not including background
# for mask and ground-truth label, not probability map
  ious = []
  iousSum = 0
  pred = torch.from_numpy(pred)
  pred = pred.view(-1)
  target = np.array(target)
  target = torch.from_numpy(target)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
      iousSum += float(intersection) / float(max(union, 1))
  return iousSum/n_classes

def calculate_miou(input,target,classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''

    # input_img = input.squeeze(1)
    # target_img = target.squeeze(1)
    # input_view = input_img.view(-1)
    # targget_view = target_img.view(-1)
    # batchMious = []  # 为该batch中每张图像存储一个miou
    # mul = input_view * targget_view  # 乘法计算后，其中1的个数为intersection
    #
    # ious = []
    # a=torch.sum(input_view)
    # b= torch.sum(targget_view)
    # intersection = torch.sum(mul)
    #
    # onehot = torch.sum(input_view) + torch.sum(targget_view)
    #
    # union = torch.sum(input_view) + torch.sum(targget_view) - intersection + 1e-6
    # un = torch.sum(input_view) + torch.sum(targget_view) - intersection
    # iou = intersection / union
    # return iou

    inputTmp = torch.zeros([input.shape[0],classNum, input.shape[1],input.shape[2]]).cuda()  # 创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]]).cuda()  # 同上
    input = input.unsqueeze(1)  # 将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)  # 同上
    ii = torch.sum(input)
    tt = torch.sum(target)
    inputOht = inputTmp.long().scatter_(index=input, dim=1, value=1)  # input作为索引，将0矩阵转换为onehot矩阵，scatter需要torch.int64
    targetOht = targetTmp.long().scatter_(index=target, dim=1, value=1)  # 同上
    iii = torch.sum(inputOht)
    ttt= torch.sum(targetOht)
    batchMious = []  # 为该batch中每张图像存储一个miou
    mul = inputOht * targetOht  # 乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):  # 遍历图像
        ious = []
        for j in range(classNum):  # 遍历类别，包括背景
            intersection = torch.sum(mul[i][j])
            a = torch.sum(inputOht)
            b = torch.sum(targetOht)
            onehot = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            if union == 1e-6:
                continue
            iou = intersection / union
            ious.append(iou.item())
        miou = np.mean(ious)  # 计算该图像的miou
        batchMious.append(miou)
    return np.mean(batchMious)


def Pa(input, target):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    tmp = input == target

    return (torch.sum(tmp).float() / input.nelement())


def calculate_fwiou(input,target,classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    inputTmp = torch.zeros([input.shape[0],classNum,input.shape[1],input.shape[2]]).cuda()#创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]]).cuda()#同上
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)#同上
    inputOht = inputTmp.scatter_(index=input, dim=1, value=1)#input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target, dim=1, value=1)#同上
    batchFwious = []#为该batch中每张图像存储一个miou
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):#遍历图像
        fwious = []
        for j in range(classNum):#遍历类别，包括背景
            TP_FN = torch.sum(targetOht[i][j])
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            if union == 1e-6:
                continue
            iou = intersection / union
            fwiou = (TP_FN/(input.shape[2]*input.shape[3])) * iou
            fwious.append(fwiou.item())
        fwiou = np.mean(fwious)#计算该图像的miou
        # print(miou)
        batchFwious.append(fwiou)
    return np.mean(batchFwious)

