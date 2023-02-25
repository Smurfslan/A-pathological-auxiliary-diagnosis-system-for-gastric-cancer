import torch
import numpy as np
def calculate_miou(input,target,classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''

    inputTmp = torch.zeros([input.shape[0],classNum, input.shape[1],input.shape[2]]).cuda()#创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]]).cuda()#同上
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)#同上
    inputOht = inputTmp.scatter_(index=input,dim=1,value=1)#input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target,dim=1,value=1)#同上
    batchMious = []#为该batch中每张图像存储一个miou
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):#遍历图像
        ious = []
        for j in range(classNum):#遍历类别，包括背景
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            if union == 1e-6:
                continue
            iou = intersection / union
            ious.append(iou.item())
        miou = np.mean(ious)#计算该图像的miou
        batchMious.append(miou)
    return np.mean(batchMious)


def calculate_mdice(input,target,classNum):
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
    inputOht = inputTmp.scatter_(index=input,dim=1,value=1)#input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target,dim=1,value=1)#同上
    batchMious = []#为该batch中每张图像存储一个miou
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):#遍历图像
        ious = []
        for j in range(classNum):#遍历类别，包括背景
            intersection = 2 * torch.sum(mul[i][j]) + 1e-6
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) + 1e-6
            iou = intersection / union
            ious.append(iou.item())
        miou = np.mean(ious)#计算该图像的miou
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
    x=torch.sum(tmp).float()
    y=input.nelement()
    return (x / y)


def PA(input, target, num_class):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    _, predict = torch.max(input.data, 1)  # 按通道维度取最大,拿到每个像素分类的类别(1xhxw)
    predict = predict + 1  # 每个都加1避免从0开始,方便后面计算PA
    target = target + 1

    labeled = (target > 0) * (target <= num_class)  # 得到一个矩阵，其中，为true的是1，为false的是0
    # 标签中同时满足大于0 小于num_classes 的地方为T,其余地方为F  构成了一个蒙版
    pixel_labeled = labeled.sum()  # 计算标签的总和，是一个batch中的所有标签的总数
    # 注意  python中默认的T为1 F为0  调用sum就是统计正确的像素点的个数
    pixel_correct = ((predict == target) * labeled).sum()  # 将一个batch中预测正确的，且在标签范围内的像素点的值统计出来
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return ((pixel_correct.cpu().numpy()) / (pixel_labeled.cpu().numpy()))


def pre(input, target):
    input = input.data.cpu().numpy()
    target = target.data.cpu().numpy()

    # if np.max(input) == 0 and np.max(target) == 0:
    #     pre = torch.from_numpy(np.array([1.0]))
    # else:
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    pre = (TP+1e-6)/(TP+FP+1e-6)
    return pre


def recall(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()

    # if np.max(input) == 0 and np.max(target) == 0:
    #     recall = torch.from_numpy(np.array([1.0]))
    #
    # else:
    # TP   predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    recall=(TP+1e-6)/(TP+FN+1e-6)

    return recall


def F1score(input, target):

    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()

    # if np.max(input) == 0 and np.max(target) == 0:
    #     F1score = torch.from_numpy(np.array([1.0]))
    # else:
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    pre = (TP) / (TP + FP + 1e-6)
    recall=(TP)/(TP+FN + 1e-6)
    F1score=(2*(pre)*(recall)+1e-6)/(pre+recall+1e-6)
    return F1score


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

