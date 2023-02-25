import os
import sys
import json
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model_vgg import vgg
import torchvision
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from dataset import MyDataset, MyDataset_test
from model_me import Efficient_b3


def accuracy_cal(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    # top1 accuracy
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)  # 返回最大的k个结果（按最大到小排序）

    pred = pred.t()  # 转置

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res = correct_k.mul_(100.0 / batch_size)
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    parser = argparse.ArgumentParser(description='Pytorch classification Framework')
    parser.add_argument('--cuda_device', '-cuda', type=str, default='4', help='cuda_device_number')
    parser.add_argument('--epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', '-batch', type=int, default=4, help='train batch size')
    parser.add_argument('--csv_file', type=str, default='/mnt/ai2019/ljl/code/software_platform/infer/matrix/big/train_CycleGAN_aug_quan_221229_classification_big.csv', help='train_data')
    parser.add_argument('--test_csv_file', type=str, default='/mnt/ai2019/ljl/code/software_platform/infer/matrix/big/val_CycleGAN_aug_quan_221229_classification_big.csv',
                        help='test_data')
    parser.add_argument('--save_path', type=str, default='/mnt/ai2019/ljl/code/gastric_slide_classification/K_fold_model_big_new/',
                        help='save_model')
    parser.add_argument('--results', type=str,
                        default='/mnt/ai2019/ljl/code/gastric_slide_classification/results_big_new/',
                        help='save_results_metric')
    parser.add_argument('--model_name', type=str, default='resnet50',
                        help='use_model, resnet18, resnet34, resnet50, Efficient_b3, mobilenet_v3_large, shufflenet_v2_x0_5')

    args = parser.parse_args()
    np.set_printoptions(threshold=np.inf)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("using {} device.".format(device))

    # batch_size = 4
    # epochs = 30
    # csv_file = '/mnt/ai2019/ljl/code/software_platform/infer/train_big.csv'
    # test_csv_file = '/mnt/ai2019/ljl/code/software_platform/infer/val_big.csv'
    # save_path = '/mnt/ai2019/ljl/code/gastric_slide_classification/K_fold_model_big_new/'
    # results = '/mnt/ai2019/ljl/code/gastric_slide_classification/results_big_new/'
    # model_name = 'mobilenet_v3_large'

    train_transform = A.Compose([
        A.Resize(100, 100),
        A.RandomRotate90(),
        A.Flip(p=0.5),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(100, 100),
        ToTensorV2()
    ])

    total_data = pd.read_csv(args.csv_file)
    test_data = pd.read_csv(args.test_csv_file)

    imgs = [img for img in total_data['pre_array']]
    masks = [mask for mask in total_data['gt']]
    test_imgs = [img for img in test_data['pre_array']]
    test_masks = [mask for mask in test_data['gt']]

    test_dataset = MyDataset(test_imgs, test_masks, val_transform)
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    AUC_list = []
    sensitivity = 0

    # for train_index, test_index in kf.split(imgs):
    for k, (train_index, test_index) in enumerate(kf.split(imgs)):
        print('--------------begin {} fold--------------------'.format(k))
        train_imgs = [imgs[i] for i in train_index]
        train_masks = [masks[i] for i in train_index]
        val_imgs = [imgs[i] for i in test_index]
        val_masks = [masks[i] for i in test_index]

        train_dataset = MyDataset(train_imgs, train_masks, train_transform)
        train_num = len(train_dataset)
        nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=nw, drop_last=True)

        validate_dataset = MyDataset(val_imgs, val_masks, val_transform)
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=args.batch_size, shuffle=False,
                                                      num_workers=1)

        print("using {} images for training, {} images for validation.".format(train_num,
                                                                               val_num))

        # create model
        if args.model_name == 'mobilenet_v3_large':
            net = torchvision.models.mobilenet_v3_large(pretrained=False)
            net.classifier[3] = nn.Linear(in_features=1280, out_features=2, bias=True)
        elif args.model_name == 'resnet18':
            net = torchvision.models.resnet18(pretrained=True)
            net.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        elif args.model_name == 'resnet34':
            net = torchvision.models.resnet34(pretrained=True)
            net.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        elif args.model_name == 'resnet50':
            net = torchvision.models.resnet50(pretrained=True)
            net.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        elif args.model_name == 'Efficient_b3':
            net = Efficient_b3(3, 2)
        elif args.model_name == 'shufflenet_v2_x0_5':
            net = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
            net.fc = nn.Linear(in_features=1024, out_features=2, bias=True)

        # net = MobileNetV2(num_classes=2)

        # net = torchvision.models.mobilenet_v3_large(pretrained=False)
        # net.classifier[3] = nn.Linear(in_features=1280, out_features=2, bias=True)

        # net = torchvision.models.vgg11(pretrained=False)
        # net.classifier[3] = nn.Linear(in_features=4096, out_features=2, bias=True)

        # net = torchvision.models.resnet18(pretrained=True)
        # net.fc = nn.Linear(in_features=512, out_features=2, bias=True)

        # net = torchvision.models.resnet34(pretrained=True)
        # net.fc = nn.Linear(in_features=512, out_features=2, bias=True)

        # net = torchvision.models.resnet50(pretrained=True)
        # net.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

        # net = Efficient_b3(3, 2)

        # net = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
        # net.fc = nn.Linear(in_features=1024, out_features=2, bias=True)


        net.to(device)

        # define loss function
        loss_function = nn.CrossEntropyLoss()

        # construct an optimizer
        params = [p for p in net.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)

        best_acc = 0.0

        train_steps = len(train_loader)
        for epoch in range(args.epochs):
            # train
            net.train()
            running_loss = 0.0
            train_acc = AverageMeter()
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data['image'], data['label']
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device).long())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                acc = accuracy_cal(logits.data, labels.to(device).data)
                train_acc.update(acc.item(), images.size(0))

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f} Acc:{:.3f}".format(epoch + 1,
                                                                         args.epochs,
                                                                         loss, train_acc.avg)

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            losses_val = AverageMeter()
            with torch.no_grad():
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data['image'], val_data['label']
                    outputs = net(val_images.to(device))
                    loss_val = loss_function(outputs, val_labels.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    losses_val.update(loss_val.item(), val_images.size(0))

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               args.epochs)

            val_accurate = acc / val_num
            test_loss = losses_val.avg

            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate >= best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), args.save_path+args.model_name+'.pth')

            scheduler.step(test_loss)

        # torch.save(net.state_dict(), save_path + model_name + '_last.pth')

        # test
        net.load_state_dict(torch.load(args.save_path + args.model_name + '.pth', map_location=device))
        net.eval()
        gt_list = []
        pre_list = []
        pre_score = []
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                test_images, test_labels = test_data['image'], test_data['label']
                outputs = net(test_images.to(device))
                # predict = outputs.argmax(1)
                # predict_value = predict.cpu().numpy()[0]
                predict = torch.softmax(outputs, 1).cpu().numpy().squeeze()[1]
                if predict > 0.4:
                    predict_value = 1
                else:
                    predict_value = 0
                test_labels_value = test_labels.cpu().numpy()[0]
                gt_list.append(test_labels_value)
                pre_list.append(predict_value)
                pre_score.append(predict)

        matrix = confusion_matrix(gt_list, pre_list)
        fpr, tpr, thresholds = roc_curve(np.array(gt_list), np.array(pre_score))
        TP = matrix[1][1]
        FP = matrix[0][1]
        FN = matrix[1][0]
        TN = matrix[0][0]

        accuracy = (TP + TN) / (TP + FP + FN + TN)
        sensitivity_tmap = TP / (TP + FN)
        specificity = TN / (TN + FP)
        AUC = auc(fpr, tpr)
        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity_tmap)
        specificity_list.append(specificity)
        AUC_list.append(AUC)

        print('accuracy: %.3f  sensitivity: %.3f  specificity: %.3f' %
              (accuracy, sensitivity_tmap, specificity))

        if sensitivity_tmap > sensitivity:
            sensitivity = sensitivity_tmap
            torch.save(net.state_dict(), args.save_path + args.model_name + '_best.pth')

    accuracy_mean = np.mean(np.array(accuracy_list))
    accuracy_std = np.std(np.array(accuracy_list))
    sensitivity_mean = np.mean(np.array(sensitivity_list))
    sensitivity_std = np.std(np.array(sensitivity_list))
    specificity_mean = np.mean(np.array(specificity_list))
    specificity_std = np.std(np.array(specificity_list))
    AUC_mean = np.mean(np.array(AUC_list))
    AUC_std = np.std(np.array(AUC_list))

    print('accuracy_mean', accuracy_mean)
    print('accuracy_std', accuracy_std)
    print('sensitivity_mean', sensitivity_mean)
    print('sensitivity_std', sensitivity_std)
    print('specificity_mean', specificity_mean)
    print('specificity_std', specificity_std)
    print('AUC_mean', AUC_mean)
    print('AUC_std', AUC_std)

    file = open(args.results+args.model_name+'.txt', 'w')
    file.write('accuracy_mean:' + str(accuracy_mean) + '\n')
    file.write('accuracy_std:' + str(accuracy_std) + '\n')
    file.write('sensitivity_mean:' + str(sensitivity_mean) + '\n')
    file.write('sensitivity_std:' + str(sensitivity_std) + '\n')
    file.write('specificity_mean:' + str(specificity_mean) + '\n')
    file.write('specificity_std:' + str(specificity_std) + '\n')
    file.write('AUC_mean:' + str(AUC_mean) + '\n')
    file.write('AUC_std:' + str(AUC_std) + '\n')
    file.close()

    print('Finished Training')


if __name__ == '__main__':
    main()
