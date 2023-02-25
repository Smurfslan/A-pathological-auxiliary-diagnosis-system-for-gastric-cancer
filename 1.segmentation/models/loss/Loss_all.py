import torch
import torch.nn as nn
import random


import numpy as np

import torch
import torch.nn.functional as F

# bce_loss = nn.BCELoss(reduction='mean')
bce_loss = nn.CrossEntropyLoss(reduction='sum')
mse_loss = torch.nn.MSELoss(reduce=True, size_average=True, reduction='sum')


def dice_loss(pred, target):
    pred = pred.sigmoid().view(-1)
    target = target.view(-1)

    numerator = 2.0 * (pred * target).sum() + 1.0
    denominator = pred.sum() + target.sum() + 1.0

    return numerator / denominator


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma, classes):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.n_classes = classes

    def to_one_hot(self, tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).to('cuda').scatter_(1, tensor.view(n, 1, h, w), 1)

        return one_hot

    def forward(self, input, target):
        target_onehot = self.to_one_hot(target.type(torch.LongTensor).to('cuda'), self.n_classes)

        loss = self.alpha * self.focal(input, target_onehot) - torch.log(dice_loss(input, target_onehot))
        return loss.mean()


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss = nn.BCELoss(reduction='mean')
    # bce_loss = nn.CrossEntropyLoss()
    # loss0 = F.cross_entropy(d0, labels_v)
    # bce_loss = CrossEntropyLoss2D()
    loss0 = bce_loss(d0.squeeze(1), labels_v)
    loss1 = bce_loss(d1.squeeze(1), labels_v)
    loss2 = bce_loss(d2.squeeze(1), labels_v)
    loss3 = bce_loss(d3.squeeze(1), labels_v)
    loss4 = bce_loss(d4.squeeze(1), labels_v)
    loss5 = bce_loss(d5.squeeze(1), labels_v)
    loss6 = bce_loss(d6.squeeze(1), labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],loss6.data[0]))

    return loss0, loss


class CrossEntropyLoss2D(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2D, self).__init__()

    def forward(self, input, target):

        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        if h != ht or w != wt:
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target,  ignore_index=250
        )
        return loss


# class FocalLoss(nn.Module):
#     def __init__(self):
#         super(FocalLoss, self).__init__()
#
#     def forward(self, input, target):
#         n, c, h, w = input.size()
#         logpt = -self.CrossEntropyLoss(input=input, target=target)
#         pt = torch.exp(logpt)
#
#         loss = -((1 - pt) ** 2) * logpt
#
#         loss /= n
#
#         return loss
#
#     def CrossEntropyLoss(self, input, target):
#         n, c, h, w = input.size()
#         nt, ht, wt = target.size()
#
#     # Handle inconsistent size between input and target
#         if h != ht or w != wt:
#             input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
#
#         input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
#         target = target.view(-1)
#         loss = F.cross_entropy(input, target,  ignore_index=250)
#         return loss



#针对多分类问题，二分类问题更简单一点。
class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).to('cuda').scatter_(1, tensor.view(n, 1, h, w), 1)

        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return (1 - loss.mean())



class DiceLoss(nn.Module):

    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    # one_hot编码了解一下
    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).to('cuda').scatter_(1, tensor.view(n, 1, h, w), 1)

        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = 2 * inter / (union + 1e-16)

        # Return average loss over classes and batch
        return (1 - loss.mean())

class DiceLossCeFocal(nn.Module):

    def __init__(self, n_classes):
        super(DiceLossCeFocal, self).__init__()
        self.n_classes = n_classes

    # one_hot编码了解一下
    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).to('cuda').scatter_(1, tensor.view(n, 1, h, w), 1)

        return one_hot

    def CrossEntropyLoss(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
        if h != ht or w != wt:
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input, target,  ignore_index=250)
        return loss

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W
        n, c, h, w = input.size()
        logpt = -self.CrossEntropyLoss(input=input, target=target)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** 2) * logpt

        focal_loss /= n

        N = len(input)
        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = 2 * inter / (union + 1e-16)
        total_loss = ((1 - loss.mean()) + focal_loss)
        # Return average loss over classes and batch
        return total_loss


class DiceLossCe(nn.Module):

    def __init__(self, n_classes):
        super(DiceLossCe, self).__init__()
        self.n_classes = n_classes

    # one_hot编码了解一下
    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).to('cuda').scatter_(1, tensor.view(n, 1, h, w), 1)

        return one_hot

    def CrossEntropyLoss(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
        if h != ht or w != wt:
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input, target,  ignore_index=250)
        return loss

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W
        ce_loss = self.CrossEntropyLoss(input=input, target=target)

        N = len(input)
        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = 2 * inter / (union + 1e-16)
        total_loss = (0.6*(1 - loss.mean()) + 0.4*ce_loss)
        # Return average loss over classes and batch
        return total_loss