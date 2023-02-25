import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
from functools import partial
import segmentation_models_pytorch as smp


class Efficient_b3(nn.Module):
    def __init__(self, in_ch, out_ch, depth=4):
        super(Efficient_b3, self).__init__()
        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=in_ch, depth=depth, weights='imagenet')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(32, out_ch)

    def forward(self, x):
        eff_model = self.eff(x)
        e0 = eff_model[0]
        e1 = eff_model[1]
        e2 = eff_model[2]
        # e3 = eff_model[3]

        cls_avg = self.avgpool(e2)
        cls_flat = torch.flatten(cls_avg, 1)
        cls_out = self.fc(cls_flat)

        return cls_out

