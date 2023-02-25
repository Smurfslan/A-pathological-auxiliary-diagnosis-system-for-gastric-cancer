"""
    Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    head= d-mi
    down = ma-mi
    dn = (d-mi)/(ma-mi)

    return dn

def normalization(data):
    if (np.max(data) < 1e-6):
        dn = data
    else:
        ma = np.max(data)
        mi = np.min(data)
        dn = (data - mi) / (ma - mi)
    return dn

    # _range = np.max(data) - np.min(data)
    # return (data - np.min(data)) / _range

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def mkdir_exists_chmod(path):
    if not os.path.exists(path):
        oldmask = os.umask(000)
        os.makedirs(path + '/state/', mode=0o777, exist_ok=True)
        os.makedirs(path + '/model/', mode=0o777, exist_ok=True)
        os.makedirs(path + '/val_img/', mode=0o777, exist_ok=True)
        os.umask(oldmask)

def makedir(path):
    if not os.path.exists(path):
        oldmask = os.umask(000)
        os.makedirs(path, mode=0o777, exist_ok=True)
        os.umask(oldmask)


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, consistency, consistency_rampup):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    # alpha = min(1 - 1 / global_step, alpha)
    alpha = min((global_step-1)*0.0001, alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
