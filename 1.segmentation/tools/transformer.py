import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter
from skimage import color
import numbers


def transforms_for_rot(ema_inputs):

    rot_mask = np.random.randint(0, 4, ema_inputs.shape[0])
    flip_mask = np.random.randint(0, 2, ema_inputs.shape[0])

    # flip_mask = [0,0,0,0,1,1,1,1]
    # rot_mask = [0,1,2,3,0,1,2,3]
    ema_outputs = torch.zeros(ema_inputs.shape).to(ema_inputs.get_device())

    for idx in range(ema_inputs.shape[0]):
        if flip_mask[idx] == 1:
            ema_outputs[idx] = torch.flip(ema_inputs[idx], [1])
        else:
            ema_outputs[idx] = ema_inputs[idx]

        if len(ema_inputs.shape) == 4:
            ema_outputs[idx] = torch.rot90(ema_outputs[idx], int(rot_mask[idx]), dims=[1, 2])
        if len(ema_inputs.shape) == 3:
            ema_outputs[idx] = torch.rot90(ema_outputs[idx], int(rot_mask[idx]), dims=[0, 1])

    return ema_outputs, rot_mask, flip_mask


def transforms_for_fixed_rot(ema_inputs, rot_mask, flip_mask):

    # rot_mask = np.random.randint(0, 4, ema_inputs.shape[0])
    # flip_mask = np.random.randint(0, 2, ema_inputs.shape[0])

    # flip_mask = [0,0,0,0,1,1,1,1]
    # rot_mask = [0,1,2,3,0,1,2,3]

    ema_outputs = torch.zeros(ema_inputs.shape).to(ema_inputs.get_device())

    for idx in range(ema_inputs.shape[0]):
        if flip_mask[idx] == 1:
            ema_outputs[idx] = torch.flip(ema_inputs[idx], [1])
        else:
            ema_outputs[idx] = ema_inputs[idx]

        if len(ema_inputs.shape) == 4:
            ema_outputs[idx] = torch.rot90(ema_outputs[idx], int(rot_mask[idx]), dims=[1, 2])
        elif len(ema_inputs.shape) == 3:
            ema_outputs[idx] = torch.rot90(ema_outputs[idx], int(rot_mask[idx]), dims=[0, 1])

    return ema_outputs


def transforms_for_noise(inputs_u2, std):

    gaussian = np.random.normal(0, std, (inputs_u2.shape[0], 3, inputs_u2.shape[-1], inputs_u2.shape[-1]))
    gaussian = torch.from_numpy(gaussian).float().cuda()
    inputs_u2_noise = inputs_u2 + gaussian

    return inputs_u2_noise, gaussian


class HEDJitter(object):
    """Randomly perturbe the HED color space value an RGB image.
    First, it disentangled the hematoxylin and eosin color channels by color deconvolution method using a fixed matrix.
    Second, it perturbed the hematoxylin, eosin and DAB stains independently.
    Third, it transformed the resulting stains into regular RGB color space.
    Args:
        theta (float): How much to jitter HED color space,
         alpha is chosen from a uniform distribution [1-theta, 1+theta]
         betti is chosen from a uniform distribution [-theta, theta]
         the jitter formula is **s' = \alpha * s + \betti**
    """
    def __init__(self, theta=0.): # HED_light: theta=0.05; HED_strong: theta=0.2
        assert isinstance(theta, numbers.Number), "theta should be a single number."
        self.theta = theta
        self.alpha = np.random.uniform(1-theta, 1+theta, (1, 3))
        self.betti = np.random.uniform(-theta, theta, (1, 3))

    @staticmethod
    def adjust_HED(img, alpha, betti):
        img = np.array(img)

        s = np.reshape(color.rgb2hed(img), (-1, 3))
        ns = alpha * s + betti  # perturbations on HED color space
        nimg = color.hed2rgb(np.reshape(ns, img.shape))

        imin = nimg.min()
        imax = nimg.max()
        rsimg = (255 * (nimg - imin) / (imax - imin)).astype('uint8')  # rescale to [0,255]
        # transfer to PIL image
        return rsimg

    def __call__(self, img):
        return self.adjust_HED(img, self.alpha, self.betti)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'theta={0}'.format(self.theta)
        format_string += ',alpha={0}'.format(self.alpha)
        format_string += ',betti={0}'.format(self.betti)
        return format_string


def transforms_for_HEDJitter(ema_inputs):
    preprocess = HEDJitter(theta=0.05)
    devive = ema_inputs.get_device()

    with torch.no_grad():
        ema_outputs = torch.zeros(ema_inputs.shape)
        ema_outputs = ema_outputs.cpu().numpy()

        for idx in range(ema_inputs.shape[0]):
            trans_arr = ema_inputs[idx].cpu().numpy()
            trans_arr = trans_arr.transpose(1, 2, 0)
            trans_arr = preprocess((trans_arr*255).astype(np.uint8))
            trans_arr = trans_arr.transpose(2, 0, 1)
            ema_outputs[idx] = trans_arr/255

        ema_outputs = torch.from_numpy(ema_outputs).to(devive)

    return ema_outputs


