from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
import numpy as np
from PIL import Image
import glob
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from models.Models import *
from models.metrics.pa_mpa_miou_fwiou import *
import models.Models as Models
import argparse
import sys
from models.u2net import U2NET
parser = argparse.ArgumentParser(description='Pytorch Segmentation Framework')
# parser.add_argument('--model_name', '-name', type=str, default='U_Net_test', help='model_name')
# parser.add_argument('--batch_size_train', '-batch', type=int, default=8, help='train batch size')
parser.add_argument('--cuda_device', '-cuda', type=str, default='0', help='cuda_device_number')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------- 1. get image path and name ---------
"""The parameters you need to change"""
model_name = 'U_Net'  # DeepLabv3_plus AttU_Net
model_dir = '/mnt/ai2019/ljl/code/software_platform/train/torch_framework/experiments/Unet_b3_total_20211105_augmentchanged/state/20000.pth'
prediction_dir = model_dir.rsplit('/', 2)[0] + '/test_img/'
image_dir = '/mnt/ai2019/ljl/data/gastric/total/2048/ljl/2048/resize_512/val_image/'
gt_dir = '/mnt/ai2019/ljl/data/gastric/total/2048/ljl/2048/resize_512/val_mask/'
metric_log = prediction_dir + '/log/'
img_name_list = glob.glob(image_dir + os.sep + '*')
img_name_list = sorted(img_name_list)
# --------- 2. dataloader ---------
# 1. dataloader
test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [], transform=transforms.Compose([RescaleT(1024), ToTensorLab(flag=0)]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=8)
# --------- 3. model define ---------
net = U2NET(3, 1).to(device)
net.load_state_dict(torch.load(model_dir))
net = net.to(device)
net.eval()

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn
def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    imo = imo.transpose(Image.FLIP_LEFT_RIGHT)

    pb_np = np.array(imo)
    pb_np = threshold_demo(pb_np)
    # asd = Image.fromarray(pb_np * 255).convert('RGB')
    arr = np.expand_dims(pb_np, axis=2)
    asd = np.concatenate((arr, arr, arr), axis=2)
    asd = Image.fromarray(asd).convert('RGB')

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    asd.save(d_dir + imidx + '.png')

def test():
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor).to(device)
        d1 = net(inputs_test)
        pred = d1[0][:, 0, :, :]
        pred = normPRED(pred).data.cpu().numpy()
        imo = Image.fromarray(pred.squeeze() * 255).convert('RGB').resize((2048, 2048), resample=Image.BILINEAR).transpose(Image.FLIP_LEFT_RIGHT)  # <PIL.Image.Image image mode=RGB size=320x320 at 0x7F9FC2490780>
        pb_np = np.array(imo)  # ndarray:(512,512,3)
        pre_bin = pb_np.copy()
        pre_bin[pre_bin >= 127] = 255
        pre_bin[pre_bin < 127] = 0

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        cv2.imwrite(prediction_dir + img_name_list[i_test].split(os.sep)[-1],
                    pre_bin)
        sys.stdout.write('\rImage number:%d/%d, inferencing:%s' % (i_test+1, len(img_name_list), img_name_list[i_test].split(os.sep)[-1]))
    print('\n--------------------metric')
    metric(prediction_dir, gt_dir, metric_log)

if __name__ == "__main__":
    test()
    
