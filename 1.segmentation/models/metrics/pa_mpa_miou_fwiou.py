# 输入：测试集标注文件夹和对应的预测文件夹，图片命名为 标注+x；predict+x
# 输出：四个指标的平均值

from models.metrics.eval_segm import *
import cv2
import os

pa = []
mpa = []
miou = []
fwiou = []


# net_pred: /mnt/ai2019/deng/project/U-2-Net/test_data/unet_results
# pre_dir = '/mnt/ai2019/deng/Data/gastric_s2_s4/test/cro_512_mask/'
# gt_dir = '/mnt/ai2019/deng/Data/gastric_s2_s4/test/cro_512_mask/'
# metric_log = pre_dir + '/log/'


def Miou(input, target):
    input_np = input.detach().to(device='cpu').numpy()
    target_np = target.detach().to(device='cpu').numpy()
    # input_np[input_np>=0.5]=255
    # input_np[input_np<0.5]=0
    # target_np[target_np >= 0.5] = 255
    # target_np[target_np < 0.5] = 0
    # input_np = input_np.astype('uint8')
    # target_np = target_np.astype('uint8')
    miou=[]
    for i in range(input.shape[0]):
        inp = input_np[i,:,:]
        tar = target_np[i,:,:]
        miou_temporary = mean_IU(inp, tar)
        miou.append(miou_temporary)

    return sum(miou)/input.shape[0]

    # for i in range(input_np.shape[0]):
    #     inp = threshold_demo(input_np[i])
    #     tar = threshold_demo(target_np[i])
    #     miou_temporary = mean_IU(inp, tar)
    #     miou.append(miou_temporary)
    # return sum(miou)/input.shape[0]


def metric(pre_dir, gt_dir, metric_log):
    pa = []
    mpa = []
    miou = []
    fwiou = []
    pre_list = [f for f in os.listdir(pre_dir) if os.path.isfile(os.path.join(pre_dir, f))]
    pre_list = sorted(pre_list)
    gt_list = sorted(os.listdir(gt_dir))
    # print(len(pre_list))
    number_photo = len(pre_list)
    i=0
    for file in pre_list:
        print(i, len(pre_list), file)
        i+=1
        gt = os.path.splitext(file)[0] + '.png'
        pre_file = pre_dir + file
        # print(pre_file)
        gt_file = gt_dir + gt


        a = cv2.imread(pre_file)
        b = cv2.imread(gt_file)
        '''
        cv.namedWindow("a", cv.WINDOW_NORMAL)
        cv.imshow("a", a)
        cv.waitKey()
        cv.namedWindow("b", cv.WINDOW_NORMAL)
        cv.imshow("b", b)
        cv.waitKey()
        '''
        binary = threshold_demo(a)
        binary1 = threshold_demo(b)
        '''
        cv.namedWindow("binary", cv.WINDOW_NORMAL)
        cv.imshow("binary", binary)
        cv.waitKey()
        cv.namedWindow("binary1", cv.WINDOW_NORMAL)
        cv.imshow("binary1", binary1)
        cv.waitKey()
        '''
        # binary1=cv.resize(binary1,(224,224))
        # print(binary.shape)
        # print(binary1.shape)

        # 计算分割指标
        pa_temporary = pixel_accuracy(binary, binary1)
        mpa_temporary = mean_accuracy(binary, binary1)
        miou_temporary = mean_IU(binary, binary1)
        fwiou_temporary = frequency_weighted_IU(binary, binary1)

        pa.append(pa_temporary)
        mpa.append(mpa_temporary)
        miou.append(miou_temporary)
        fwiou.append(fwiou_temporary)

    print('average pa:', sum(pa)/number_photo)
    print('average mpa:', sum(mpa)/number_photo)
    print('average miou:', sum(miou)/number_photo)
    print('average fwiou:', sum(fwiou)/number_photo)

    print("save metric_log_file to " + metric_log)
    if not os.path.exists(metric_log):
        os.makedirs(metric_log)
    file = open(os.path.join(metric_log, 'log.txt'), 'w')
    file.write('average pa:' + str(sum(pa)/number_photo) + '\n')
    file.write('average mpa:' + str(sum(mpa)/number_photo) + '\n')
    file.write('average miou:' + str(sum(miou)/number_photo) + '\n')
    file.write('average fwiou:' + str(sum(fwiou)/number_photo) + '\n')
    file.close()


def threshold_demo(a):
    # gray = cv2.cvtColor(a.transpose(2, 0, 1), cv2.COLOR_BGR2GRAY)  # 把输入图像灰度化
    gray = a[:, :, 0]
    # plt.figure()
    # plt.subplot(121),plt.imshow(gray,'gray')
    # plt.subplot(122),plt.hist(gray.ravel(),256)
    # plt.show()
    # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割
    ret, binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    # ret, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((3, 3),np.uint8)
    # erosion = cv2.erode(binary, kernel)
    # dst = cv2.dilate(erosion, np.ones((5, 5), np.uint8))
    # print("threshold value %s" % ret)
    return binary
# def threshold_demo(a):
#     gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
#     # plt.figure()
#     # plt.subplot(121),plt.imshow(gray,'gray')
#     # plt.subplot(122),plt.hist(gray.ravel(),256)
#     # plt.show()
#     # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
#     # ret, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
#     ret, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
#     kernel = np.ones((3, 3),np.uint8)
#     erosion = cv2.erode(binary, kernel)
#     dst = cv2.dilate(erosion, np.ones((5, 5), np.uint8))
#     # print("threshold value %s" % ret)
#     return dst
#
# def metric(pre_dir, gt_dir, metric_log):
#     pa = []
#     mpa = []
#     miou = []
#     fwiou = []
#     pre_list = sorted(os.listdir(pre_dir))
#     pre_list = [f for f in os.listdir(pre_dir) if os.path.isfile(os.path.join(pre_dir, f))]
#     gt_list = sorted(os.listdir(gt_dir))
#     # print(gt_list)
#     number_photo = len(pre_list)
#     for file in pre_list:
#         # print(file)
#         gt = os.path.splitext(file)[0] + '_mask.png'
#         pre_file = pre_dir + file
#         # print(pre_file)
#         gt_file = gt_dir + gt
#         # print(gt_file)
#
#
#         a = cv2.imread(pre_file)
#         b = cv2.imread(gt_file)
#         '''
#         cv.namedWindow("a", cv.WINDOW_NORMAL)
#         cv.imshow("a", a)
#         cv.waitKey()
#         cv.namedWindow("b", cv.WINDOW_NORMAL)
#         cv.imshow("b", b)
#         cv.waitKey()
#         '''
#         binary = threshold_demo(a)
#         binary1 = threshold_demo(b)
#         '''
#         cv.namedWindow("binary", cv.WINDOW_NORMAL)
#         cv.imshow("binary", binary)
#         cv.waitKey()
#         cv.namedWindow("binary1", cv.WINDOW_NORMAL)
#         cv.imshow("binary1", binary1)
#         cv.waitKey()
#         '''
#         # binary1=cv.resize(binary1,(224,224))
#         # print(binary.shape)
#         # print(binary1.shape)
#
#         # 计算分割指标
#         pa_temporary = pixel_accuracy(binary, binary1)
#         mpa_temporary = mean_accuracy(binary, binary1)
#         miou_temporary = mean_IU(binary, binary1)
#         fwiou_temporary = frequency_weighted_IU(binary, binary1)
#
#         pa.append(pa_temporary)
#         mpa.append(mpa_temporary)
#         miou.append(miou_temporary)
#         fwiou.append(fwiou_temporary)
#
#     print('average pa:', sum(pa)/number_photo)
#     print('average mpa:', sum(mpa)/number_photo)
#     print('average miou:', sum(miou)/number_photo)
#     print('average fwiou:', sum(fwiou)/number_photo)
#
#     print("save metric_log_file to " + metric_log)
#     if not os.path.exists(metric_log):
#         os.makedirs(metric_log)
#     file = open(os.path.join(metric_log, 'log.txt'), 'w')
#     file.write('average pa:' + str(sum(pa) / number_photo) + '\n')
#     file.write('average mpa:' + str(sum(mpa) / number_photo) + '\n')
#     file.write('average miou:' + str(sum(miou) / number_photo) + '\n')
#     file.write('average fwiou:' + str(sum(fwiou) / number_photo) + '\n')
#     file.close()

if __name__ == '__main__':
    metric(pre_dir, gt_dir, metric_log)