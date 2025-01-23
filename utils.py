import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from AnomalyCLIP_lib.transform import image_transform
from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)
def get_transform(args):
    preprocess = image_transform(args.image_size, is_train=False, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    preprocess.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                    max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))
    return preprocess, target_transform
class winSplit(nn.Module):
    def __init__(self, imgsize, patchsize, winsize, stride):
        '''
            #mvtec imgsize=518, patchsize=14, winsize=17, stride=10, winZipSize=17 91.9,93.2
            #visa  imgsize=518, patchsize=14, winsize=17, stride=10, winZipSize=17 96.1,84.1
            #mpdd imgsize=518, patchsize=14, winsize=17, stride=10, winZipSize=17 96.9,79.5
            #btad imgsize=518, patchsize=14, winsize=17, stride=10, winZipSize=17 96.6,94.5
            #sdd imgsize=518, patchsize=14, winsize=17, stride=10, winZipSize=17  94.2,93.8
            #dagm imgsize=518, patchsize=14, winsize=17, stride=10, winZipSize=17 96.8,99.1
            #dtd imgsize=518, patchsize=14, winsize=17, stride=10, winZipSize=17
            37-17=20
            37-13=24
            37-10=27
            imgsize: 要切分的图像大小
            patchsize: 切分单位大小
            winsize: 窗口大小, 单位patch
            stride: 滑动步长, 单位patch 2,3,4,6
            winZipSize: 窗口压缩尺寸, 单位patch
        '''
        super(winSplit, self).__init__()
        self.imgsize = imgsize
        self.patchsize = patchsize
        self.winsize = winsize
        self.stride = stride
        self.masklist, self.pmasklist = self.makemask()

    def makemask(self):
        # patch_H = int(self.imgsize / self.patchsize)
        winsize = self.winsize * self.patchsize
        stride = self.stride * self.patchsize

        masklist = []
        pmasklist = []
        for row in range(0, self.imgsize - winsize + 1, stride):
            for col in range(0, self.imgsize - winsize + 1, stride):
                masklist.append([row, row + winsize, col, col + winsize])
                pmasklist.append([int(row / self.patchsize), int((row + winsize) / self.patchsize),
                                  int(col / self.patchsize), int((col + winsize) / self.patchsize)])

        return masklist, pmasklist

    def forward(self, x):
        # B, C, H, _ = x.shape
        x_ = []
        for mask in self.masklist:
            row_start = mask[0]
            row_end = mask[1]
            col_start = mask[2]
            col_end = mask[3]

            x__ = x[:, :, row_start:row_end, col_start:col_end]  # b * c * winsize * winsize
            x_.append(x__)

        x_ = torch.stack(x_)  # winnum * b * c * winszie(13*14) * winszie
        _, B, C, H, _ = x_.shape
        x_ = x_.view(-1, C, H, H)  # (winnum * b) * c * winszie * winszie
        winzipsize = self.winsize * self.patchsize
        x_ = F.interpolate(x_, size=winzipsize, mode='bilinear',
                           align_corners=True)  # (winnum * b) * c * winzipsize * winzipsize

        return x_, self.pmasklist


def calWinAnoMap(winmap, masklist, winsize, picsize, device, winzsize=0, usewinimgencoder=0, convRes=0):
    '''
        winmap: (b*winnum) * n-1 * 2
        masklist: winnum * 4
        winsize: int 5
        winzsize: int 7
        picsize: int 37
        usewinimgencoder 是否使用了滑动窗口图像编辑器
    '''
    winzsize = winsize
    winnum = len(masklist)
    #print('winnum: {}'.format(winnum))
    B, N, C = winmap.shape
    #print('winmap shape: {}'.format(winmap.shape))
    # if winzsize != winsize:  # 当压缩窗口 需要放大操作
    #     winmap = winmap.permute(0, 2, 1)
    #     H = int(np.sqrt(winmap.shape[2]))
    #     winmap = winmap.view(B, C, H, H)
    #     winmap = F.interpolate(winmap, size=winsize, mode='bilinear', align_corners=True)
    #     winmap = winmap.view(B, C, -1)
    #     winmap = winmap.permute(0, 2, 1)

    # 如果没有使用滑动窗口编码器 需要缩小操作
    if usewinimgencoder == 0 and winsize != picsize:
        winmap = winmap.permute(0, 2, 1)
        H = int(np.sqrt(N))
        winmap = winmap.view(B, C, H, H)
        # print('winmap shape: {}'.format(winmap.shape))
        winmap = F.interpolate(winmap, size=winsize, mode='bilinear', align_corners=True)#9,2,17*17
        # print('winmap shape: {}'.format(winmap.shape))
        winmap = winmap.view(B, C, -1)#9,2,289
        winmap = winmap.permute(0, 2, 1)#9,289,2
        # print('winmap shape: {}'.format(winmap.shape))
    B, N, C = winmap.shape# 9,289,2

    winmap = winmap.view(-1, winnum, N, C)  # 1,9,289,2
    b, _, _, _ = winmap.shape

    anoMaps = []
    for wm in winmap:  # wm: 9,289,2
        anowm = wm[:, :, 1]  # 9,289
        anowm = anowm.view(winnum, winsize, winsize)  # winnum * wsize * wsize(9,17,17)

        anoMap = torch.zeros(picsize, picsize).to(device)
        calnum = torch.zeros(picsize, picsize).to(device)
        #np.set_printoptions(threshold=np.inf)
        for i in range(winnum):
            mask = masklist[i]
            row_start = mask[0]
            row_end = mask[1]
            col_start = mask[2]
            col_end = mask[3]
            anoMap[row_start:row_end, col_start:col_end] += anowm[i]
            calnum[row_start:row_end, col_start:col_end] += 1
            #print(calnum.cpu().numpy())
        anoMap = anoMap / calnum
        anoMaps.append(anoMap)  # b * picsize * picsize
    anoMaps = torch.stack(anoMaps)
    anoMaps = anoMaps.view(b, -1)  # b * n
    t = [1. - anoMaps, anoMaps]
    t = torch.stack(t)  # 2 * b * n
    t = t.permute(1, 2, 0)  # b * n * 2
    if convRes != 0:
        t = convRes(t)  # b * n * 2
    anoMaps = t[:, :, 1]
    anoMaps = anoMaps.view(b, picsize, picsize)#[b,37,37]
    anoMaps = anoMaps.unsqueeze(1)#[b,1,37,37]
    return anoMaps