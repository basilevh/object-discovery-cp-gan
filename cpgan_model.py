# Basile Van Hoorick, March 2020
# Common code for PyTorch implementation of Copy-Pasting GAN

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cpgan_data import *
from cpgan_tools import *


def create_gaussian_filter(blur_sigma):
    bs_round = int(blur_sigma)
    kernel_size = bs_round * 2 + 1
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1.0) / 2.0
    variance = blur_sigma ** 2.0
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)
    gaussian_filter = nn.Conv2d(3, 3, kernel_size=kernel_size, padding=bs_round, groups=3, bias=False)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter
    

# ==== Code below is adapted from ====
# https://github.com/milesial/Pytorch-UNet

# Adapted for CP-GAN: 3, 64, 128, 256, 512, 256, 128, 64, C


class MyUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, blur_sigma=0.0, border_zero=False):
        super(MyUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.blur_sigma = blur_sigma
        self.border_zero = border_zero

        self.inc = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.down1 = MyDown(64, 128)
        self.down2 = MyDown(128, 256)
        self.down3 = MyDown(256, 512)

        # Fully connected layers for discriminator output score
        self.avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1)
        )

        self.up1 = MyUp(768, 256, bilinear)
        self.up2 = MyUp(384, 128, bilinear)
        self.up3 = MyUp(192, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        if blur_sigma:
            self.gaussian_filter = create_gaussian_filter(blur_sigma)
            # print(gaussian_filter.weight.data) # sums to 1 per channel


    def forward(self, x):
        # First blur if specified
        if self.blur_sigma:
            x = self.gaussian_filter(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        score = F.sigmoid(self.avg(x4)) # value in range [0, 1]
        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        output = F.sigmoid(self.outc(x7)) # mask in range [0, 1]

        return output, score


class MyDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.my_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        return self.my_conv(x)


class MyUp(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.my_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return self.my_conv(x)
