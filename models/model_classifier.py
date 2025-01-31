import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint as cp

from typing import Any, cast, Dict, List, Optional, Union
import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple

import config


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(config.n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

import random
import numpy as np
#Reproducibility
seed = 42;
random.seed(seed);
np.random.seed(seed);
torch.manual_seed(seed);
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed);
torch.backends.cudnn.deterministic = True;
torch.backends.cudnn.benchmark = False;
###########################################

class ACDNetV2(nn.Module):
    def __init__(self, input_length, n_class, sr, ch_conf=None):
        super(ACDNetV2, self).__init__();
        self.input_length = input_length;
        self.ch_config = ch_conf;

        stride1 = 2;
        stride2 = 2;
        channels = 8;
        k_size = (3, 3);
        n_frames = (sr/1000)*10; #No of frames per 10ms

        sfeb_pool_size = int(n_frames/(stride1*stride2));
        # tfeb_pool_size = (2,2);
        if self.ch_config is None:
            self.ch_config = [channels, channels*8, channels*4, channels*8, channels*8, channels*16, channels*16, channels*32, channels*32, channels*64, channels*64, n_class];
        # avg_pool_kernel_size = (1,4) if self.ch_config[1] < 64 else (2,4);
        fcn_no_of_inputs = self.ch_config[-1];
        conv1, bn1 = self.make_layers(1, self.ch_config[0], (1, 9), (1, stride1));
        conv2, bn2 = self.make_layers(self.ch_config[0], self.ch_config[1], (1, 5), (1, stride2));
        conv3, bn3 = self.make_layers(1, self.ch_config[2], k_size, padding=1);
        conv4, bn4 = self.make_layers(self.ch_config[2], self.ch_config[3], k_size, padding=1);
        conv5, bn5 = self.make_layers(self.ch_config[3], self.ch_config[4], k_size, padding=1);
        conv6, bn6 = self.make_layers(self.ch_config[4], self.ch_config[5], k_size, padding=1);
        conv7, bn7 = self.make_layers(self.ch_config[5], self.ch_config[6], k_size, padding=1);
        conv8, bn8 = self.make_layers(self.ch_config[6], self.ch_config[7], k_size, padding=1);
        conv9, bn9 = self.make_layers(self.ch_config[7], self.ch_config[8], k_size, padding=1);
        conv10, bn10 = self.make_layers(self.ch_config[8], self.ch_config[9], k_size, padding=1);
        conv11, bn11 = self.make_layers(self.ch_config[9], self.ch_config[10], k_size, padding=1);
        conv12, bn12 = self.make_layers(self.ch_config[10], self.ch_config[11], (1, 1));
        fcn = nn.Linear(fcn_no_of_inputs, n_class);
        nn.init.kaiming_normal_(fcn.weight, nonlinearity='sigmoid') # kaiming with sigoid is equivalent to lecun_normal in keras

        self.sfeb = nn.Sequential(
            #Start: Filter bank
            conv1, bn1, nn.ReLU(),\
            conv2, bn2, nn.ReLU(),\
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        );

        tfeb_modules = [];
        self.tfeb_width = int(((self.input_length / sr)*1000)/10); # 10ms frames of audio length in seconds
        tfeb_pool_sizes = self.get_tfeb_pool_sizes(self.ch_config[1], self.tfeb_width);
        p_index = 0;
        for i in [3,4,6,8,10]:
            tfeb_modules.extend([eval('conv{}'.format(i)), eval('bn{}'.format(i)), nn.ReLU()]);

            if i != 3:
                tfeb_modules.extend([eval('conv{}'.format(i+1)), eval('bn{}'.format(i+1)), nn.ReLU()]);

            h, w = tfeb_pool_sizes[p_index];
            if h>1 or w>1:
                tfeb_modules.append(nn.MaxPool2d(kernel_size = (h,w)));
            p_index += 1;

        tfeb_modules.append(nn.Dropout(0.2));
        tfeb_modules.extend([conv12, bn12, nn.ReLU()]);
        h, w = tfeb_pool_sizes[-1];
        if h>1 or w>1:
            tfeb_modules.append(nn.AvgPool2d(kernel_size = (h,w)));
        tfeb_modules.extend([nn.Flatten(), fcn]);

        self.tfeb = nn.Sequential(*tfeb_modules);

        self.output = nn.Sequential(
            nn.Softmax(dim=1)
        );

    def forward(self, x):
        x = self.sfeb(x);
        #swapaxes
        x = x.permute((0, 2, 1, 3));
        x = self.tfeb(x);
        y = self.output[0](x);
        return y;

    def make_layers(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=0, bias=False):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias);
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); # kaiming with relu is equivalent to he_normal in keras
        bn = nn.BatchNorm2d(out_channels);
        return conv, bn;

    def get_tfeb_pool_sizes(self, con2_ch, width):
        h = self.get_tfeb_pool_size_component(con2_ch);
        w = self.get_tfeb_pool_size_component(width);
        # print(w);
        pool_size = [];
        for  (h1, w1) in zip(h, w):
            pool_size.append((h1, w1));
        return pool_size;

    def get_tfeb_pool_size_component(self, length):
        # print(length);
        c = [];
        index = 1;
        while index <= 6:
            if length >= 2:
                if index == 6:
                    c.append(length);
                else:
                    c.append(2);
                    length = length // 2;
            else:
               c.append(1);

            index += 1;

        return c;

def GetACDNetModel(input_len=30225, nclass=50, sr=20000, channel_config=None):
    net = ACDNetV2(input_len, nclass, sr, ch_conf=channel_config);
    return net;