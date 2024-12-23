import os
os.environ["CUDA_VISIBLE_DEVICES"] ="4"
import torch
from torchvision import transforms
from torchvision import models
# from my_dataset import MyDataSet
# from utils import read_split_data, plot_data_loader_image
from torch import nn
import torch.optim as optim
from torch import nn
import os
import json
import pickle
import random
import math
import matplotlib.pyplot as plt
import torch
import numpy as np
# from channel.my_channel import AWGN_channel, RayleighChannel

from torch.nn import functional as F
from torchvision import models
# model = models.resnet50(pretrained=True)
 # 提取fc层中固定的参数
# fc_features = model.fc.in_features  # in_features是全连接层的输入层的意思
# # 修改类别为9
# model.fc = nn.Linear(fc_features, 5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class lidar_conv(nn.Module):
    def __init__(self):
        super(lidar_conv, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dense_1 = nn.Linear(288000, 160*160*1)
    def forward(self, x):
          # 先转成了[batch_size, features]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        print('x_conv_shape:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.dense_1(x)
        return x

def main():
    x = torch.randn(2, 3, 300, 60)
   # print(x)
    model = lidar_conv()
    x = x.to(device)
    model =model.to(device)
    x = model(x)
    print('x.shape:', x.shape) #[2,5]
    print(x, x.dtype)  #torch.float32，一个数用32个bit表示，


if __name__=='__main__':
    main()






