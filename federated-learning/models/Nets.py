#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F

# 第一个简单的多层感知机模型，包含一个输入层、一个隐藏层和一个输出层。
# 它使用线性层进行线性变换，ReLU激活函数引入非线性变换，并应用Dropout层以减少过拟合。
# 这个模型可以通过调用 forward 方法来进行前向传播，将输入数据经过网络的层操作得到输出结果。
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

# 定义了一个基于卷积神经网络的模型 CNNMnist，用于处理MNIST数据集。
# 该模型通过两个卷积层提取图像特征，然后通过线性层进行分类。
# ReLU激活函数和最大池化层用于非线性变换和特征降采样，Dropout层用于减少过拟合。
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        # 用于提取图像的特征。args.num_channels 表示输入图像的通道数，32表示输出通道数，kernel_size=5 表示卷积核的大小为5x5
        self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=5)
        # 创建第二个二维卷积层，进一步提取特征
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # 创建一个二维Dropout层，用于随机失活卷积层的输出特征图。
        self.conv2_drop = nn.Dropout2d()
        # 创建一个线性层，将卷积层输出的特征图转换为256维的向量。
        self.fc1 = nn.Linear(64 *4 * 4 , 256)
        # 创建第二个线性层，将卷积层输出的特征图转换为128维的向量。
        self.fc2 = nn.Linear(256 , 128)
        # 创建最后一个线性层，将50维的向量映射到类别数量
        self.fc3 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        # 通过第一个卷积层，并应用ReLU激活函数和最大池化层来提取特征
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 通过第二个卷积层，并应用ReLU激活函数、Dropout和最大池化层来进一步提取特征
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 对特征图进行形状变换，将其展平为一维向量
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        # 通过线性层进行特征到隐藏层的线性变换，并应用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 应用Dropout层，随机失活一部分隐藏层神经元
        x = F.dropout(x, training=self.training)
        # 通过线性层进行特征到隐藏层的线性变换，并应用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 应用Dropout层，随机失活一部分隐藏层神经元
        x = F.dropout(x, training=self.training)
        # 通过线性层进行隐藏层到输出层的线性变换
        x = self.fc3(x)
        return x

# 定义了一个卷积神经网络模型 CNNCifar，用于处理CIFAR数据集。
# 该模型通过两个卷积层提取图像特征，然后通过线性层进行分类。
# ReLU激活函数和最大池化层用于非线性变换和特征降采样。
class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        # 创建一个二维卷积层，用于提取图像的特征。输入通道数为3，输出通道数为6，卷积核大小为5x5。
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 创建一个最大池化层，用于特征降采样
        self.pool = nn.MaxPool2d(2, 2)
        # 创建第二个二维卷积层，进一步提取特征
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 创建一个线性层，将卷积层输出的特征图转换为120维的向量
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 创建一个线性层，将120维的向量映射到84维的向量
        self.fc2 = nn.Linear(120, 84)
        # 创建最后一个线性层，将84维的向量映射到类别数量
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        # 通过第一个卷积层，并应用ReLU激活函数和最大池化层来提取特征
        x = self.pool(F.relu(self.conv1(x)))
        # 通过第二个卷积层，并应用ReLU激活函数和最大池化层来进一步提取特征
        x = self.pool(F.relu(self.conv2(x)))
        # 对特征图进行形状变换，将其展平为一维向量
        x = x.view(-1, 16 * 5 * 5)
        # 通过线性层进行特征到隐藏层的线性变换，并应用ReLU激活函数 
        x = F.relu(self.fc1(x))
        # 通过线性层进行隐藏层到隐藏层的线性变换，并应用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 通过线性层进行隐藏层到输出层的线性变换
        x = self.fc3(x)
        return x
