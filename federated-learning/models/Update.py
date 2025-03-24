#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

# 定义了一个名为DatasetSplit的自定义数据集类，继承自Dataset类。
# 通过使用DatasetSplit类，可以从原始数据集中创建一个子数据集，该子数据集仅包含特定的样本。
# 这在分割数据集用于训练和验证时非常有用，可以根据索引划分数据集并创建相应的训练集和验证集。
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# 定义了一个名为LocalUpdate的类，用于在本地进行模型的训练和更。
# 在train方法中，通过迭代数据加载器的批次，对模型进行前向传播、计算损失、反向传播和参数更新，最终返回模型的状态字典和训练周期的平均损失
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        # 保存传入的参数args，用于配置训练过程中的超参数
        self.args = args
        # 保存一个交叉熵损失函数的实例，用于计算训练过程中的损失
        self.loss_func = nn.CrossEntropyLoss()
        # 用于保存选择的客户端
        self.selected_clients = []
        # 创建一个数据加载器DataLoader，加载一个子数据集DatasetSplit，其中子数据集由参数dataset和idxs指定，设置批量大小为self.args.local_bs，并进行随机洗牌
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        # 将模型设置为训练模式
        net.train()
        # train and update
        # 创建一个torch.optim.SGD的优化器，使用net.parameters()作为优化器的参数，设置学习率为self.args.lr和动量为self.args.momentum
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # 用于保存每个训练周期的损失
        epoch_loss = []
        for iter in range(self.args.local_ep):
            # 用于保存每个批次的损失
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # 清零模型参数的梯度
                net.zero_grad()
                # 通过模型进行前向传播，获取预测的对数概率
                log_probs = net(images)
                # 使用损失函数计算损失
                loss = self.loss_func(log_probs, labels)
                # 对损失进行反向传播和参数更新
                loss.backward()
                optimizer.step()
                # 批次索引能被10整除，打印当前训练进度和损失
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                # 计算每个训练周期的平均损失，并将其添加到epoch_loss中
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # 返回模型的状态字典和所有训练周期的平均损失
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

