#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 该函数用于对给定的测试数据集进行模型评估。它通过迭代数据加载器，对每个批量的数据进行前向传播和损失计算，然后累加损失和正确分类的样本数。最后计算平均测试损失和准确率，并将其返回.

# net_g表示要测试的模型，datatest表示测试数据集，args表示其他参数。在函数开头，将net_g设置为评估模式
def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    # 计算测试损失和正确分类的样本数
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    # 对数据加载器进行迭代，每次迭代获取一个批量的数据和对应的目标标签
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        # 调用net_g模型对数据进行前向传播
        log_probs = net_g(data)
        # sum up batch loss
        # 使用交叉熵损失函数F.cross_entropy计算损失并累加到test_loss中
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        # 利用预测的对数概率计算预测的类别，并与目标标签进行比较，统计正确分类的样本数
        y_pred = log_probs.data.max(1, keepdim=True)[1]  # keep dimention
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()    # .eq()对比预测值与真实标签,并返回 True/False 张量 ； .long().cpu().sum() 转换为整数，并移动到cpu进行累加。
    
    # 计算平均测试损失和准确率
    test_loss /= len(data_loader.dataset) # 总损失 / 总样本数
    accuracy = 100.00 * correct / len(data_loader.dataset)
    # 是否打印详细的测试结果
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

