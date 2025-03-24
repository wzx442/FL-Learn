#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')   # 使用无 gui 后端，避免绘图时弹出窗口
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid       # 用于数据划分
from utils.options import args_parser                               # 解析命令行参数
from models.Update import LocalUpdate                               # 本地更新
from models.Nets import MLP, CNNMnist, CNNCifar                     # 定义神经网络
from models.Fed import FedAvg       
from models.test import test_img                                    # 测试模型性能


if __name__ == '__main__':
    # parse args
    args = args_parser()            # 解析命令行参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')   # 自动选择 gpu(如果有)

    # load dataset and split users
    if args.dataset == 'mnist':
        # 定义了一系列数据转换操作，并将其组合成一个转换管道。其中包括将图像转换为张量（transforms.ToTensor()）和进行归一化操作（transforms.Normalize()）。这些转换操作将应用于MNIST数据集。
        # transforms.Compose()是一个组合多个数据转换操作的函数，将两个数据转换操作transforms.ToTensor()和transforms.Normalize()组合在一起，形成一个转换管道trans_mnist。
        # transforms.ToTensor()是一个数据转换操作，它将图像数据转换为张量格式
        # transforms.Normalize()是另一个数据转换操作，用于数据归一化。它通过减去均值并除以标准差的方式对图像数据进行归一化，通过指定(0.1307,)和(0.3081,)作为均值和标准差，对MNIST图像进行归一化操作。
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # 加载MNIST训练集，并设置了数据的存储路径、是否下载以及应用的转换操作。
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        # 加载MNIST测试集
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        # sample users
        # 如果args.iid为True，表示采用独立同分布（IID）的方式划分用户，调用了mnist_iid函数来生成用户字典dict_users。mnist_iid函数接受MNIST训练集和用户数量作为参数，返回一个用户字典，其中包含了每个用户的数据。
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        # 表示采用非独立同分布（Non-IID）的方式划分用户：
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    # 也是同理，Normalize中的三个0.5分别对应图像的三个通道（红色、绿色、蓝色），通过减去0.5并除以0.5的方式将像素值范围缩放到-1到1之间，以提高模型训练的效果
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        # cifar-10 只支持 IID 设置
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    # 获取训练集中第一个样本的图像大小，并将其赋值给变量img_size，用于MLP结构匹配。
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # **核心部分**：实现了联邦学习的训练过程。它首先将全局模型的权重复制到每个客户端进行局部训练，然后根据一定的策略聚合客户端的权重，更新全局模型，并打印每轮训练的平均损失值。
    # copy weights  复制当前全局模型net_glob的权重
    w_glob = net_glob.state_dict()

    # training
    # 训练过程中的损失函数列表
    loss_train = []
    # 存储交叉验证的损失和准确率列表
    cv_loss, cv_acc = [], []
    # 存储上一次迭代的验证集损失值和计数器。这些变量通常用于早停策略，在验证集损失不再下降时停止训练，以防止过拟合
    val_loss_pre, counter = 0, 0
    # 存储表现最好的模型和最佳模型对应的验证集损失值
    net_best = None
    best_loss = None
    # 用于存储验证集准确率和模型权重，常用于跟踪验证集上的性能变化和保存模型的快照
    val_acc_list, net_list = [], []

    # 如果为真，表示要对所有客户端进行聚合
    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        # 存储每个客户端的局部损失值
        loss_locals = []
        # 如果不是所有客户，创建列表w_locals存储每个客户端的局部权重
        if not args.all_clients:
            w_locals = []
        # 根据命令行参数args.frac和args.num_users，确定参与本轮训练的客户端数量m
        m = max(int(args.frac * args.num_users), 1)
        # 随机选择m个客户端的索引，存储在idxs_users中
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # 遍历被选的客户端
        for idx in idxs_users:
            # 执行本地更新
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            # 传入当前的全局模型net_glob的副本，并获取更新后的权重w和局部损失loss
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # 如果是所有客户端，将更新后的权重w赋值给w_locals的对应索引位置
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            # 否则，添加权重w到w_locals列表中
            else:
                w_locals.append(copy.deepcopy(w))   # 最后更新完，w_local的索引还是连续的。
            # 将局部损失loss添加到loss_locals列表中
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        # 使用FedAvg函数对w_locals进行聚合，得到更新后的全局权重w_glob
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        # 将更新后的全局权重w_glob加载到net_glob中，以便在下一轮迭代中使用
        net_glob.load_state_dict(w_glob)

        # print loss
        # 计算本轮训练的平均损失，并添加到loss_train列表中
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)



    #     完成了以下几个任务：
    # 绘制损失函数曲线并保存为图片。
    # 将全局模型设置为评估模式。
    # 在训练集和测试集上对模型进行测试，计算准确率和损失。
    # 打印训练准确率和测试准确率。
    # 通过绘制损失函数曲线和计算准确率，可以评估模型的训练效果和泛化能力，并对模型的性能进行分析和比较。

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

