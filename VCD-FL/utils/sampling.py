#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # 计算每个客户端应该获得的图像数量，即数据集总大小除以客户端数量
    num_items = int(len(dataset)/num_users)
    # 用于保存生成的字典，键为客户端的标识符，值为图像的索引集合
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # 从all_idxs中无重复地随机选择num_items个索引，将其作为当前客户端的图像索引集合，并将其添加到dict_users中
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # 从all_idxs中移除已分配给当前客户端的索引集合
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 将数据集分成多少个shard（分片），每个shard应该包含多少个图像
    num_shards, num_imgs = 600, 100
    # 初始化为包含0到num_shards减1的索引列表
    idx_shard = [i for i in range(num_shards)]
    # 用于保存生成的字典，键为客户端的标识符，值为图像的索引数组
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # 初始化为包含0到(num_shards * num_imgs) - 1的索引数组
    idxs = np.arange(num_shards*num_imgs)
    # 获取数据集的标签并转换为NumPy数组
    labels = dataset.train_labels.numpy()

    # sort labels
    # 将idxs和labels按列堆叠为二维数组idxs_labels
    idxs_labels = np.vstack((idxs, labels))
    # 排序，以保证相同标签的图像在一起
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # 更新idxs为排序后的索引数组
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        # 随机选择2个shard的索引，将其作为当前客户端的shard集合，并将其从idx_shard中移除
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        # 对于每个选择的shard索引
        for rand in rand_set:
            # 将对应shard中的图像索引范围添加到当前客户端的索引数组中
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
