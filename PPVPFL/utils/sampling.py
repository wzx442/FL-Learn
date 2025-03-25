import random
import numpy as np
import torch
from typing import Dict, List, Set, Union, Tuple
from torch.utils.data import Dataset


def iid(dataset: Dataset, num_users: int) -> Dict[int, Set[int]]:
    """Sample I.I.D. client data from dataset by randomly dividing into equal parts.

    Args:
        dataset: The full dataset to sample from
        num_users: Number of clients to divide data between

    Returns:
        Dict mapping client IDs to sets of data indices assigned to that client

    中文说明:
    通过随机平均划分数据集来采样独立同分布(I.I.D.)的客户端数据。

    参数:
        dataset: 要采样的完整数据集
        num_users: 要将数据分配给的客户端数量
    
    返回:
        返回一个字典,将客户端ID映射到分配给该客户端的数据索引集合
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid(
    dataset: Dataset,
    num_users: int,
    shard_per_user: int,
    server_data_ratio: float = 0.0,
    size: Union[int, None] = None,
    rand_set_all: List = [],
) -> Tuple[Dict[Union[int, str], Union[np.ndarray, Set[int]]], np.ndarray]:
    """Sample non-I.I.D client data from dataset by dividing data by class labels.

    Args:
        dataset: The full dataset to sample from
        num_users: Number of clients to divide data between
        shard_per_user: Number of class shards to assign to each user
        server_data_ratio: Fraction of data to reserve for server (default: 0.0)
        size: Optional size to limit each user's data to
        rand_set_all: Optional pre-defined random class assignments

    Returns:
        Tuple containing:
            - Dict mapping client IDs to arrays of assigned data indices
            - Array of random class assignments used for the split

    中文说明:
    通过按类别标签划分数据来采样非独立同分布(non-I.I.D)的客户端数据。

    参数:
        dataset: 要采样的完整数据集
        num_users: 要将数据分配给的客户端数量
        shard_per_user: 分配给每个用户的类别分片数量
        server_data_ratio: 为服务器保留的数据比例(默认: 0.0)
        size: 可选参数,用于限制每个用户的数据大小
        rand_set_all: 可选的预定义随机类别分配

    返回:
        返回一个元组,包含:
            - 将客户端ID映射到分配的数据索引数组的字典
            - 用于数据划分的随机类别分配数组
    """
    dict_users, all_idxs = {i: np.array([], dtype="int64") for i in range(num_users)}, [
        i for i in range(len(dataset))
    ]

    targets = None
    # targets = [elem[1].item() for elem in dataset] 旧代码
    # 新代码:###################################
    targets = []
    for elem in dataset:
        if isinstance(elem[1], torch.Tensor):
            targets.append(elem[1].item())
        else:
            targets.append(elem[1])
    # 新代码###########################


    # dictionary of indices in the dataset for each label
    # 字典,键为类别标签,值为该类别标签在数据集中的索引列表
    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    # 为每个用户分配数据
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(targets)[value])
        assert (len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert len(test) == len(dataset)
    assert len(set(list(test))) == len(dataset)

    if server_data_ratio > 0.0:
        dict_users["server"] = set(
            np.random.choice(
                all_idxs, int(len(dataset) * server_data_ratio), replace=False
            )
        )

    for i in range(num_users):
        num_elem = len(dict_users[i])
        dict_users[i] = np.concatenate(
            [
                dict_users[i][k : k + size]
                for k in range(0, num_elem, num_elem // shard_per_user + 1)
            ]
        )

    return dict_users, rand_set_all
