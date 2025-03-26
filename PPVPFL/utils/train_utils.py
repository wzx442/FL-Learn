from torchvision import datasets, transforms
from utils.sampling import iid, noniid
import numpy as np
import torch
from typing import Dict, List, Tuple, Any


class DatasetSplit(torch.utils.data.Dataset):
    """Custom Dataset class that returns a subset of another dataset based on indices. 自定义数据集类，根据索引返回另一个数据集的子集。

    Args:
        dataset: The base dataset to sample from 要从中采样的基础数据集
        idxs: Indices to use for sampling from the base dataset 要从中采样的基础数据集的索引
    """

    def __init__(self, dataset: torch.utils.data.Dataset, idxs: List[int]) -> None:
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[self.idxs[item]]
        return image, label


trans_mnist = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
trans_cifar10_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar10_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar100_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)
trans_cifar100_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)


def get_data(
    args: Any,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, Dict, Dict, np.ndarray]:
    """Get train and test datasets and user splits for federated learning. 获取用于联邦学习的训练和测试数据集以及用户划分。 

    Args:
        args: Arguments containing dataset configuration 包含数据集配置的参数

    Returns:
        dataset_train: Training dataset 训练数据集
        dataset_test: Test dataset 测试数据集
        dict_users_train: Dictionary mapping users to training data indices 字典映射用户到训练数据索引
        dict_users_test: Dictionary mapping users to test data indices 字典映射用户到测试数据索引
        rand_set_all: Random set assignments for non-iid splitting 非独立同分布划分随机集分配
    """
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST("data/mnist", train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST("data/mnist", train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10("data/cifar10", train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10("data/cifar10", train=False, download=True, transform=trans_cifar10_val)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.iid:
        dict_users_train = iid(dataset_train, args.num_users) # 将训练数据集划分为num_users个用户   
        dict_users_test = iid(dataset_test, args.num_users) # 将测试数据集划分为num_users个用户
        rand_set_all = np.array([]) # 随机集分配
    else: # 非独立同分布划分
        dict_users_train, rand_set_all = noniid(
            dataset_train,
            args.num_users,
            args.shard_per_user, # 分配给每个用户的类别分片数量
            args.server_data_ratio, # 为服务器保留的数据比例(默认: 0.0)
            size=args.num_samples, # 可选参数,用于限制每个用户的数据大小
        )
        dict_users_test, rand_set_all = noniid(
            dataset_test,
            args.num_users,
            args.shard_per_user,
            args.server_data_ratio,
            size=args.test_size,
            rand_set_all=rand_set_all, # 可选的预定义随机类别分配
        )

    return dataset_train, dataset_test, dict_users_train, dict_users_test, rand_set_all # 返回训练和测试数据集以及用户划分


def prepare_dataloaders(
    dataset_train: torch.utils.data.Dataset,
    dict_users_train: Dict,
    dataset_test: torch.utils.data.Dataset,
    dict_users_test: Dict,
    args: Any,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Prepare train and test data loaders for a user. 为单个用户准备训练和测试数据加载器。

    Args:
        dataset_train: Training dataset 训练数据集
        dict_users_train: Dictionary mapping users to training data indices 字典映射用户到训练数据索引
        dataset_test: Test dataset 测试数据集
        dict_users_test: Dictionary mapping users to test data indices 字典映射用户到测试数据索引
        args: Arguments containing batch size configuration 包含批量大小配置的参数

    Returns:
        ldr_train: Training data loader 训练数据加载器
        ldr_test: Test data loader 测试数据加载器   
    """
    ldr_train = torch.utils.data.DataLoader(
        DatasetSplit(dataset_train, dict_users_train), # 创建一个数据集分割对象,用于从训练数据集中获取指定用户的数据
        batch_size=args.local_bs, # 本地批量大小
        shuffle=True, # 打乱数据    
    )
    ldr_test = torch.utils.data.DataLoader(
        DatasetSplit(dataset_test, dict_users_test), # 创建一个数据集分割对象,用于从测试数据集中获取指定用户的数据  
        batch_size=args.local_bs, # 本地批量大小
        shuffle=False, # 不打乱数据 
    )
    return ldr_train, ldr_test # 返回训练和测试数据加载器
