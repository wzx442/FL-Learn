import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, OrderedDict, Tuple, Optional, Any

# Custom Libraries
from utils.options import lth_args_parser
from utils.train_utils import prepare_dataloaders, get_data
from utils.train_functions import evaluate, train_personalized
from pflopt.optimizers import MaskLocalAltSGD, local_alt
from lottery_ticket import init_mask_zeros, delta_update
from broadcast import (
    broadcast_server_to_client_initialization,
    div_server_weights,
    add_masks,
    add_server_weights,
)
import random
from torchvision.models import resnet18
from torchvision import models
from models import load_model as load_model_from_models
from fedselect import fedselect_algorithm, cross_client_eval


def get_cross_correlation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Get cross correlation between two tensors using F.conv2d.
    使用 F.conv2d 获取两个张量之间的交叉相关。
    Args:
        A: First tensor
        B: Second tensor

    Returns:
        torch.Tensor: Cross correlation result
        返回两个张量之间的交叉相关结果
    """
    # Normalize A
    A = A.cuda() if torch.cuda.is_available() else A # 如果GPU可用,将A转换为GPU上的张量,否则保持不变
    B = B.cuda() if torch.cuda.is_available() else B # 如果GPU可用,将B转换为GPU上的张量,否则保持不变
    A = A.unsqueeze(0).unsqueeze(0) # 在A的第0维和第1维添加一个维度
    B = B.unsqueeze(0).unsqueeze(0) # 在B的第0维和第1维添加一个维度
    A = A / (A.max() - A.min()) if A.max() - A.min() != 0 else A # 如果A的最大值和最小值之差不为0,则将A除以最大值和最小值之差,否则保持不变
    B = B / (B.max() - B.min()) if B.max() - B.min() != 0 else B # 如果B的最大值和最小值之差不为0,则将B除以最大值和最小值之差,否则保持不变
    return F.conv2d(A, B) # 返回两个张量之间的交叉相关结果


def run_base_experiment(model: nn.Module, args: Any) -> None:
    """Run base federated learning experiment.
    运行基础的联邦学习实验。
    Args:
        model: Neural network model 神经网络模型
        args: Experiment arguments 实验参数
    """
    dataset_train, dataset_test, dict_users_train, dict_users_test, labels = get_data(args) # 获取数据
    idxs_users = np.arange(args.num_users * args.frac) # 生成用户索引
    m = max(int(args.frac * args.num_users), 1) # 最大用户数量
    idxs_users = np.random.choice(range(args.num_users), m, replace=False) # 随机选择用户
    idxs_users = [int(i) for i in idxs_users] # 将用户索引转换为整数
    fedselect_algorithm(
        model, # 神经网络模型
        args, # 实验参数
        dataset_train, # 训练数据集
        dataset_test, # 测试数据集
        dict_users_train, # 训练数据索引
        dict_users_test, # 测试数据索引
        labels, # 数据标签
        idxs_users, # 用户索引列表
    )


def load_model(args: Any) -> nn.Module:
    """Load and initialize model.
    Args:
        args: Model arguments 模型参数

    Returns:
        nn.Module: Initialized model 初始化后的模型
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    # Load the specified model
    model = load_model_from_models(
        model_name=args.model if hasattr(args, 'model') else 'resnet18',
        num_classes=args.num_classes,
        dataset=args.dataset if hasattr(args, 'dataset') else 'cifar10',
        device=device
    )
    
    return model


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility.设置随机种子以确保可重复性。
    Args:
        seed: Random seed value 随机种子值  
    """
    torch.manual_seed(seed) # 设置手动种子
    torch.cuda.manual_seed_all(seed) # 设置所有GPU的种子
    np.random.seed(seed) # 设置NumPy的种子
    random.seed(seed) # 设置随机种子


if __name__ == "__main__":
    # Argument Parser
    args = lth_args_parser()

    # Set the seed
    setup_seed(args.seed)
    model = load_model(args)

    run_base_experiment(model, args)
