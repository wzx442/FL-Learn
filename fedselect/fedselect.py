# Importing Libraries
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


def evaluate(
    model: nn.Module, ldr_test: torch.utils.data.DataLoader, args: Any
) -> float:
    """Evaluate model accuracy on test data loader.

    Args:
        model: Neural network model to evaluate
        ldr_test: Test data loader
        args: Arguments containing device info

    中文说明:
    在测试数据集上评估模型准确性。

    参数:
        model: 要评估的神经网络模型
        ldr_test: 测试数据加载器
        args: 包含设备信息的参数

    Returns:
        float: Average accuracy on test set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    average_accuracy = 0
    # 将模型设置为评估模式。在评估模式下,模型的行为会有所不同:
    # 1. 停用 dropout 层
    # 2. 使用批量归一化层的运行时统计数据而不是计算新的统计数据
    # 3. 不会计算梯度,节省内存
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(ldr_test):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(target.view_as(pred)).sum().item() / len(data)
            average_accuracy += acc
        average_accuracy /= len(ldr_test)
    return average_accuracy


def train_personalized(
    model: nn.Module, # 要训练的神经网络模型
    # ldr_train:训练数据加载器    
    ldr_train: torch.utils.data.DataLoader, 
    mask: OrderedDict, # 二进制掩码,用于参数更新
    args: Any, # 训练参数
    initialization: Optional[OrderedDict] = None, # 可选的初始模型状态
    verbose: bool = False, # 是否打印训练进度
    eval: bool = True, # 是否在训练期间进行评估
) -> Tuple[nn.Module, float]:
    """Train model with personalized local alternating optimization. 使用个性化的本地交替优化来训练模型。

    Args:
        model: Neural network model to train               要训练的神经网络模型
        ldr_train: Training data loader                    训练数据加载器
        mask: Binary mask for parameters                   用于参数的二进制掩码
        args: Training arguments                           训练参数
        initialization: Optional initial model state       可选的模型初始状态
        verbose: Whether to print training progress        是否打印训练进度
        eval: Whether to evaluate during training          是否在训练期间进行评估

    Returns:
        Tuple containing:           返回一个元组,包含:
            - Trained model         训练后的模型
            - Final training loss   最终的训练损失
    """
    if initialization is not None:
        model.load_state_dict(initialization)
    optimizer = MaskLocalAltSGD(model.parameters(), mask, lr=args.lr)
    epochs = args.la_epochs # rounds of training for local alt optimization 本地交替优化训练轮数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    train_loss = 0  # 训练损失
    with tqdm(total=epochs) as pbar: # 进度条
        for i in range(epochs): # 遍历本地交替优化训练轮数
            train_loss = local_alt(
                model,
                criterion,
                optimizer,
                ldr_train,
                device,
                clip_grad_norm=args.clipgradnorm, # 梯度裁剪
            )
            if verbose: # 是否打印训练进度
                print(f"Epoch: {i} \tLoss: {train_loss}")
            pbar.update(1) # 更新进度条
            pbar.set_postfix({"Loss": train_loss}) # 设置进度条的损失
    return model, train_loss


def fedselect_algorithm(
    model: nn.Module, # 要训练的神经网络模型
    args: Any, # 训练参数
    dataset_train: torch.utils.data.Dataset, # 训练数据集
    dataset_test: torch.utils.data.Dataset, # 测试数据集
    dict_users_train: Dict[int, np.ndarray], # 训练数据索引
    dict_users_test: Dict[int, np.ndarray], # 测试数据索引
    labels: np.ndarray, # 数据标签
    idxs_users: List[int], # 用户索引列表
) -> Dict[str, Any]:
    """Main FedSelect federated learning algorithm. 主联邦学习算法。

    Args:
        model: Neural network model 神经网络模型
        args: Training arguments 训练参数
        dataset_train: Training dataset 训练数据集
        dataset_test: Test dataset 测试数据集
        dict_users_train: Mapping of users to training data indices 用户到训练数据索引的映射
        dict_users_test: Mapping of users to test data indices 用户到测试数据索引的映射
        labels: Data labels 数据标签
        idxs_users: List of user indices 用户索引列表

    Returns:
        Dict containing:
            - client_accuracies: Accuracy history for each client 每个客户端的准确率历史
            - labels: Data labels 数据标签
            - client_masks: Final client masks 最终的客户端掩码
            - args: Training arguments 训练参数
            - cross_client_acc: Cross-client accuracy matrix 跨客户端准确率矩阵
            - lth_convergence: Lottery ticket convergence history 彩票收敛历史
    """

    # initialize model
    initial_state_dict = copy.deepcopy(model.state_dict())  # 初始模型状态
                                                            # 创建一个对象的完全独立的副本
                                                            # 复制对象及其包含的所有嵌套对象
    com_rounds = args.com_rounds # 通信轮数(联邦平均训练轮数)
    # initialize server
    client_accuracies = [{i: 0 for i in idxs_users} for _ in range(com_rounds)] # 每个客户端的准确率历史
    client_state_dicts = {i: copy.deepcopy(initial_state_dict) for i in idxs_users} # 每个客户端的模型状态
    client_state_dict_prev = {i: copy.deepcopy(initial_state_dict) for i in idxs_users} # 每个客户端的上一轮模型状态
    client_masks = {i: None for i in idxs_users} # 每个客户端的掩码
    client_masks_prev = {i: init_mask_zeros(model) for i in idxs_users} # 每个客户端的上一轮掩码
    server_accumulate_mask = OrderedDict()  # 服务器累加掩码
    server_weights = OrderedDict()  # 服务器权重
    lth_iters = args.lth_epoch_iters # 彩票迭代次数
    prune_rate = args.prune_percent / 100 # 剪枝率
    prune_target = args.prune_target / 100 # 剪枝目标
    lottery_ticket_convergence = [] # 彩票收敛历史


    #####################################################################################################################
    # Begin FL
    #####################################################################################################################
    for round_num in range(com_rounds): # 遍历通信轮数
        round_loss = 0 # 本轮损失
        for i in idxs_users: # 遍历每个客户端
            # initialize model
            model.load_state_dict(client_state_dicts[i]) # 加载客户端模型状态
            # get data
            # ldr_train:训练数据加载器
            ldr_train, _ = prepare_dataloaders(
                dataset_train,
                dict_users_train[i],
                dataset_test,
                dict_users_test[i],
                args,
            )
            # Update LTN_i on local data
            client_mask = client_masks_prev.get(i) # 获取客户端上一轮掩码
            # Update u_i parameters on local data
            # 0s are global parameters, 1s are local parameters 0表示全局参数,1表示本地参数
            client_model, loss = train_personalized(model, ldr_train, client_mask, args)
            round_loss += loss
            # Send u_i update to server
            if round_num < com_rounds - 1:
                server_accumulate_mask = add_masks(server_accumulate_mask, client_mask)
                server_weights = add_server_weights(
                    server_weights, client_model.state_dict(), client_mask
                )
            client_state_dicts[i] = copy.deepcopy(client_model.state_dict())
            client_masks[i] = copy.deepcopy(client_mask)

            if round_num % lth_iters == 0 and round_num != 0:
                client_mask = delta_update(
                    prune_rate, # 剪枝率
                    client_state_dicts[i], # 当前模型状态
                    client_state_dict_prev[i], # 上一轮模型状态
                    client_masks_prev[i], # 上一轮掩码
                    bound=prune_target, # 剪枝上限
                    invert=True, # 反转
                )
                client_state_dict_prev[i] = copy.deepcopy(client_state_dicts[i]) # 更新上一轮模型状态
                client_masks_prev[i] = copy.deepcopy(client_mask) # 更新上一轮掩码
        round_loss /= len(idxs_users) # 计算本轮损失
        cross_client_acc = cross_client_eval(
            model, # 模型
            client_state_dicts, # 当前模型状态
            dataset_train, # 训练数据集
            dataset_test, # 测试数据集
            dict_users_train, # 训练数据索引
            dict_users_test, # 测试数据索引
            args, # 训练参数
        )

        accs = torch.diag(cross_client_acc) # 对角线上的元素
        for i in range(len(accs)):
            client_accuracies[round_num][i] = accs[i] # 更新每个客户端的准确率
        print("Client Accs: ", accs, " | Mean: ", accs.mean()) # 打印每个客户端的准确率

        if round_num < com_rounds - 1:
            # Server averages u_i
            server_weights = div_server_weights(server_weights, server_accumulate_mask)
            # Server broadcasts non lottery ticket parameters u_i to every device
            # 服务器广播非彩票参数 u_i 到每个设备
            for i in idxs_users:
                client_state_dicts[i] = broadcast_server_to_client_initialization(
                    server_weights, # 服务器权重
                    client_masks[i], # 客户端掩码
                    client_state_dicts[i] # 客户端模型状态
                )
            server_accumulate_mask = OrderedDict() # 服务器累加掩码
            server_weights = OrderedDict() # 服务器权重

    # 计算跨客户端准确率
    cross_client_acc = cross_client_eval(
        model, # 模型
        client_state_dicts, # 客户端模型状态
        dataset_train, # 训练数据集
        dataset_test, # 测试数据集
        dict_users_train, # 训练数据索引
        dict_users_test, # 测试数据索引
        args, # 训练参数
        no_cross=False, # 是否只评估自己的数据
    )

    # 输出字典
    out_dict = {
        "client_accuracies": client_accuracies, # 每个客户端的准确率历史
        "labels": labels,                       # 数据标签
        "client_masks": client_masks,           # 每个客户端的掩码
        "args": args,                           # 训练参数
        "cross_client_acc": cross_client_acc,   # 跨客户端准确率矩阵
        "lth_convergence": lottery_ticket_convergence, # 彩票收敛历史
    }

    return out_dict


def cross_client_eval(
    model: nn.Module, # 模型
    client_state_dicts: Dict[int, OrderedDict], # 客户端模型状态
    dataset_train: torch.utils.data.Dataset, # 训练数据集
    dataset_test: torch.utils.data.Dataset, # 测试数据集
    dict_users_train: Dict[int, np.ndarray], # 训练数据索引
    dict_users_test: Dict[int, np.ndarray], # 测试数据索引
    args: Any, # 训练参数
    no_cross: bool = True, # 是否只评估自己的数据
) -> torch.Tensor:
    """Evaluate models across clients.

    Args:
        model: Neural network model
        client_state_dicts: Client model states
        dataset_train: Training dataset
        dataset_test: Test dataset
        dict_users_train: Mapping of users to training data indices
        dict_users_test: Mapping of users to test data indices
        args: Evaluation arguments
        no_cross: Whether to only evaluate on own data

    Returns:
        torch.Tensor: Matrix of cross-client accuracies
    """
    cross_client_acc_matrix = torch.zeros(
        (len(client_state_dicts), len(client_state_dicts))
    )
    idx_users = list(client_state_dicts.keys())
    for _i, i in enumerate(idx_users):
        model.load_state_dict(client_state_dicts[i])
        for _j, j in enumerate(idx_users):
            if no_cross:
                if i != j:
                    continue
            # eval model i on data from client j
            _, ldr_test = prepare_dataloaders(
                dataset_train,
                dict_users_train[j],
                dataset_test,
                dict_users_test[j],
                args,
            )
            acc = evaluate(model, ldr_test, args)
            cross_client_acc_matrix[_i, _j] = acc
    return cross_client_acc_matrix


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
    加载和初始化模型。
    Args:
        args: Model arguments 模型参数

    Returns:
        nn.Module: Initialized model 初始化后的模型
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 如果GPU可用,则使用GPU,否则使用CPU
    args.device = device # 将设备设置为GPU或CPU
    # model = resnet18(pretrained=args.pretrained_init) # 加载预训练的ResNet18模型
    model = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # 加载预训练的ResNet18模型
    num_ftrs = model.fc.in_features # 获取模型的全连接层输入特征数量
    model.fc = nn.Linear(num_ftrs, args.num_classes) # 将模型的全连接层替换为新的线性层
    model = model.to(device) # 将模型移动到GPU或CPU
    return model.to(device) # 返回初始化后的模型


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    设置随机种子以确保可重复性。
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
