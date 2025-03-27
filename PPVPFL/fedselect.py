import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, OrderedDict, Tuple, Optional, Any

# Custom Libraries
from utils.train_utils import prepare_dataloaders
from utils.train_functions import evaluate, train_personalized
from pflopt.optimizers import MaskLocalAltSGD, local_alt
from lottery_ticket import init_mask_zeros, delta_update
from broadcast import (
    broadcast_server_to_client_initialization,
    div_server_weights,
    add_masks,
    add_server_weights,
)

# 自定义库
from Enc_and_Dec.init import init_A, init_R

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
    """Evaluate models across clients. 评估跨客户端的模型。

    Args:
        model: Neural network model 神经网络模型
        client_state_dicts: Client model states 客户端模型状态
        dataset_train: Training dataset 训练数据集
        dataset_test: Test dataset 测试数据集
        dict_users_train: Mapping of users to training data indices 用户到训练数据索引的映射
        dict_users_test: Mapping of users to test data indices 用户到测试数据索引的映射
        args: Evaluation arguments 评估参数
        no_cross: Whether to only evaluate on own data 是否只评估自己的数据

    Returns:
        torch.Tensor: Matrix of cross-client accuracies 跨客户端准确率矩阵
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
    com_rounds = args.com_rounds # 通信轮数(联邦平均训练轮数)
    # initialize server
    client_accuracies = [{i: 0 for i in idxs_users} for _ in range(com_rounds)] # 每个客户端的准确率历史
    client_state_dicts = {i: copy.deepcopy(initial_state_dict) for i in idxs_users} # 每个客户端的模型状态
    client_state_dict_prev = {i: copy.deepcopy(initial_state_dict) for i in idxs_users} # 每个客户端的上一轮模型状态
    client_masks = {i: None for i in idxs_users} # 每个客户端的掩码
    client_masks_prev = {i: init_mask_zeros(model) for i in idxs_users} # 每个客户端的上一轮掩码
    # noinspection PyTypeHints
    server_accumulate_mask = OrderedDict()  # 服务器累加掩码
    # noinspection PyTypeHints
    server_weights = OrderedDict()  # 服务器权重
    lth_iters = args.lth_epoch_iters # 彩票迭代次数
    prune_rate = args.prune_percent / 100 # 剪枝率
    prune_target = args.prune_target / 100 # 剪枝目标
    lottery_ticket_convergence = [] # 彩票收敛历史

    # 系数集合
    B = []


    #####################################################################################################################
    # Begin FL
    #####################################################################################################################
    model_params = 0 # 模型参数数量
    M = args.M # 每个分组的参数数量
    num_group = 0 # 分组数量
    A_dict = [] # 初始化A，用来保存所有客户端的随机整数序列
    flag_A = False # 是否初始化A
    for round_num in range(com_rounds): # 遍历通信轮数
        round_loss = 0 # 本轮损失
        for i in idxs_users: # 遍历每个客户端
            # initialize model
            model.load_state_dict(client_state_dicts[i]) # 加载客户端模型状态
            model_params = sum(p.numel() for p in model.parameters()) # 计算模型参数数量
            # print(f"In round {round_num}, client {i} model params number: {model_params}")

            # 计算分组数量。分组策略：将模型参数分组，每组M个，如果最后一组的参数数量小于M，则用0填充。
            num_group = model_params // M
            if model_params % M != 0:
                num_group += 1
            print(f"In round {round_num}, client {i} model params number: {model_params}, num_group: {num_group}")

            #######################从这里初始化######################
            if round_num == 0 and not flag_A:  # 只需要初始化一次
                # 初始化A，用来保存所有客户端的随机整数序列
                A_dict = init_A(len(idxs_users), num_group) # 初始化A，用来保存所有客户端的随机整数序列
                # 初始化R，所有客户端共享一个R
                R = init_R(len(idxs_users), num_group)
                flag_A = True # 标记A已经初始化
                print(f"A_dict: {A_dict}")
                print(f"R: {R}")

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
            # Send u_i update to server 将u_i更新发送给服务器 在这里改
            if round_num < com_rounds - 1:
                server_accumulate_mask = add_masks(server_accumulate_mask, client_mask)
                # 将客户端模型参数和掩码添加到服务器权重中
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

        # if round_num == 0:
        #     # 计算一个客户端的模型参数数量
        #     model_params = sum(p.numel() for p in model.parameters())

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
            # noinspection PyTypeHints 不检查类型提示
            server_accumulate_mask = OrderedDict() # 服务器累加掩码
            # noinspection PyTypeHints 不检查类型提示
            server_weights = OrderedDict() # 服务器权重

    # print(f"Round {round_num} model params: {model_params}")

    # 计算跨客户端准确率
    cross_client_acc = cross_client_eval(
        model,              # 模型
        client_state_dicts, # 客户端模型状态
        dataset_train,      # 训练数据集
        dataset_test,       # 测试数据集
        dict_users_train,   # 训练数据索引
        dict_users_test,    # 测试数据索引
        args,               # 训练参数
        no_cross=False,     # 是否只评估自己的数据
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