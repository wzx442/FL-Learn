import torch
import torch.nn as nn
from typing import OrderedDict, Tuple, Optional, Any
from tqdm import tqdm
from pflopt.optimizers import MaskLocalAltSGD, local_alt

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