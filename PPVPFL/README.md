# PPVPFL


## 项目简介

PPVPFL 基于 fedselect 和 VCD-FL

## 文件结构说明

- `fedselect.py`: 主要实现文件，包含核心的联邦学习训练逻辑和评估函数
- `broadcast.py`: 实现服务器与客户端之间的权重广播和聚合功能
- `lottery_ticket.py`: 实现彩票假说相关的模型剪枝和掩码初始化功能
- `pflopt/`: 优化器相关实现目录
  - 包含自定义优化器如 MaskLocalAltSGD
- `utils/`: 工具函数目录
  - 包含数据加载、参数解析等辅助功能

## 环境要求

- Python 3.12+
- PyTorch
- NumPy
- tqdm


## 使用方法

1. 准备数据集：
   - 项目支持标准数据集（如 CIFAR-10、MNIST 等）
   - 数据集会在首次运行时自动下载

2. 运行训练：
```bash
python fedselect.py --dataset cifar10 --num_users 100 --frac 1.0
```

主要参数说明：
- `--dataset`: 选择数据集
- `--num_users`: 联邦学习中的客户端数量
- `--lth_epoch_iters`: 本地交替优化迭代次数
- `--com_rounds`: 联邦学习迭代轮数


## 运行记录

1. 10个客户端，resnet18，cifar10，本地训练15轮

```bash
python fedselect.py --dataset cifar10 --num_users 10 --frac 1.0
```

- 结果
> Client Accs:  tensor([0.9569, 0.7776, 0.8855, 0.9643, 0.9463, 0.8544, 0.9668, 0.9145, 0.9203, 0.9114])  | Mean:  tensor(0.9098)


2. 10个客户端，resnet18，本地训练10轮
```bash
python main.py --dataset cifar10  --frac 1.0 --num_users 10 --model resnet18  
```

- 结果
> Client Accs:  tensor([0.9633, 0.8252, 0.8846, 0.8350, 0.9434, 0.7434, 0.9785, 0.9057, 0.8436, 0.8818])  | Mean:  tensor(0.8805)

3.  10个客户端，cifar10，resnet18，本地训练15轮
```bash
python main.py --dataset cifar10  --frac 1.0 --num_users 10 --model resnet18 --la_epochs 15
```

- 结果
> Client Accs:  tensor([0.9707, 0.7997, 0.9736, 0.5495, 0.9609, 0.8936, 0.9785, 0.8191, 0.9086, 0.9521])  | Mean:  tensor(0.8806)