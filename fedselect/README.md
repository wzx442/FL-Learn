# FedSelect: 基于参数自定义选择的个性化联邦学习


## 项目简介

FedSelect 是一个创新的个性化联邦学习框架，它允许客户端自定义选择需要微调的模型参数。该方法通过让每个客户端独立决定哪些参数需要本地化训练，哪些参数保持全局共享，从而实现更好的个性化效果。

## 文件结构说明

- `fedselect.py`: 主要实现文件，包含核心的联邦学习训练逻辑和评估函数
- `broadcast.py`: 实现服务器与客户端之间的权重广播和聚合功能
- `lottery_ticket.py`: 实现彩票假说相关的模型剪枝和掩码初始化功能
- `pflopt/`: 优化器相关实现目录
  - 包含自定义优化器如 MaskLocalAltSGD
- `utils/`: 工具函数目录
  - 包含数据加载、参数解析等辅助功能

## 环境要求

- Python 3.7+
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

1. 小样本测试

```bash
python fedselect.py --dataset cifar10 --num_users 10 --frac 1.0
```

- 结果
> Client Accs:  tensor([0.9569, 0.7776, 0.8855, 0.9643, 0.9463, 0.8544, 0.9668, 0.9145, 0.9203, 0.9114])  | Mean:  tensor(0.9098)

2. 修改好预训练
```bash
python fedselect.py --dataset cifar10 --num_users 5
```
- 结果
> Client Accs:  tensor([0.7142])  | Mean:  tensor(0.7142)
> 
> Client Accs:  tensor([0.6339])  | Mean:  tensor(0.6339)
> 
> Client Accs:  tensor([0.8003])  | Mean:  tensor(0.8003)
> 
> ('1: 1280', '0: 3840', 'fc.weight', 0.75)
> 
> Mask Sparsity: 72.46%
> 
> Client Accs:  tensor([0.8153])  | Mean:  tensor(0.8153)
> 
> 每一个输出是一次联邦学习的平均准确率，--com_rounds 默认是4

3. sf
```bash
python fedselect.py --dataset cifar10 --num_users 10 --lr 0.01
```

- 结果
> 
> Client Accs:  tensor([0.9330])  | Mean:  tensor(0.9330)
> 
> Client Accs:  tensor([0.9600])  | Mean:  tensor(0.9600)
> 
> Client Accs:  tensor([0.9448])  | Mean:  tensor(0.9448)
> 
> ('1: 1280', '0: 3840', 'fc.weight', 0.75)
> 
> Mask Sparsity: 24.99%
> 
> Client Accs:  tensor([0.9408])  | Mean:  tensor(0.9408)
> 
4. 使用凯明初始化并且加载预训练权重
```bash
python fedselect.py --dataset cifar10 --num_users 10 --lr 0.01
```

- 结果

> Client Accs:  tensor([0.9404])  | Mean:  tensor(0.9404)
> 
> Client Accs:  tensor([0.9688])  | Mean:  tensor(0.9688)
> 
> Client Accs:  tensor([0.9535])  | Mean:  tensor(0.9535)
> 
> ('1: 1280', '0: 3840', 'fc.weight', 0.75)
> 
> Mask Sparsity: 24.99%
> 
> Client Accs:  tensor([0.9756])  | Mean:  tensor(0.9756)
