# FedSelect: 基于参数自定义选择的个性化联邦学习

<p>
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.7+-1f425f.svg?color=purple">
    </a>
    <a href="https://copyright.illinois.edu/">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
</p>

这是 CVPR 2024 论文 "FedSelect: Personalized Federated Learning with Customized Selection of Parameters for Fine-Tuning" 的官方代码仓库。

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

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/[username]/fedselect.git
cd fedselect
```

2. 安装依赖：
```bash
pip install torch numpy tqdm
```

## 使用方法

1. 准备数据集：
   - 项目支持标准数据集（如 CIFAR-10、MNIST 等）
   - 数据集会在首次运行时自动下载

2. 运行训练：
```bash
python fedselect.py --dataset cifar10 --num_users 100 --epochs 200
```

主要参数说明：
- `--dataset`: 选择数据集
- `--num_users`: 联邦学习中的客户端数量
- `--epochs`: 训练轮数
- `--local_ep`: 每轮本地训练的轮数
- `--local_bs`: 本地训练的批次大小

## 引用

如果您使用了本项目的代码，请引用我们的论文：

```bibtex
@misc{tamirisa2024fedselectpersonalizedfederatedlearning,
      title={FedSelect: Personalized Federated Learning with Customized Selection of Parameters for Fine-Tuning}, 
      author={Rishub Tamirisa and Chulin Xie and Wenxuan Bao and Andy Zhou and Ron Arel and Aviv Shamsian},
      year={2024},
      eprint={2404.02478},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.02478}, 
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 运行记录

1. 小样本测试

```bash
python fedselect.py --dataset cifar10 --num_users 10 --frac 1.0
```

- 结果
> Client Accs:  tensor([0.9569, 0.7776, 0.8855, 0.9643, 0.9463, 0.8544, 0.9668, 0.9145, 0.9203, 0.9114])  | Mean:  tensor(0.9098)
