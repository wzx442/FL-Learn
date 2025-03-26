import argparse


def lth_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",                type=float, default=0.05,    help="Learning rate 学习率")
    parser.add_argument("--batch_size",        type=int,   default=60,      help="Batch size 批量大小")
    parser.add_argument("--lth_epoch_iters",   type=int,   default=3,       help="LTH epoch iterations 本地交替优化迭代次数")
    parser.add_argument("--dataset",           type=str,   default="cifar10", help="dataset name 数据集名称")
    # parser.add_argument("--arch_type",default="resnet18",type=str,help="architecture type 架构类型")
    parser.add_argument( "--setting",          type=str,   default="",      help="setting name 设置名称" )
    parser.add_argument("--prune_percent",     type=float, default=25,      help="Pruning percent 剪枝百分比")
    parser.add_argument("--prune_target",      type=int,   default=80,      help="Pruning target 剪枝目标")
    parser.add_argument( "--com_rounds",       type=int,   default=1,       help="rounds of fedavg training 联邦平均训练轮数")
    parser.add_argument("--la_epochs",         type=int,   default=10,      help="rounds of training for local alt optimization 本地交替优化训练轮数")
    parser.add_argument("--iid",               action="store_true",         help="whether i.i.d or not 是否独立同分布")
                                        # 当使用 action="store_true" 时： 如果命令行中包含这个参数，它的值会被设置为 True
    parser.add_argument("--num_users",         type=int,   default=10,      help="number of users: K 用户数量")
    parser.add_argument("--shard_per_user",    type=int,   default=2,       help="classes per user 每个用户的类别数量")
    parser.add_argument("--local_bs",          type=int,   default=32,      help="local batch size: B 本地批量大小")
    parser.add_argument("--frac",              type=float, default=0.1,     help="the fraction of clients: C 参与训练的客户端比例")
    parser.add_argument("--num_classes",       type=int,   default=10,      help="number of classes 类别数量")
    parser.add_argument("--model",             type=str,   default="resnet18", help="model name 模型名称")
    parser.add_argument("--bs",                type=int,   default=128,     help="test batch size 测试批量大小")
    parser.add_argument("--lth_freq",          type=int,   default=1,       help="frequency of lth 本地交替优化频率")
    parser.add_argument("--pretrained_init",   action="store_true",         help="pretrained initialization 预训练初始化")
    parser.add_argument("--clipgradnorm",      action="store_true",         help="clip gradient norm 梯度裁剪")
    parser.add_argument("--num_samples",       type=int,   default=-1,      help="number of samples 样本数量")
    parser.add_argument("--test_size",         type=int,   default=-1,      help="test size 测试大小")
    parser.add_argument("--exp_name",          type=str,   default="prune_rate_vary", help="experiment name 实验名称")
    parser.add_argument("--server_data_ratio", type=float, default=0.0,     help="The percentage of data that servers also have across data of all clients. 服务器拥有的数据占所有客户端数据的比例")
    parser.add_argument("--seed",              type=int,   default=1,       help="random seed (default: 1)")

    args = parser.parse_args()
    return args
