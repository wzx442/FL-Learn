import matplotlib
matplotlib.use('Agg')   # 使用无 gui 后端，避免绘图时弹出窗口
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid       # 用于数据划分
from utils.options import args_parser                               # 解析命令行参数
from models.Update import LocalUpdate                               # 本地更新
from models.Nets import MLP, CNNMnist, CNNCifar                     # 定义神经网络
from models.Fed import FedAvg       
from models.Fed import Fed_VCD
from models.test import test_img                                    # 测试模型性能
from Initialization import Initialization
from Interpolation_OPtimization import Interpolation_OPtimization   
from Enc_and_Commit import Enc_and_Commit
from AggregatedResultVerification import AggregatedResultVerification

if __name__ == '__main__':
    # parse args
    args = args_parser()            # 解析命令行参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')   # 自动选择 gpu(如果有)


    # 加载数据集
    if args.dataset == 'mnist':
        # 定义了一系列数据转换操作，并将其组合成一个转换管道。其中包括将图像转换为张量（transforms.ToTensor()）和进行归一化操作（transforms.Normalize()）。这些转换操作将应用于MNIST数据集。
        # transforms.Compose()是一个组合多个数据转换操作的函数，将两个数据转换操作transforms.ToTensor()和transforms.Normalize()组合在一起，形成一个转换管道trans_mnist。
        # transforms.ToTensor()是一个数据转换操作，它将图像数据转换为张量格式
        # transforms.Normalize()是另一个数据转换操作，用于数据归一化。它通过减去均值并除以标准差的方式对图像数据进行归一化，通过指定(0.1307,)和(0.3081,)作为均值和标准差，对MNIST图像进行归一化操作。
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # 加载MNIST训练集，并设置了数据的存储路径、是否下载以及应用的转换操作。
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        # 加载MNIST测试集
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        # 样本用户
         # 如果args.iid为True，表示采用独立同分布（IID）的方式划分用户，调用了mnist_iid函数来生成用户字典dict_users。mnist_iid函数接受MNIST训练集和用户数量作为参数，返回一个用户字典，其中包含了每个用户的数据。
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        # cifar-10 只支持 IID 设置
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    # 获取训练集中第一个样本的图像大小，并将其赋值给变量img_size，用于MLP结构匹配。
    img_size = dataset_train[0][0].shape

    # build model   
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train() # 将神经网络设置为训练模式

    # **核心部分**：实现了联邦学习的训练过程。它首先将全局模型的权重复制到每个客户端进行局部训练，然后根据一定的策略聚合客户端的权重，更新全局模型，并打印每轮训练的平均损失值。
    # copy weights  复制当前全局模型net_glob的权重
    w_glob = net_glob.state_dict()


##########################################################################################    
   # 初始化辅助信息
    s_ij, rho, A, Z, U = Initialization(args).initialize()
    Z_idx = 0
    B = {}
    d = args.grad_scale
    M = args.m
    num_groups = (d + M - 1) // M
    num_users = args.num_users
##########################################################################################   
# region 初始化训练所需参数
    # training
    # 训练过程中的损失函数列表
    loss_train = []
    # 存储交叉验证的损失和准确率列表
    cv_loss, cv_acc = [], []
    # 存储上一次迭代的验证集损失值和计数器。这些变量通常用于早停策略，在验证集损失不再下降时停止训练，以防止过拟合
    val_loss_pre, counter = 0, 0
    # 存储表现最好的模型和最佳模型对应的验证集损失值
    net_best = None
    best_loss = None
    # 用于存储验证集准确率和模型权重，常用于跟踪验证集上的性能变化和保存模型的快照
    val_acc_list, net_list = [], []

# endregion

# region 插值优化
    # 为每个客户端初始化G_i序列
    G_i = {}
    for i in range(args.num_users):
        # 为每个客户端创建一个 1 * grad_scale的零序列
        G_i[i] = torch.zeros(1 ,args.grad_scale).to(args.device)
# endregion

    # 如果为真，表示要对所有客户端进行聚合
    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]


    for iter in range(args.epochs):# 联邦学习迭代轮数
        # 创建一个字典来存储所有客户端的承诺
        all_commitments = {}

        # 创建一个字典来存储所有客户端的多项式系数
        all_polynomials = {}
########################################################################################## 
        # 存储每个客户端的局部损失值
        loss_locals = []
        # 如果不是所有客户，创建列表w_locals存储每个客户端的局部权重
        if not args.all_clients:
            w_locals = []
        # 根据命令行参数args.frac和args.num_users，确定参与本轮训练的客户端数量m
        num_users = max(int(args.frac * args.num_users), 1)
        # 随机选择m个客户端的索引，存储在idxs_users中
        idxs_users = np.random.choice(range(args.num_users), num_users, replace=False)

        # 遍历被选的客户端
        for idx in idxs_users:
##########################################################################################
#########解密与验证########################################################################
# region 构造聚合梯度
            # 构造聚合梯度
            aggregated_gradients = [[] for _ in range(d)]  # 预先创建d个空列表
            gradients_idx = 0 # AggregatedResultVerification()
            for k in range(num_groups):  # 遍历每个组
                for m in range(M):  # 遍历每个组中的每个元素
                    if gradients_idx >= d:  # 如果已经达到梯度数量上限，退出循环
                        break
                    
                    if k == 0:  # 第0组 m:0-M-1
                        value = np.polyval(B[k], Z[0][m])
                        aggregated_gradients[gradients_idx].append(value)
                        gradients_idx += 1
                    else:  # 其他组 m:k*M+1-(k+1)*M-1
                        Z_a_idx = k*M + 1 + m
                        value = np.polyval(B[k], Z[0][Z_a_idx])
                        aggregated_gradients[gradients_idx].append(value)
                        gradients_idx += 1

# endregion

# region 构造全局梯度
            # 经过构造聚合梯度得到了全局聚合梯度
            # 将全局梯度平均
            aggregated_gradients = [[value / len(idxs_users) for value in sublist] for sublist in aggregated_gradients]
            # 使用全局聚合梯度来反向传播更新模型
            # 将聚合梯度转换为字典格式，与模型参数名称对应
            aggregated_gradients_dict = {}
            param_idx = 0
            for name, param in net_glob.named_parameters():
                if param.requires_grad:
                    # 获取参数的形状
                    param_shape = param.shape
                    # 计算参数的总数
                    param_size = param.numel()
                    # 从aggregated_gradients中获取对应的梯度值并重塑为参数的形状
                    gradient = torch.tensor(aggregated_gradients[param_idx:param_idx+param_size])
                    gradient = gradient.reshape(param_shape)
                    aggregated_gradients_dict[name] = gradient
                    param_idx += param_size
# endregion
            # 使用聚合梯度更新模型参数
            for name, param in net_glob.named_parameters():
                if name in aggregated_gradients_dict:
                    param.data -= args.lr * aggregated_gradients_dict[name]


            # 执行本地更新
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            
            # 传入当前的全局模型net_glob的副本，并获取更新后的权重w、局部损失loss和梯度g
            w, loss, gradients = local.train(net=copy.deepcopy(net_glob).to(args.device))
             
            # 将局部损失loss添加到loss_locals列表中
            loss_locals.append(copy.deepcopy(loss))

            # # update global weights
            # # 使用FedAvg函数对w_locals进行聚合，得到更新后的全局权重w_glob
            # w_glob = FedAvg(w_locals)

            # copy weight to net_glob
            # 将更新后的全局权重w_glob加载到net_glob中，以便在下一轮迭代中使用
            # net_glob.load_state_dict(w_glob)

            # print loss
            # 计算本轮训练的平均损失，并添加到loss_train列表中
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)

            # 在最后一轮打印模型梯度和参数
            if iter == args.epochs - 1:
                print("\n=== Final Round Model Information ===")
                for name, param in net_glob.named_parameters():
                    if param.requires_grad:
                        print(f"\nLayer: {name}")
                        print(f"Parameters:\n{param.data}")
                        print(f"Gradients:\n{param.grad}")
                print("=====================================\n")





            # 对每一层选取G_i中最大的p%元素替换梯度
            p = 0.1  # 选取前10%的元素
            
            # 将梯度展开为行向量
            gradients = gradients.flatten()

            # 使用插值优化方法更新梯度
            gradients = Interpolation_OPtimization(gradients, G_i, p)
##########################################################################################
            # 客户端i经过算法2得到多项式系数和承诺
            polynomials, C_i = Enc_and_Commit(args, gradients, U, s_ij, A[idx], Z, idx)
            all_polynomials[idx] = polynomials
            all_commitments[idx] = C_i

##########################################################################################
##########################################################################################
# region 算法2
#             # 盲化梯度
#             # 使用伪随机数生成器PRG
#             def PRG(seed):
#                 # 设置随机种子
#                 torch.manual_seed(seed)
#                 # 生成与梯度相同形状的随机整数序列
#                 return torch.randint(low=-100, high=100, size=gradients.shape, device=gradients.device)

            
#             blind_gradients = gradients;

#             # 对于当前客户端idx
#             for j in range(args.num_users):
#                 if j < idx:
#                     # 加上PRG(s_ij)
#                     blind_gradients += PRG(s_ij[idx][j])
#                 elif j > idx:
#                     # 减去PRG(s_ij) 
#                     blind_gradients -= PRG(s_ij[idx][j])

#             d = args.grad_scale
#             M = args.m
# ########################################################################################
#             # 计算需要多少组
#             num_groups = (d + M - 1) // M

#             # 创建一个新的张量来存储分组后的梯度
#             # 大小为 num_groups x M，初始化为0
#             blind_gradients_grouped = torch.zeros(num_groups, M, device=blind_gradients.device)

#             # 将原始盲化梯度复制到新张量中
#             # 最后一组如果不足M个元素会自动用0填充
#             # 将blind_gradients分组,每组M个元素
#             for k in range(num_groups-1):
#                 start_idx = k * M
#                 blind_gradients_grouped[k] = blind_gradients[0, start_idx:start_idx+M]

#             # 处理最后一组,可能需要填充0
#             remaining = d - (num_groups-1)*M
#             blind_gradients_grouped[-1, :remaining] = blind_gradients[0, -remaining:]
#             blind_gradients_grouped[-1, remaining:] = 0  # 用0填充剩余位置
#             # 现在blind_gradients_grouped[k]表示第k组盲化梯度
#             # 每组包含M个元素(最后一组可能不足M个,已用0填充)

# ########################################################################################
#             # 将梯度分组
#             # 创建一个新的张量来存储分组后的梯度
#             # 大小为 num_groups x M，初始化为0
#             gradients_grouped = torch.zeros(num_groups, M, device=gradients.device)

#             # 将原始盲化梯度复制到新张量中
#             # 最后一组如果不足M个元素会自动用0填充
#             # 将blind_gradients分组,每组M个元素
#             for k in range(num_groups-1):
#                 start_idx = k * M
#                 gradients_grouped[k] = gradients[0, start_idx:start_idx+M]

#             # 处理最后一组,可能需要填充0
#             remaining = d - (num_groups-1)*M
#             gradients_grouped[-1, :remaining] = gradients[0, -remaining:]
#             gradients_grouped[-1, remaining:] = 0  # 用0填充剩余位置

#             # 现在blind_gradients_grouped[k]表示第k组盲化梯度
#             # 每组包含M个元素(最后一组可能不足M个,已用0填充)

#             ###########################
#             ###                     ###
#             ###现在两个分组都是行向量####
#             ###                     ###
#             ###########################

# #########################################################################################

#             # 计算承诺
#             # 为每个梯度分组计算承诺 C_i[k]
#             # C_i[k] = U · gradients_grouped[k]
#             C_i = torch.zeros(num_groups, M, device=gradients.device)
#             for k in range(num_groups):
#                 # 矩阵乘法: U(M×M) · gradients_grouped[k](M)
#                 C_i[k] = torch.matmul(U, gradients_grouped[k])

#             # 将当前客户端的承诺添加到all_commitments字典中
#             all_commitments[idx] = C_i

# #########################################################################################
#             # 构造拉格朗日多项式
#             # 配对
#             # 为每个客户端i构造拉格朗日插值多项式
#             polynomials = []  # 存储每个客户端i的拉格朗日多项式系数
#             A_i_k = 0
#             Z_idx = 0
#             for k in range(num_groups):
#                 # 准备插值点
#                 x_points = []  # x坐标点
#                 y_points = []  # y坐标点
                
#                 # 添加M个配对点 (Z_(k-1)(M+1)+j, blind_gradients_i((k-1)M+j))
#                 for j in range(M+1):
#                     if j < M:
#                         x_points.append(Z[0][Z_idx])  # Z的索引从k*(M+1)开始
#                         Z_idx += 1
#                         temp_blind_gradients = blind_gradients_grouped[k][j].numpy()
#                         y_points.append(temp_blind_gradients)  # 第k组的第j个元素转化为numpy形式
#                     else:
#                         x_points.append(Z[0][Z_idx])  # Z的最后一个元素
#                         Z_idx += 1
#                         y_points.append(A[idx][0, A_i_k])
#                         A_i_k += 1
                
#                 # 客户端i的第k组就配好了对
#                 # 计算拉格朗日插值多项式f_{i,k}(x)的系数
#                 # 返回一个数组，包含拟合多项式的系数。数组的长度为 M+1，从最高次幂到常数项排列。
#                 coefficients = np.polyfit(x_points, y_points, M)
#                 # 保存客户端i的第k组的系数，已经是降序排列了
#                 polynomials.append(coefficients)



#             # 将当前客户端的拉格朗日多项式添加到all_polynomials字典中
#             all_polynomials[idx] = polynomials
# endregion
##########################################################################################
##########################################################################################
###########密文聚合########################################################################
        # 结束本地训练
        # 开始全局聚合
        # 使用Fed_VCD函数对all_polynomials进行聚合，得到更新后的全局系数
        B = Fed_VCD(all_polynomials, args.num_users, args.num_groups, args.M)

##########################################################################################
##########################################################################################   

            




        

    #     完成了以下几个任务：
    # 绘制损失函数曲线并保存为图片。
    # 将全局模型设置为评估模式。
    # 在训练集和测试集上对模型进行测试，计算准确率和损失。
    # 打印训练准确率和测试准确率。
    # 通过绘制损失函数曲线和计算准确率，可以评估模型的训练效果和泛化能力，并对模型的性能进行分析和比较。

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))