import torch
import numpy as np

class Enc_and_Commit:
    def __init__(self, args):
        self.args = args

    def Enc_and_Commit(self, gradients, U, s_ij, A_i, Z, idx):
        # 盲化梯度
        # 使用伪随机数生成器PRG
        def PRG(seed):
            # 设置随机种子
            torch.manual_seed(seed)
            # 生成与梯度相同形状的随机整数序列
            return torch.randint(low=-100, high=100, size=gradients.shape, device=gradients.device)

        
        blind_gradients = gradients;

        # 对于当前客户端idx
        for j in range(self.args.num_users):
            if j < idx:
                # 加上PRG(s_ij)
                blind_gradients += PRG(s_ij[idx][j])
            elif j > idx:
                # 减去PRG(s_ij) 
                blind_gradients -= PRG(s_ij[idx][j])

        d = self.args.grad_scale
        M = self.args.m
########################################################################################
        # 计算需要多少组
        num_groups = (d + M - 1) // M

        # 创建一个新的张量来存储分组后的梯度
        # 大小为 num_groups x M，初始化为0
        blind_gradients_grouped = torch.zeros(num_groups, M, device=blind_gradients.device)

        # 将原始盲化梯度复制到新张量中
        # 最后一组如果不足M个元素会自动用0填充
        # 将blind_gradients分组,每组M个元素
        for k in range(num_groups-1):
            start_idx = k * M
            blind_gradients_grouped[k] = blind_gradients[0, start_idx:start_idx+M]

        # 处理最后一组,可能需要填充0
        remaining = d - (num_groups-1)*M
        blind_gradients_grouped[-1, :remaining] = blind_gradients[0, -remaining:]
        blind_gradients_grouped[-1, remaining:] = 0  # 用0填充剩余位置
        # 现在blind_gradients_grouped[k]表示第k组盲化梯度
        # 每组包含M个元素(最后一组可能不足M个,已用0填充)

########################################################################################
        # 将梯度分组
        # 创建一个新的张量来存储分组后的梯度
        # 大小为 num_groups x M，初始化为0
        gradients_grouped = torch.zeros(num_groups, M, device=gradients.device)

        # 将原始盲化梯度复制到新张量中
        # 最后一组如果不足M个元素会自动用0填充
        # 将blind_gradients分组,每组M个元素
        for k in range(num_groups-1):
            start_idx = k * M
            gradients_grouped[k] = gradients[0, start_idx:start_idx+M]

        # 处理最后一组,可能需要填充0
        remaining = d - (num_groups-1)*M
        gradients_grouped[-1, :remaining] = gradients[0, -remaining:]
        gradients_grouped[-1, remaining:] = 0  # 用0填充剩余位置

        # 现在blind_gradients_grouped[k]表示第k组盲化梯度
        # 每组包含M个元素(最后一组可能不足M个,已用0填充)

        ###########################
        ###                     ###
        ###现在两个分组都是行向量####
        ###                     ###
        ###########################
#########################################################################################
        
        # 计算承诺
        # 为每个梯度分组计算承诺 C_i[k]
        # C_i[k] = U · gradients_grouped[k]
        C_i = torch.zeros(num_groups, M, device=gradients.device)
        for k in range(num_groups):
            # 矩阵乘法: U(M×M) · gradients_grouped[k](M)
            C_i[k] = torch.matmul(U, gradients_grouped[k])

        

#########################################################################################
        # 构造拉格朗日多项式
        # 配对
        # 为每个客户端i构造拉格朗日插值多项式
        polynomials = []  # 存储每个客户端i的拉格朗日多项式系数
        A_i_k = 0
        Z_idx = 0
        for k in range(num_groups):
            # 准备插值点
            x_points = []  # x坐标点
            y_points = []  # y坐标点
            
            # 添加M个配对点 (Z_(k-1)(M+1)+j, blind_gradients_i((k-1)M+j))
            for j in range(M+1):
                if j < M:
                    x_points.append(Z[0][Z_idx])  # Z的索引从k*(M+1)开始
                    Z_idx += 1
                    temp_blind_gradients = blind_gradients_grouped[k][j].numpy()
                    y_points.append(temp_blind_gradients)  # 第k组的第j个元素转化为numpy形式
                else:
                    x_points.append(Z[0][Z_idx])  # Z的最后一个元素
                    Z_idx += 1
                    y_points.append(A_i[0, A_i_k])
                    A_i_k += 1
            
            # 客户端i的第k组就配好了对
            # 计算拉格朗日插值多项式f_{i,k}(x)的系数
            # 返回一个数组，包含拟合多项式的系数。数组的长度为 M+1，从最高次幂到常数项排列。
            coefficients = np.polyfit(x_points, y_points, M)
            # 保存客户端i的第k组的系数，已经是降序排列了
            polynomials.append(coefficients)

        return polynomials, C_i

