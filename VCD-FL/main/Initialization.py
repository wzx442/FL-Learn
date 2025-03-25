import numpy as np
import torch

class Initialization:
    def __init__(self, args):
        self.args = args

    def initialize(self):
        # 初始化辅助信息

        d = self.args.grad_scale  # 使用 grad_scale 作为矩阵维度
        M = self.args.m
        s_ij = {}  # 用于存储客户端对之间的种子的字典
        rho = {}  # 用于存储每个客户端的种子的字典
        A = {}    # 用于存储每个客户端序列的字典
        # 生成全局序列 Z
        np.random.seed(self.args.seed)  # 使用全局种子以保证可重现性
        Z = np.random.randint(0, 2**10-1, size=(1, 2*d))
        # 将数组转换为张量并移动到指定设备
        Z = torch.from_numpy(Z).to(self.args.device)

        # 生成 M*M 不可逆整数方阵 U
        # 创建一个随机整数矩阵
        U = np.random.randint(-10, 10, size=(M, M))
        # 将第二列设置为第一列的2倍，确保矩阵不可逆
        U[:, 1] = 2 * U[:, 0]
        # 转换为张量并移动到指定设备
        U = torch.from_numpy(U).float().to(self.args.device)

        for i in range(self.args.num_users):
            for j in range(i+1, self.args.num_users):
                # Generate unique seed for each client pair
                s_ij[(i,j)] = np.random.randint(0, 2**10-1)
                s_ij[(j,i)] = s_ij[(i,j)]  # Symmetric seeds

            # Generate random seed for client i
            rho[i] = np.random.randint(0, 2**10-1)
            
            
            # Generate random sequence using PRG with seed rho[i]
            np.random.seed(rho[i])
            # Generate 1 x grad_scale random sequence
            prg_output = np.random.randint(-10, 11, size=(1, d))
            
            # Calculate A_i = PRG(rho_i) / max(|PRG(rho_i)|)
            max_abs = np.max(np.abs(prg_output))
            if max_abs > 0:  # Avoid division by zero
                A[i] = prg_output / max_abs
            else:
                A[i] = prg_output
                
            # Convert to tensor and move to device
            A[i] = torch.from_numpy(A[i]).to(self.args.device)
        
        return s_ij, rho, A, Z, U


