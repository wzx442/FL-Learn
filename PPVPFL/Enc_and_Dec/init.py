import numpy as np
from typing import Dict, Tuple
import secrets
import sys

# 初始化函数


# 为每个客户端生成一个随机整数序列 A_i ，A_i 的长度为组数 num_group



# 定义一个 init_seed_pair 函数，输入为 num_users，生成 客户端i 和客户端j 之间的种子对 s_{i,j}
# 种子对 s_{i,j} 是伪随机函数PRG的输入，其中 s_{i,j} 是客户端i 和客户端j 之间的种子对，且 s_{i,j} = s_{j,i}
def init_seed_pair(num_users: int, seed_length: int = 32) -> Dict[Tuple[int, int], int]:
    """为每对客户端生成共享的随机种子。
    
    Args:
        num_users (int): 客户端总数
        seed_length (int): 种子长度(比特数), 默认32位
        
    Returns:
        Dict[Tuple[int, int], int]: 种子对字典，键为客户端对(i,j)，值为它们共享的种子
        
    注意:
        - 种子对是对称的, 即s_{i,j} = s_{j,i}
        - 使用密码学安全的随机数生成器
        - 种子长度默认32位, 可根据安全需求调整
    """
    # 初始化种子对字典
    seed_pairs: Dict[Tuple[int, int], int] = {}
    
    # 为每对客户端生成共享种子
    for i in range(num_users):
        for j in range(i + 1, num_users):
            # 使用密码学安全的随机数生成器
            seed = secrets.randbits(seed_length)
            # 存储种子对，保持对称性
            seed_pairs[(i, j)] = seed
            seed_pairs[(j, i)] = seed
            
    return seed_pairs

# 为每个客户端生成一个随机整数序列 A_i ，A_i 的长度为组数 num_group
def init_A(num_users: int, num_groups: int) -> Dict[int, np.ndarray]:
    """为每个客户端生成一个随机正整数序列 A_i.
    
    Args:
        num_users (int): 客户端总数
        num_groups (int): 组数
        
    Returns:
        Dict[int, np.ndarray]: 每个客户端的随机正整数序列字典，键为客户端索引，值为随机正整数序列
        
    Note:
        - 生成的是1到max_val之间的随机正整数
        - 确保序列中的数字不重复
        - 使用较小的范围以避免数值过大
    """
    A_dict: Dict[int, np.ndarray] = {}
    # 设置一个合理的最大值，比如使用组数的10倍，确保有足够的不重复数字可选
    max_val = max(num_groups * 10, 1000)
    
    for i in range(num_users):
        # 生成1到max_val之间的num_groups个不重复的随机正整数
        A_dict[i] = np.random.choice(np.arange(1, max_val + 1), 
                                   size=num_groups, 
                                   replace=False)
    
    # 尝试将完整的A_dict写入日志文件
    try:
        import os
        if hasattr(sys.stdout, "log") and hasattr(sys.stdout.log, "write"):
            # 构建格式化的输出
            formatted_output = "A_dict (完整内容):\n{\n"
            for k, v in A_dict.items():
                formatted_output += f"    {k}: {v.tolist()},\n"
            formatted_output += "}\n"
            # 直接写入日志文件
            sys.stdout.log.write(formatted_output)
            sys.stdout.log.flush()
    except Exception as e:
        # 忽略任何错误，确保主程序继续运行
        pass
    
    return A_dict

def init_R(num_users: int, num_groups: int) -> np.ndarray:
    """为所有客户端生成一个共享的随机整数序列 R.
    
    Args:
        num_users (int): 客户端总数
        num_groups (int): 组数

    Returns:
        np.ndarray: 共享的随机整数序列 R
    """
    R = np.random.randint(0, 2**16, size=num_users + num_groups)
    
    # 尝试将完整的R写入日志文件
    try:
        if hasattr(sys.stdout, "log") and hasattr(sys.stdout.log, "write"):
            # 构建格式化的输出
            formatted_output = "R (完整内容):\n"
            formatted_output += f"{R.tolist()}\n"
            # 直接写入日志文件
            sys.stdout.log.write(formatted_output)
            sys.stdout.log.flush()
    except Exception as e:
        # 忽略任何错误，确保主程序继续运行
        pass
    
    return R
