import sys
import os
import numpy as np
from datetime import datetime
from contextlib import contextmanager

class Logger:
    def __init__(self, log_file_path: str):
        """初始化Logger，设置输出到文件和终端。
        
        Args:
            log_file_path (str): 日志文件路径
        """
        self.terminal = sys.stdout
        self.log_dir = os.path.dirname(log_file_path)
        
        # 如果目录不存在则创建
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.log = open(log_file_path, "a", encoding='utf-8')
        
        # 保存原始的NumPy打印选项
        self.original_np_options = {
            'threshold': np.get_printoptions()['threshold'],
            'linewidth': np.get_printoptions()['linewidth'],
            'precision': np.get_printoptions()['precision']
        }
        
        # 写入开始时间
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_initial(f"\n{'='*50}\nLog started at: {start_time}\n{'='*50}\n")

    def _write_initial(self, message):
        """初始化时使用的简单写入方法。"""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def write(self, message):
        """将消息同时写入终端和文件。"""
        # 终端使用默认打印选项
        self.terminal.write(message)
        
        # 检查消息是否需要特殊处理
        if not isinstance(message, str):
            self.log.write(message)
            self.log.flush()
            return
            
        # 文件使用完整打印选项
        with self.full_numpy_precision():
            # 检查是否包含NumPy数组的输出
            if ('array(' in message and '...' in message) or ('ndarray' in message and '...' in message):
                try:
                    # 检查是否为字典格式
                    if '{' in message and '}' in message and ':' in message:
                        # 提取字典部分
                        prefix = message[:message.find('{')]
                        dict_str = message[message.find('{'):message.rfind('}')+1]
                        suffix = message[message.rfind('}')+1:]
                        
                        # 记录原始消息作为备份
                        self.log.write(f"# 原始输出: {message}\n")
                        
                        # 尝试解析但不执行，仅作为参考
                        self.log.write(f"{prefix}字典内容太长，请参考特定对象的完整内容部分{suffix}\n")
                    else:
                        # 如果不是字典格式，尝试提取数组部分
                        if '[' in message and ']' in message and '...' in message:
                            # 记录原始消息作为备份
                            self.log.write(f"# 原始输出: {message}\n")
                            self.log.write("# 数组内容太长，请参考特定对象的完整内容部分\n")
                        else:
                            self.log.write(message)
                except:
                    # 解析失败时使用原始消息
                    self.log.write(message)
            else:
                # 不包含省略号的正常消息
                self.log.write(message)
        
        self.log.flush()

    def flush(self):
        """刷新输出。"""
        self.terminal.flush()
        self.log.flush()

    @contextmanager
    def full_numpy_precision(self):
        """临时设置NumPy打印选项为完整显示。"""
        original_options = np.get_printoptions()
        try:
            np.set_printoptions(
                threshold=np.inf,       # 显示所有元素
                linewidth=np.inf,       # 不换行
                precision=8,            # 保留8位小数
                suppress=True           # 禁止科学计数法
            )
            yield
        finally:
            # 恢复原始设置
            np.set_printoptions(**original_options)

def setup_logger(log_file_name: str = "output.txt"):
    """设置日志记录器。
    
    Args:
        log_file_name (str): 日志文件名，默认为output.txt
        
    Returns:
        原始的stdout，用于恢复输出
    """
    # 创建logs目录（如果不存在）
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 使用时间戳创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"{timestamp}_{log_file_name}")
    
    # 保存原始的stdout
    original_stdout = sys.stdout
    
    # 设置新的Logger
    sys.stdout = Logger(log_file_path)
    
    return original_stdout

def restore_stdout(original_stdout):
    """恢复原始的stdout输出。
    
    Args:
        original_stdout: 原始的stdout对象
    """
    if hasattr(sys.stdout, "log"):
        sys.stdout.log.close()
    sys.stdout = original_stdout 