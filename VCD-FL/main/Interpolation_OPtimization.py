import torch
class Interpolation_OPtimization:
    def __init__(self, args):
        self.args = args

    def Interpolation_OPtimization(self, gradients, G_i, p):
        # 更新G_i: G_i = g_i + 0.5 * G_i
        for name in gradients:
            G_i[name] = gradients[name] + 0.5 * G_i[name]

        for name in gradients:
                # 将G_i展平并获取阈值
                flattened = G_i[name].flatten()
                k = int(len(flattened) * p)
                threshold = torch.kthvalue(flattened.abs(), len(flattened)-k)[0]
                
                # 创建掩码标识大于阈值的元素
                mask = G_i[name].abs() >= threshold
                
                # 用G_i中的值替换梯度中的对应元素
                gradients[name][mask] = G_i[name][mask]
                
                # 从G_i中减去被选中的元素
                G_i[name][mask] = 0
        return gradients
