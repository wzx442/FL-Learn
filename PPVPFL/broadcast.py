import torch
from typing import OrderedDict


def broadcast_server_to_client_initialization( # 这个函数是用来将服务器权重广播到客户端初始化，不需要改
    server_weights: OrderedDict[str, torch.Tensor],
    mask: OrderedDict[str, torch.Tensor],
    client_initialization: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Broadcasts server weights to client initialization for non-masked parameters. 将服务器 **权重** 广播到客户端初始化的非掩码参数。

    Args:
        server_weights: Server model state dict 服务器模型状态字典
        mask: Binary mask indicating which parameters are local (1) vs global (0) 二进制掩码,指示哪些参数是本地的(1)或全局的(0)
        client_initialization: Client model state dict to update 要更新的客户端模型状态字典
   
    Returns:
        Updated client model state dict with server weights broadcast to non-masked parameters 返回:更新后的客户端模型状态字典,其中非掩码参数已被服务器权重广播更新
    """
    for key in client_initialization.keys(): # 在第一轮，遍历客户端初始化字典的键
        # only override client_initialization where mask is non-zero 仅在掩码为非零处覆盖client_initialization
        if "weight" in key or "bias" in key:
            client_initialization[key][mask[key] == 0] = server_weights[key][mask[key] == 0] # 简单实现了哈达玛积
    return client_initialization


def div_server_weights( # 这个函数是用来将服务器权重除以掩码值，要改，改为用聚合的多项式系数除以权重
    server_weights: OrderedDict[str, torch.Tensor],
    server_mask: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Divides server weights by mask values where mask is non-zero. 在掩码非零处将服务器权重除以掩码值。
    
    Args:
        server_weights: Server model state dict 服务器模型状态字典
        server_mask: Mask indicating number of contributions to each parameter 指示每个参数贡献数量的掩码

    Returns:
        Server weights normalized by number of contributions 按贡献数量归一化的服务器权重
    """
    for key in server_weights.keys():
        # only divide where server_mask is non-zero  仅在server_mask非零处进行除法    
        if "weight" in key or "bias" in key:
            server_weights[key][server_mask[key] != 0] /= server_mask[key][server_mask[key] != 0] 
    return server_weights


def add_masks( # 这个函数是用来将客户端的掩码累加到服务器掩码字典中，不需要改
    server_dict: OrderedDict[str, torch.Tensor],
    client_dict: OrderedDict[str, torch.Tensor],
    invert: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Accumulates client masks into server mask dictionary. 将客户端掩码累加到服务器掩码字典中。

    Args:
        server_dict: Server mask accumulator 服务器掩码累加器
        client_dict: Client mask to add 要添加的客户端掩码
        invert: Whether to invert client mask before adding 是否在添加前反转客户端掩码

    Returns:
        Updated server mask accumulator 更新后的服务器掩码累加器
    """
    for key in client_dict.keys():
        if "weight" in key or "bias" in key:
            if key not in server_dict.keys():
                server_dict[key] = 1 - client_dict[key] if invert else client_dict[key]
            else:
                server_dict[key] += (
                    (1 - client_dict[key]) if invert else client_dict[key]
                )
    return server_dict


def add_server_weights( # 这个函数是用来将客户端的权重累加到服务器权重中，所以要改这个，让这个函数累加的是我们的多项式系数
    server_weights: OrderedDict[str, torch.Tensor],
    client_weights: OrderedDict[str, torch.Tensor],
    client_mask: OrderedDict[str, torch.Tensor],
    invert: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Accumulates masked client weights into server weights. 将掩码客户端权重累加到服务器权重中。    

    Args:
        server_weights: Server weights accumulator 服务器权重累加器
        client_weights: Client model weights to add 要添加的客户端模型权重
        client_mask: Binary mask indicating which parameters to add 指示要添加哪些参数的二进制掩码
        invert: Whether to invert mask before applying 是否在应用前反转掩码

    Returns:
        Updated server weights accumulator 更新后的服务器权重累加器
    """
    for key in client_weights.keys(): # 按位计算权重
        if "weight" in key or "bias" in key:
            mask = 1 - client_mask[key] if invert else client_mask[key]
            if key not in server_weights.keys():
                server_weights[key] = client_weights[key] * mask
            else:
                server_weights[key] += client_weights[key] * mask
    return server_weights
