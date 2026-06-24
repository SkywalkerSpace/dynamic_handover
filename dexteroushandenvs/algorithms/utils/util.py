import copy
import numpy as np

import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    # 统一封装权重和偏置初始化逻辑，便于各网络模块复用
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    # 深拷贝出 N 份相同结构的模块，常用于重复堆叠网络层
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    # 如果输入是 numpy 数组，就转成 torch Tensor
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
