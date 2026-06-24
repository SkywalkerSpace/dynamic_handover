import torch
import torch.nn as nn
from .util import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
对标准 PyTorch 分布接口做一层封装，让动作采样、对数概率和 mode 的返回形状与本代码库保持一致。
"""

#
# Standardize distribution interfaces
# 统一不同分布的接口与张量形状
#

# Categorical


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        # 采样结果补一个最后维度，便于和后续网络输出对齐
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        # 离散动作的 log_prob 需要先去掉多余维度，再汇总到 batch 维
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        # 取最大概率类别作为确定性动作
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        # 连续动作逐维计算 log_prob，保留与原实现兼容的返回形式
        return super().log_prob(actions)
        # return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        # 高斯分布的 mode 就是均值
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        # 二值动作按维汇总 log_prob
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        # 概率大于 0.5 的位置取 1，否则取 0
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        # 先把特征映射到 logits，再把不可用动作屏蔽掉
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


# class DiagGaussian(nn.Module):
#     def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
#         super(DiagGaussian, self).__init__()
#
#         init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
#         def init_(m):
#             return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
#
#         self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
#         self.logstd = AddBias(torch.zeros(num_outputs))
#
#     def forward(self, x, available_actions=None):
#         action_mean = self.fc_mean(x)
#
#         #  An ugly hack for my KFAC implementation.
#         zeros = torch.zeros(action_mean.size())
#         if x.is_cuda:
#             zeros = zeros.cuda()
#
#         action_logstd = self.logstd(zeros)
#         return FixedNormal(action_mean, action_logstd.exp())

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, config=None):
        super(DiagGaussian, self).__init__()
        # 注意这里会优先使用配置中的 actor_gain，确保初始化尺度和策略网络一致
        gain = config["actor_gain"]

        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if config is not None:
            self.std_x_coef = config["std_x_coef"]
            self.std_y_coef = config["std_y_coef"]
        else:
            self.std_x_coef = 1.
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        # 输出动作均值和标准差，供后续策略采样
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(
            self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        # 将特征直接映射为二值动作的 logits
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        # 根据输入维度决定 bias 的广播形状，兼容 MLP 和 CNN
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
