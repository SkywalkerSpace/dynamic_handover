import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from utils.util import get_shape_from_obs_space


class Actor(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, config, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.config = config
        self._gain = config["gain"]
        self._use_orthogonal = config["use_orthogonal"]
        self._use_policy_active_masks = config["use_policy_active_masks"]
        self._use_naive_recurrent_policy = config["use_naive_recurrent_policy"]
        self._use_recurrent_policy = config["use_recurrent_policy"]
        self._recurrent_N = config["recurrent_N"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        # 根据观测空间的维度，选择使用 CNN 还是 MLP 提取基础特征
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(self.config, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            # 初始化循环神经网络（RNN）层，处理时间序列依赖
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size,
                                self._recurrent_N, self._use_orthogonal)

        # 实例化动作输出层（ACTLayer），根据特征输出具体的动作分布和样本
        self.act = ACTLayer(action_space, self.hidden_size,
                            self._use_orthogonal, self._gain, self.config)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        """
        # 将输入数组转换为 PyTorch Tensor 并转移至对应设备
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # 提取局部观察的基本网络特征
        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            # 经过 RNN 层，更新隐状态并输出时序特征
            actor_features, rnn_states = self.rnn(
                actor_features, rnn_states, masks)

        # ACTLayer 根据特征决策具体动作和对数概率
        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # 提取基本特征
        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            # 通过 RNN 层
            actor_features, rnn_states = self.rnn(
                actor_features, rnn_states, masks)

        # 分别对 HAPPO/HATRPO 或普通 PPO/MAPPO 计算新动作对数概率和熵值，供 Policy Gradient 参数更新
        if self.config["algorithm_name"] == "hatrpo":
            action_log_probs, dist_entropy, action_mu, action_std, all_probs = self.act.evaluate_actions_trpo(
                actor_features,
                action, available_actions,
                active_masks=active_masks if self._use_policy_active_masks
                else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features,
                action, available_actions,
                active_masks=active_masks if self._use_policy_active_masks
                else None)

            return action_log_probs, dist_entropy


class Critic(nn.Module):
    """
    Critic network class for HAPPO. Outputs value function predictions given centralized input (HAPPO) or local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, config, cent_obs_space, device=torch.device("cuda:0")):
        super(Critic, self).__init__()
        self.hidden_size = config["hidden_size"]
        self._use_orthogonal = config["use_orthogonal"]
        self._use_naive_recurrent_policy = config["use_naive_recurrent_policy"]
        self._use_recurrent_policy = config["use_recurrent_policy"]
        self._recurrent_N = config["recurrent_N"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        # 根据集中式观测空间的维度，选择使用 CNN 还是 MLP 提取特征
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(config, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            # 实例化 Critic 的 RNN 层
            self.rnn = RNNLayer(
                self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=0)

        # 价值输出头：从隐藏状态回归到一个标量状态价值 (state value)
        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # 提取全局共享观测特征
        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            # RNN 层时序前向更新
            critic_features, rnn_states = self.rnn(
                critic_features, rnn_states, masks)
        # 通过线性层回归输出当前状态的预估价值
        values = self.v_out(critic_features)

        return values, rnn_states
