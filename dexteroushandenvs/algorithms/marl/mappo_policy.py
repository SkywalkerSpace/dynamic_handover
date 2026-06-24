"""
# @Time    : 2021/7/1 6:53 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : rMAPPOPolicy.py
"""

import torch
from algorithms.marl.actor_critic import Actor, Critic
from utils.util import update_linear_schedule


class MAPPO_Policy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, config, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = config["lr"]
        self.critic_lr = config["critic_lr"]
        self.opti_eps = config["opti_eps"]
        self.weight_decay = config["weight_decay"]

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # 实例化 Actor 网络（策略网络）和 Critic 网络（价值网络）
        self.actor = Actor(config, self.obs_space, self.act_space, self.device)
        self.critic = Critic(config, self.share_obs_space, self.device)
        print(self.actor)
        print(self.critic)
        # 初始化 Actor 和 Critic 网络对应的 Adam 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr, eps=self.opti_eps,
            weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        # 对 Actor 和 Critic 的学习率执行线性衰减以促进后期收敛
        update_linear_schedule(self.actor_optimizer,
                               episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer,
                               episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        # Actor 网络向前传播，生成动作 (actions)、动作对数概率 (action_log_probs) 以及新的 RNN 状态
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic)

        # Critic 网络向前传播，根据集中式/全局观察输入，评估当前状态下的价值预测 (values)
        values, rnn_states_critic = self.critic(
            cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        # 仅通过 Critic 价值网络估算状态价值（用于 GAE 返回值的估计）
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        # 计算在当前更新步中，已采取动作的对数概率 (action_log_probs) 以及策略分布熵 (dist_entropy)
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions,
            active_masks)

        # 获取对应的状态价值估计
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        # 只生成动作，不生成价值预测（在测试/评估流程中被 Runner.eval() 调用）
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
