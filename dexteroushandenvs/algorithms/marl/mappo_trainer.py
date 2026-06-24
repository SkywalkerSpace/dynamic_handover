"""
# @Time    : 2021/7/1 6:52 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : r_mappo.py
"""

import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from algorithms.marl.utils.valuenorm import ValueNorm
from algorithms.marl.utils.popart import PopArt
from algorithms.utils.util import check


class MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 config,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = config["clip_param"]
        self.ppo_epoch = config["ppo_epoch"]
        self.num_mini_batch = config["num_mini_batch"]
        self.data_chunk_length = config["data_chunk_length"]
        self.value_loss_coef = config["value_loss_coef"]
        self.entropy_coef = config["entropy_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        self.huber_delta = config["huber_delta"]

        self._use_valuenorm = config["use_valuenorm"]
        self._use_recurrent_policy = config["use_recurrent_policy"]
        self._use_naive_recurrent = config["use_naive_recurrent_policy"]
        self._use_max_grad_norm = config["use_max_grad_norm"]
        self._use_clipped_value_loss = config["use_clipped_value_loss"]
        self._use_huber_loss = config["use_huber_loss"]
        self._use_popart = config["use_popart"]
        self._use_value_active_masks = config["use_value_active_masks"]
        self._use_policy_active_masks = config["use_policy_active_masks"]

        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(
                return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(
                return_batch) - values

        if self._use_popart:
            error_clipped = self.value_normalizer(
                return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / \
                active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, _ = sample

        old_action_log_probs_batch = check(
            old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        # 计算在当前最新策略网络下，采样的动作对应的预测价值、对数概率以及概率分布的熵值
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch)
        # actor update
        # 计算重要性采样权重 ratio = exp(new_log_prob - old_log_prob)
        imp_weights = torch.exp(
            (action_log_probs - old_action_log_probs_batch).sum(dim=-1, keepdim=True))

        # 计算 PPO 优势裁剪项
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ

        # 计算 Actor/Policy 的总损失函数（在有效状态掩码 active_masks 下进行加权）
        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = - \
                torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            # 目标是最小化 policy_loss 并最大化策略的分布熵 (dist_entropy) 以维持探索性
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            # 裁剪梯度范数以保证网络参数平稳更新，避免梯度爆炸
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        # 执行梯度更新步骤优化 Actor 参数
        self.policy.actor_optimizer.step()

        # critic update
        # 计算 Critic 的价值函数 Loss（可以选用 Huber 损失或常规的 MSE 均方误差损失）
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        # 计算 Critic 损失并乘以对应的价值系数
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            # 对 Critic 网络裁剪梯度范数
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        # 执行梯度更新步骤优化 Critic 参数
        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        # 计算当前的优势函数估计 A = Target_Returns - Value_Predictions (考虑是否使用 ValueNorm)
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - \
                self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.clone()
        # advantages_copy[buffer.active_masks[:-1] == 0.0] = torch.nan
        mean_advantages = torch.mean(advantages_copy)
        # std_advantages = torch.std(advantages_copy)
        std_advantages = torch.std(advantages_copy)
        # 对优势估计进行局部归一化以保证 PPO 算法更加平稳地收敛
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        # 进行多个 Epoch 的 PPO 梯度下降循环
        for _ in range(self.ppo_epoch):
            # 将收集的整个交互缓冲区拆分成多个小批量的生成器数据
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                # 传入每一组采样的小批量数据，在 ppo_update 中执行前向反向更新
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
