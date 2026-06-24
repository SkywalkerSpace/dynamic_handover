# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from gym import spaces

import torch
import numpy as np


# 多智能体向量化环境封装类，适配为 MARL 算法可用的标准接口
class MultiVecTaskAllegro():
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_environments = task.num_envs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        # self.agent_index = self.task.agent_index
        self.num_agents = 2  # 双手协作场景下的智能体数量

        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = task.device

        print("RL device: ", task.device)

        # 定义每个智能体的观测空间、共享观测空间（用于 CTDE 中的 Critic）和动作空间
        self.obs_space = [spaces.Box(
            low=-np.Inf, high=np.Inf, shape=(450,)) for _ in range(self.num_agents)]
        self.share_observation_space = [
            spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_states,)) for _ in
            range(self.num_agents)]

        self.act_space = tuple(
            [spaces.Box(low=np.ones(self.num_actions) * -clip_actions,
                        high=np.ones(self.num_actions) * clip_actions) for _ in
             range(self.num_agents)])

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_number_of_agents(self):
        return self.num_agents

    def get_env_info(self):
        """返回环境信息字典，包含全局状态维度、单智能体观测维度、动作维度和智能体数量。"""
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.num_agents}
        return env_info

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations


# Python CPU/GPU 多智能体环境的具体实现类，负责观测切分与数据整形
class MultiVecTaskPythonAllegro(MultiVecTaskAllegro):

    def get_state(self):
        # 获取并裁剪全局状态缓冲区
        return torch.clamp(
            self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):
        # 将各智能体单独的动作张量合并为底层物理仿真环境所需的大动作张量
        a_hand_actions = actions[0]
        for i in range(1, len(actions)):
            a_hand_actions = torch.hstack((a_hand_actions, actions[i]))
        actions = a_hand_actions

        # 裁剪动作以避免越界
        actions_tensor = torch.clamp(
            actions, -self.clip_actions, self.clip_actions)

        # 步进底层任务的物理模拟
        self.task.step(actions_tensor)

        # 裁剪并提取局部观察缓冲区，分别属于右手(智能体 0)和左手(智能体 1)
        hand_obs = []
        obs_buf = torch.clamp(
            self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        # 右手观测从对应通道切片重构
        hand_obs.append(
            torch.cat([obs_buf[:, :150], obs_buf[:, 300:450], obs_buf[:, 600:750]], dim=1))
        # 左手观测从对应通道切片重构
        hand_obs.append(torch.cat(
            [obs_buf[:, 150:300], obs_buf[:, 450:600], obs_buf[:, 750:900]], dim=1))
        state_buf = torch.clamp(
            self.task.states_buf, -self.clip_obs, self.clip_obs)

        # 获取环境奖励和终止标志
        rewards = self.task.rew_buf.unsqueeze(-1).to(self.rl_device)
        dones = self.task.reset_buf.to(self.rl_device)

        sub_agent_obs = []
        agent_state = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(2):
            sub_agent_obs.append(hand_obs[i])
            agent_state.append(state_buf)
            sub_agent_reward.append(rewards)
            sub_agent_done.append(dones)
            sub_agent_info.append(torch.Tensor(0))

        # 整理外层维度，转置后以便在 MARL 框架中使用
        obs_all = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)
        reward_all = torch.transpose(torch.stack(sub_agent_reward), 1, 0)
        done_all = torch.transpose(torch.stack(sub_agent_done), 1, 0)
        info_all = torch.stack(sub_agent_info)

        return obs_all, state_all, reward_all, done_all, info_all, None

    def reset(self):
        # 产生一个小的随机探索动作进行环境重置的初始步进
        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.num_actions *
                          self.num_agents], dtype=torch.float32, device=self.rl_device))

        # 步进物理模拟器
        self.task.step(actions)

        # 提取重置后两只手的初始局部观测和全局状态
        hand_obs = []
        obs_buf = torch.clamp(
            self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        hand_obs.append(
            torch.cat([obs_buf[:, :150], obs_buf[:, 300:450], obs_buf[:, 600:750]], dim=1))
        hand_obs.append(torch.cat(
            [obs_buf[:, 150:300], obs_buf[:, 450:600], obs_buf[:, 750:900]], dim=1))
        state_buf = torch.clamp(
            self.task.states_buf, -self.clip_obs, self.clip_obs)

        sub_agent_obs = []
        agent_state = []

        for i in range(2):
            sub_agent_obs.append(hand_obs[i])
            agent_state.append(state_buf)

        obs = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)

        return obs, state_all, None

