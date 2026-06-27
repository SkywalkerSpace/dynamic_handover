import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from algorithms.marl.utils.separated_buffer import SeparatedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner:

    def __init__(self,
                 vec_env,
                 config,
                 model_dir=""
                 ):
        self.envs = vec_env
        self.eval_envs = vec_env
        # parameters
        self.env_name = vec_env.task.cfg["env"]["env_name"]
        self.algorithm_name = config["algorithm_name"]
        self.experiment_name = config["experiment_name"]
        self.use_centralized_V = config["use_centralized_V"]
        self.use_obs_instead_of_state = config["use_obs_instead_of_state"]
        self.num_env_steps = config["num_env_steps"]
        self.episode_length = config["episode_length"]
        self.n_rollout_threads = config["n_rollout_threads"]
        self.n_eval_rollout_threads = config["n_eval_rollout_threads"]
        self.use_linear_lr_decay = config["use_linear_lr_decay"]
        self.hidden_size = config["hidden_size"]
        self.use_render = config["use_render"]
        self.recurrent_N = config["recurrent_N"]
        self.use_single_network = config["use_single_network"]
        # interval
        self.save_interval = config["save_interval"]
        self.use_eval = config["use_eval"]
        self.eval_interval = config["eval_interval"]
        self.eval_episodes = config["eval_episodes"]
        self.log_interval = config["log_interval"]

        self.seed = self.envs.task.cfg["seed"]
        self.model_dir = model_dir

        self.num_agents = self.envs.num_agents
        self.device = self.envs.rl_device
        config["device"] = self.envs.rl_device

        torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir + '/' + self.env_name + '/' +
                           self.algorithm_name + '/logs_seed{}'.format(self.seed))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir + '/' + self.env_name + '/' +
                            self.algorithm_name + '/models_seed{}'.format(self.seed))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.algorithm_name == "mappo":
            from algorithms.marl.mappo_trainer import MAPPO as TrainAlgo
            from algorithms.marl.mappo_policy import MAPPO_Policy as Policy

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[
                agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # policy network
            po = Policy(config,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device=self.device)
            self.policy.append(po)

        if self.model_dir != "":
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(config, self.policy[agent_id], device=self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[
                agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(config,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        # 初始化/重置环境及 replay buffer，为收集第一步样本做准备
        self.warmup()

        start = time.time()
        episodes = int(
            self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = torch.zeros(
            1, self.n_rollout_threads, device=self.device)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            done_episodes_rewards = []

            for step in range(self.episode_length):
                # Sample actions
                # 从策略网络中前向传播，采样当前步的动作、动作概率对数以及Critic的状态价值估计
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(
                    step)

                # Obser reward and next obs
                # 步进仿真环境，获取下一步的观测值、全局共享观测值、奖励、终止信号和其它环境信息
                obs, share_obs, rewards, dones, infos, _ = self.envs.step(
                    actions)

                dones_env = torch.all(dones, dim=1)

                reward_env = torch.mean(rewards, dim=1).flatten()

                train_episode_rewards += reward_env

                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(
                            train_episode_rewards[:, t].clone())
                        train_episode_rewards[:, t] = 0

                data = obs, share_obs, rewards, dones, infos, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic

                # insert data into buffer
                # 将当前步采集到的多智能体轨迹数据，分别插入到对应的智能体 replay buffer 中
                self.insert(data)

            # compute return and update network
            # 探索期满一个 Episode 后，利用最后一帧的预测价值，计算 GAE (广义优势估计) 和 returns
            self.compute()
            # 运行多智能体 PPO 更新算法，更新 Actor 和 Critic 神经网络权重
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * \
                self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\nAlgo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

            if len(done_episodes_rewards) != 0:
                aver_episode_rewards = torch.stack(
                    done_episodes_rewards).mean().cpu().numpy().tolist()
                print("some episodes done, average rewards: ",
                      aver_episode_rewards)
                self.writter.add_scalar("train_episode_rewards", aver_episode_rewards,
                                        total_num_steps)
                self.writter.add_scalar("train_episode_fps", int(total_num_steps / (end - start)),
                                        total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0].copy_(share_obs[:, agent_id])
            self.buffer[agent_id].obs[0].copy_(obs[:, agent_id])

    @torch.no_grad()
    def collect(self, step):
        # 收集当前步(step)的决策数据
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        for agent_id in range(self.num_agents):
            # 将该智能体的策略网络设置为 Rollout 评估模式
            self.trainer[agent_id].prep_rollout()
            # 获取该智能体的动作、动作对数概率、状态价值估计以及下一时刻的 RNN/Critic RNN 状态
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(
                    self.buffer[agent_id].share_obs[step],
                    self.buffer[agent_id].obs[step],
                    self.buffer[agent_id].rnn_states[step],
                    self.buffer[agent_id].rnn_states_critic[step],
                    self.buffer[agent_id].masks[step])
            value_collector.append(value.detach())
            action_collector.append(action.detach())
            action_log_prob_collector.append(action_log_prob.detach())
            rnn_state_collector.append(rnn_state.detach())
            rnn_state_critic_collector.append(rnn_state_critic.detach())

        # 整理维度排列：[线程数envs, 智能体数agents, 维度dim]
        values = torch.transpose(torch.stack(value_collector), 1, 0)
        # TODO:
        # actions = torch.transpose(torch.stack(action_collector), 1, 0)
        # action_log_probs = torch.transpose(torch.stack(action_log_prob_collector), 1, 0)
        rnn_states = torch.transpose(torch.stack(rnn_state_collector), 1, 0)
        rnn_states_critic = torch.transpose(
            torch.stack(rnn_state_critic_collector), 1, 0)

        return values, action_collector, action_log_prob_collector, rnn_states, rnn_states_critic

    def insert(self, data):
        # 将当步采样得到的一组多智能体数据解包
        obs, share_obs, rewards, dones, infos, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # 判断整条环境(所有智能体)是否均结束，若结束则需要重置 RNN 隐藏状态
        dones_env = torch.all(dones, axis=1)

        rnn_states[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size, device=self.device)
        rnn_states_critic[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:], device=self.device)

        # 构建 Mask：若环境已结束，对应的 Mask 清零以在下步重置 RNN 状态
        masks = torch.ones(self.n_rollout_threads,
                           self.num_agents, 1, device=self.device)
        masks[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, 1, device=self.device)

        active_masks = torch.ones(
            self.n_rollout_threads, self.num_agents, 1, device=self.device)
        active_masks[dones == True] = torch.zeros(
            (dones == True).sum(), 1, device=self.device)
        active_masks[dones_env == True] = torch.ones(
            (dones_env == True).sum(), self.num_agents, 1, device=self.device)

        if not self.use_centralized_V:
            share_obs = obs

        # 分别把智能体的数据存入其对应的 SeparatedReplayBuffer 中
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(
                share_obs[:, agent_id], obs[:,
                                            agent_id], rnn_states[:, agent_id],
                rnn_states_critic[:,
                                  agent_id], actions[agent_id],
                action_log_probs[agent_id],
                values[:, agent_id], rewards[:,
                                             agent_id], masks[:, agent_id], None,
                active_masks[:, agent_id], None)

    def train(self):
        # 训练并更新网络的主方法
        train_infos = []
        # 随机乱序更新各个智能体的策略网络，避免特定智能体网络更新顺序带来的偏见

        action_dim = 1
        factor = torch.ones(
            self.episode_length, self.n_rollout_threads, action_dim, device=self.device)

        for agent_id in torch.randperm(self.num_agents):
            action_dim = self.buffer[agent_id].actions.shape[-1]

            # 切换为训练(training)模式，激活像 Dropout / BatchNorm 等层
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

            # 评估当前策略下已采样动作的 log-probability，供计算重要性采样权重
            if self.algorithm_name == "hatrpo":
                old_actions_logprob, _, _, _, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                    self.buffer[agent_id].obs[:-
                                              1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                    self.buffer[agent_id].rnn_states[0:1].reshape(
                        -1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]),
                    self.buffer[agent_id].masks[:-1].reshape(
                        -1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                old_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                    self.buffer[agent_id].obs[:-
                                              1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                    self.buffer[agent_id].rnn_states[0:1].reshape(
                        -1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]),
                    self.buffer[agent_id].masks[:-1].reshape(
                        -1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            # 执行单步 PPO/MAPPO 算法更新核心逻辑并返回损失/梯度范数等监控信息
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            if self.algorithm_name == "hatrpo":
                new_actions_logprob, _, _, _, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                    self.buffer[agent_id].obs[:-
                                              1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                    self.buffer[agent_id].rnn_states[0:1].reshape(
                        -1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]),
                    self.buffer[agent_id].masks[:-1].reshape(
                        -1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                new_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                    self.buffer[agent_id].obs[:-
                                              1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                    self.buffer[agent_id].rnn_states[0:1].reshape(
                        -1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]),
                    self.buffer[agent_id].masks[:-1].reshape(
                        -1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            action_prod = torch.exp((new_actions_logprob.detach()-old_actions_logprob.detach()).reshape(
                self.episode_length, self.n_rollout_threads, action_dim).sum(dim=-1, keepdim=True))
            factor = factor*action_prod.detach()
            train_infos.append(train_info)
            # 完成一次网络更新后，清理本轮 Buffer，将最后一帧数据移至首位，用作下个 Epoch/Episode 的初始观测
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self, episode):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                os.makedirs(str(self.save_dir) +
                            "/{}".format(episode), exist_ok=True)
                torch.save(policy_model.state_dict(), str(
                    self.save_dir) + "/{}/model_agent".format(episode) + str(agent_id) + ".pt")
            else:
                os.makedirs(str(self.save_dir) +
                            "/{}".format(episode), exist_ok=True)
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), str(
                    self.save_dir) + "/{}/actor_agent".format(episode) + str(agent_id) + ".pt")
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(
                    self.save_dir) + "/{}/critic_agent".format(episode) + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(
                    str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].model.load_state_dict(
                    policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(
                    self.model_dir) + '/actor_agent' + str(agent_id) + '.pt', map_location=self.device)
                self.policy[agent_id].actor.load_state_dict(
                    policy_actor_state_dict)
                policy_critic_state_dict = torch.load(str(
                    self.model_dir) + '/critic_agent' + str(agent_id) + '.pt', map_location=self.device)
                self.policy[agent_id].critic.load_state_dict(
                    policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalar(agent_k, v, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            self.writter.add_scalars(k, {k: torch.mean(v)}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        # 测试与评估流程
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])

        # 重置评估环境，获取初始观测值
        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        eval_rnn_states = torch.zeros(
            self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size,
            device=self.device)
        eval_masks = torch.ones(
            self.n_eval_rollout_threads, self.num_agents, 1, device=self.device)

        while True:
            eval_actions_collector = []
            eval_rnn_states_collector = []
            for agent_id in range(self.num_agents):
                # 切换对应智能体的网络状态为 Rollout (评估/采样) 模式
                self.trainer[agent_id].prep_rollout()
                # 动作推断：采用确定性动作输出 (deterministic=True)
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        deterministic=True)
                eval_rnn_states[:, agent_id] = temp_rnn_state
                eval_actions_collector.append(eval_actions)

            eval_actions = eval_actions_collector

            # 步进评估环境
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = self.eval_envs.step(
                eval_actions)

            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = torch.all(eval_dones, dim=1)

            # 当环境结束时，重置对应的 RNN 状态和 Mask
            eval_rnn_states[eval_dones_env == True] = torch.zeros(
                (eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size, device=self.device)

            eval_masks = torch.ones(
                self.n_eval_rollout_threads, self.num_agents, 1, device=self.device)
            eval_masks[eval_dones_env == True] = torch.zeros(
                (eval_dones_env == True).sum(), self.num_agents, 1, device=self.device)

            # 累加已完成 Episode 的奖励
            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(
                        torch.sum(torch.cat(one_episode_rewards[eval_i]), dim=0))
                    one_episode_rewards[eval_i] = []

            try:
                print('eval_episode: ', eval_episode,
                      'eval_average_episode_rewards', torch.mean(
                          torch.cat(eval_episode_rewards, dim=-1)).item(),
                      )
            except:
                pass

            # 达到指定的评估 Episode 数量后，计算指标并退出
            if eval_episode >= self.eval_episodes:
                eval_episode_rewards = torch.cat(eval_episode_rewards, dim=-1)
                eval_env_infos = {
                    'eval_average_episode_rewards': torch.mean(eval_episode_rewards),
                    'eval_max_episode_rewards': torch.max(eval_episode_rewards)}
                print(eval_env_infos)
                self.log_env(eval_env_infos, total_num_steps)
                print("eval_average_episode_rewards is {}.".format(
                    torch.mean(eval_episode_rewards)))
                break

    @torch.no_grad()
    def compute(self):
        # 计算每个智能体的 GAE (广义优势估计)
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            # 获取最后一个观测状态 (step=episode_length) 的 Critic 价值预测值，用作计算折扣回报的基底
            next_value = self.trainer[agent_id].policy.get_values(
                self.buffer[agent_id].share_obs[-1],
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1])
            next_value = next_value.detach()
            # 计算优势值 (Advantages) 以及 Target Returns (用于更新 Critic 价值网络)
            self.buffer[agent_id].compute_returns(
                next_value, self.trainer[agent_id].value_normalizer)
