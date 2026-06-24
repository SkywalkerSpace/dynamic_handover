# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_marl import process_MultiAgentRL, get_AgentIndex

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train():
    print("Algorithm: ", args.algo)
    # 获取智能体索引映射（区分左手和右手）
    agent_index = get_AgentIndex(cfg)

    if args.algo in ["mappo", ]:
        # maddpg exists a bug now
        args.task_type = "MultiAgent"
        # 如果指定了预训练模型目录，则进入测试/评估模式
        if args.model_dir != "":
            cfg["is_test"] = True
        else:
            cfg["is_test"] = False

        # 解析并实例化具体的任务(Task)以及环境封装类(VecEnv)
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

        # 实例化多智能体强化学习的训练器 Runner (包含策略网络与ReplayBuffer初始化)
        runner = process_MultiAgentRL(
            args, env=env, config=cfg_train, model_dir=args.model_dir)

        # 测试与训练的分支
        if args.play:
            # 运行模型测试/评估（共评估 1000 个 steps/episodes）
            runner.eval(1000)
        else:
            # 开始运行模型训练主循环
            runner.run()

    else:
        print(
            "Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo,ippo,maddpg,sac,td3,trpo,ppo,ddpg]"
        )


if __name__ == '__main__':
    # 格式化 numpy 数组打印的精度与样式
    set_np_formatting()
    # 解析命令行输入的各项参数
    args = get_args()
    # 从配置文件加载环境(yaml)与训练(yaml)参数
    cfg, cfg_train, logdir = load_cfg(args)
    # 解析 Isaac Gym 仿真器的配置参数
    sim_params = parse_sim_params(args, cfg, cfg_train)
    # 设定全局随机种子，保证实验的可复现性
    set_seed(
        cfg_train.get("seed", -1),
        cfg_train.get("torch_deterministic", False)
    )
    train()
