# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


def get_AgentIndex(config):
    agent_index = []
    # 获取右手对应的智能体索引映射
    agent_index.append(eval(config["env"]["handAgentIndex"]))
    # 获取左手对应的智能体索引映射
    agent_index.append(eval(config["env"]["handAgentIndex"]))

    return agent_index


def process_MultiAgentRL(args, env, config, model_dir=""):

    # 设定并行采样(Rollout)线程数与评估线程数均等于矢量化环境中的环境总数 env.num_envs
    config["n_rollout_threads"] = env.num_envs
    config["n_eval_rollout_threads"] = env.num_envs

    if args.algo in ["mappo", ]:
        # on policy marl
        from algorithms.marl.runner import Runner
        # 实例化 MAPPO 算法对应的执行管理器 Runner
        marl = Runner(
            vec_env=env,
            config=config,
            model_dir=model_dir
        )

    return marl
