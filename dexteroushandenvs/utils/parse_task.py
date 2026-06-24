# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from tasks.allegro_hand_dynamic_handover import AllegroHandDynamicHandover

from tasks.hand_base.multi_vec_task_allegro import MultiVecTaskPythonAllegro
from utils.config import warn_task_name


def parse_task(args, cfg, cfg_train, sim_params, agent_index):

    # 获取执行的设备信息
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    if args.task_type == "MultiAgent":
        print("Task type: MultiAgent")

        try:
            # 实例化具体的 Isaac Gym 多智能体物理仿真任务：双灵巧手动态交接
            task = AllegroHandDynamicHandover(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=True)
        except NameError as e:
            print(e)
            warn_task_name()
        # 将底层任务类用 MultiVecTaskPythonAllegro 包装成支持 PyTorch 矢量化数据交互的强化学习环境
        env = MultiVecTaskPythonAllegro(task, rl_device)

        return task, env
    else:
        print(
            "Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo,ippo,maddpg,sac,td3,trpo,ppo,ddpg]"
        )
