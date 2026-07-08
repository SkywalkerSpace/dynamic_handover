# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# =============================================================================
# 文件概述：
#   AllegroHand 双手动态接物（Dynamic Handover）Isaac Gym 仿真环境
#   任务：左手（投手）将物体抛出，右手（接手）在空中接住物体
#   机器人：xArm6 机械臂 + AllegroHand 灵巧手（共两套，左右各一）
#   算法：MAPPO（多智能体近端策略优化）
#   关键特性：
#     - 多物体训练（ball/block/obj0-10/novel_obj系列）
#     - 轨迹预测网络（TrajEstimator）在线预测落点
#     - 接住成功判定：连续稳定持有N帧才算成功
#     - 非对称观测（Asymmetric Obs）用于Critic
#     - TensorBoard 记录成功率曲线
# =============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# 1. 导入模块
# ──────────────────────────────────────────────────────────────────────────────

from matplotlib.pyplot import axis  # matplotlib坐标轴工具（此处实际未使用，可能是遗留import）
import numpy as np                   # 数值计算
import os                            # 文件系统操作
import random                        # 标准库随机数（用于部分可视化扰动）
import torch                         # PyTorch深度学习框架
import pickle                        # Python对象序列化/反序列化

from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志写入器

from utils.torch_jit_utils import *       # 工具函数：quat_apply, quat_mul, scale, unscale 等
# from isaacgym.torch_utils import *      # （已注释）Isaac Gym官方工具函数，被自定义版本替代

from tasks.hand_base.base_task import BaseTask  # 基础任务类，提供gym/sim初始化框架
from isaacgym import gymtorch                   # Isaac Gym PyTorch张量接口
from isaacgym import gymapi                     # Isaac Gym 核心API（资产加载、环境创建等）

import matplotlib.pyplot as plt      # 绘图库（用于调试可视化）
from PIL import Image as Im          # 图像处理（点云调试）
import cv2                           # OpenCV（图像处理辅助）
from torch import nn                 # PyTorch神经网络模块
import torch.nn.functional as F      # PyTorch函数式接口（MSE Loss等）


# =============================================================================
# 2. 轨迹预测网络（TrajEstimator）
# =============================================================================

class TrajEstimator(nn.Module):
    """
    轨迹落点估计网络（在线学习版）
    ─────────────────────────────────────────────────────────────────────────
    功能：
        根据过去 object_seq_len 帧的物体位置序列，预测物体的最终落点（接触位置）。
        本网络在仿真过程中在线训练（非离线预训练），每步都会做一次前向推断和梯度更新。

    结构：
        4层全连接网络：input_dim → 512 → 256 → 128 → output_dim
        激活函数：ELU（比ReLU在负值区域有更平滑的梯度，适合连续控制）

    输入（input_dim）：
        物体位置历史序列，默认 object_seq_len=20 帧 × 3维(x,y,z) = 60维

    输出（output_dim）：
        预测的接触位置，3维 (x, y, z)
        同时返回倒数第二层特征向量（128维），供外部使用
    """

    def __init__(self, input_dim, output_dim):
        """
        参数：
            input_dim  (int): 输入特征维度，默认60（20帧×3坐标）
            output_dim (int): 输出预测维度，默认3（xyz落点）
        """
        super(TrajEstimator, self).__init__()

        # 第1层：输入维度 → 512
        self.linear1 = nn.Linear(input_dim, 512)
        # 第2层：512 → 256
        self.linear2 = nn.Linear(512, 256)
        # 第3层：256 → 128
        self.linear3 = nn.Linear(256, 128)
        # 输出层：128 → output_dim（落点坐标）
        self.output_layer = nn.Linear(128, output_dim)

        # 激活函数：ELU（Exponential Linear Unit），负值区域有指数衰减梯度
        self.activate_func = nn.ELU()

    def forward(self, inputs):
        """
        前向传播

        参数：
            inputs (Tensor): 形状 [num_envs, input_dim]，物体位置历史序列

        返回：
            outputs (Tensor): 形状 [num_envs, output_dim]，预测的落点坐标
            x       (Tensor): 形状 [num_envs, 128]，倒数第二层的隐层特征（pose_latent_vector）
        """
        # 第1层线性变换 + ELU激活
        x = self.activate_func(self.linear1(inputs))
        # 第2层线性变换 + ELU激活
        x = self.activate_func(self.linear2(x))
        # 第3层线性变换 + ELU激活，得到128维隐层特征
        x = self.activate_func(self.linear3(x))
        # 输出层（无激活函数，直接输出回归值）
        outputs = self.output_layer(x)

        # 同时返回预测结果和隐层特征（x可用于downstream任务或分析）
        return outputs, x


# =============================================================================
# 3. 临时梯度上下文管理器（TemporaryGrad）
# =============================================================================

class TemporaryGrad(object):
    """
    临时开启梯度计算的上下文管理器
    ─────────────────────────────────────────────────────────────────────────
    用途：
        Isaac Gym 仿真环境中，大量操作在 torch.no_grad() 模式下运行以节省显存。
        但 TrajEstimator 需要计算梯度来做在线反向传播。
        使用此上下文管理器，可以在局部代码块中强制启用梯度，退出后自动恢复原状态。

    使用示例：
        with TemporaryGrad():
            predict_pose, latent = traj_estimator(object_state_stack_frames)
            loss = F.mse_loss(predict_pose, target)
            loss.backward()   # 此处梯度正常计算
        # 退出后恢复原先的 no_grad 状态
    """

    def __enter__(self):
        """进入上下文：保存当前梯度状态，然后强制开启梯度"""
        self.prev = torch.is_grad_enabled()   # 保存进入前的梯度开关状态
        torch.set_grad_enabled(True)           # 强制开启梯度计算

    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文：恢复进入前的梯度状态"""
        torch.set_grad_enabled(self.prev)      # 恢复原状态（可能是 False）


# =============================================================================
# 4. 主环境类：AllegroHandDynamicHandover
# =============================================================================

class AllegroHandDynamicHandover(BaseTask):
    """
    双手动态接物仿真环境
    ─────────────────────────────────────────────────────────────────────────
    继承自 BaseTask，实现了 Isaac Gym 强化学习环境的完整接口：
        create_sim()           → 创建仿真世界（地面、资产、环境）
        compute_observations() → 计算策略输入的观测量
        compute_reward()       → 计算奖励信号
        pre_physics_step()     → 物理仿真前处理动作（动作缩放、目标写入）
        post_physics_step()    → 物理仿真后更新观测和奖励
        reset()                → 环境 episode 重置

    场景布局：
        - 左手（another_hand / 投手）：xArm6+AllegroHand，负责抓住并抛出物体
        - 右手（hand / 接手）：xArm6+AllegroHand，负责在空中接住物体
        - 物体（object）：从左手被抛出，飞向右手
        - 目标标记（goal object）：半透明球，显示期望的接住位置
        - 预测标记（predict_goal_object）：橙色球，显示TrajEstimator预测的落点

    坐标系：
        - x轴：左右方向
        - y轴：前后方向（负方向为"从左手到右手"方向，距离约1.35m）
        - z轴：垂直向上
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]],
                 is_multi_agent=False):
        """
        环境初始化

        参数：
            cfg           (dict):  YAML配置字典，包含环境参数、奖励权重等
            sim_params    (SimParams): Isaac Gym 仿真参数（dt、重力等）
            physics_engine(PhysicsEngine): 物理引擎选择（PhysX/Flex）
            device_type   (str):   计算设备类型，"cuda" 或 "cpu"
            device_id     (int):   GPU设备编号（多卡时使用）
            headless      (bool):  是否无头模式（不渲染画面，用于服务器训练）
            agent_index   (list):  多智能体索引映射，默认两个智能体各控制6个DOF
            is_multi_agent(bool):  是否启用多智能体模式（MAPPO），True时每个agent输出22维动作
        """
        # ── 4.1 基础配置保存 ──────────────────────────────────────────────────
        self.cfg = cfg                        # 完整YAML配置字典
        self.sim_params = sim_params          # 物理仿真参数（时间步长、重力等）
        self.physics_engine = physics_engine  # 物理引擎（Isaac Gym PhysX/Flex）
        self.agent_index = agent_index        # 多智能体索引：指定每个agent控制哪些DOF

        self.is_multi_agent = is_multi_agent  # 是否多智能体（影响动作空间划分）

        # ── 4.2 域随机化配置 ──────────────────────────────────────────────────
        # 域随机化（Domain Randomization）：在仿真中随机化物理参数，提升Sim2Real迁移能力
        self.randomize = self.cfg["task"]["randomize"]
        # 具体随机化参数（质量抖动范围、摩擦系数范围等）
        self.randomization_params = self.cfg["task"]["randomization_params"]
        # 聚合模式：将环境中多个actor合并管理，提高仿真效率（0=不聚合，1/2=聚合）
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # ── 4.3 奖励函数超参数 ────────────────────────────────────────────────
        # 位置距离奖励系数（目标位置与物体位置之间的距离惩罚权重）
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        # 姿态旋转对齐奖励系数（物体朝向与目标朝向的角度误差权重）
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        # 动作惩罚系数（抑制过大的关节速度/扭矩输出，防止抖动）
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        # 成功判定容差（旋转角度误差在此阈值内视为"方向对齐成功"）
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        # 达到目标时给予的奖励加成（bonus）
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        # 物体掉落判定距离（物体高度低于此值认为已掉落）
        self.fall_dist = self.cfg["env"]["fallDistance"]
        # 物体掉落时的惩罚值（负奖励）
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        # 旋转奖励计算中的数值稳定性epsilon（防止除零）
        self.rot_eps = self.cfg["env"]["rotEps"]

        # ── 4.4 观测量缩放系数 ────────────────────────────────────────────────
        # 关节速度观测的缩放因子（将速度值缩放到合理区间，便于网络学习）
        self.vel_obs_scale = 0.2
        # 力/力矩传感器观测的缩放因子
        self.force_torque_obs_scale = 10.0

        # ── 4.5 重置时的噪声参数 ──────────────────────────────────────────────
        # 物体位置重置时添加的随机噪声幅度（米），增加环境多样性
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        # 物体旋转重置时添加的随机噪声幅度（弧度）
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        # 关节位置重置时的随机扰动范围（弧度）
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        # 关节速度重置时的随机扰动范围（rad/s）
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        # ── 4.6 控制参数 ──────────────────────────────────────────────────────
        # DOF（自由度）运动速度缩放系数（将归一化动作[-1,1]映射到实际关节角速度）
        self.allegro_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        # 是否使用相对控制（True：动作表示关节角度增量；False：直接设置目标角度）
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        # 动作移动平均系数（平滑相邻帧动作，防止关节运动跳变抖动）
        # cur_target = alpha * new_target + (1-alpha) * prev_target
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        # ── 4.7 调试与训练配置 ────────────────────────────────────────────────
        # 是否开启调试可视化（如绘制坐标轴等debug线条）
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # episode最大长度（步数），超过此步数强制reset
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        # 按时间（秒）设置episode长度（-1.0表示不使用，改用max_episode_length步数）
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        # 是否打印成功率统计
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        # 连续成功次数上限（达到后触发reset，用于课程学习）
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        # 连续成功滑动平均系数（EMA平滑，越小越稳定）
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        # 控制频率倒数（每N个物理步执行一次控制动作）
        # 例如 control_freq_inv=4，物理dt=0.01s → 控制频率=25Hz
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            # 根据实际时间和物理步长重新计算episode最大长度
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        # ── 4.8 物体类型与资产路径 ───────────────────────────────────────────
        # 操作物体类型（从配置文件读取）
        self.object_type = self.cfg["env"]["objectType"]
        # 仅允许特定类型，否则报错
        assert self.object_type in ["block", "egg", "pen", "ycb/banana", "ycb/can", "ycb/mug", "ycb/brick"]

        # pen类物体忽略z轴旋转（因为笔的轴对称性，绕长轴旋转无意义）
        self.ignore_z = (self.object_type == "pen")

        # ── 4.8.1 资产路径字典（第一版：标准YCB物体集）
        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",   # 彩色方块
            "ball": "urdf/objects/ball.urdf",                # 球
            "egg": "mjcf/open_ai_assets/hand/egg.xml",       # 鸡蛋（OpenAI资产）
            "pen": "mjcf/open_ai_assets/hand/pen.xml",       # 笔
            "ycb/banana": "urdf/ycb/011_banana/011_banana.urdf",               # YCB香蕉
            "ycb/can": "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf", # YCB罐头
            "ycb/mug": "urdf/ycb/025_mug/025_mug.urdf",                        # YCB杯子
            "ycb/brick": "urdf/ycb/061_foam_brick/061_foam_brick.urdf"          # YCB泡沫砖
        }

        # ── 4.8.2 资产路径字典（第二版：自定义binghao_obj物体集，覆盖第一版）
        # 注意：后面的赋值会覆盖前面的，Python中最后一次赋值生效
        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "obj0": "urdf/binghao_obj/objects/obj0.urdf",    # 自定义物体0
            "obj1": "urdf/binghao_obj/objects/obj1.urdf",
            "obj2": "urdf/binghao_obj/objects/obj2.urdf",
            # "obj3": "urdf/binghao_obj/objects/obj3.urdf",  # obj3已注释（可能有问题）
            "obj4": "urdf/binghao_obj/objects/obj4.urdf",
            # "obj5": "urdf/binghao_obj/objects/obj5.urdf",
            "obj6": "urdf/binghao_obj/objects/obj6.urdf",
            "obj7": "urdf/binghao_obj/objects/obj7.urdf",
            # "obj8": "urdf/binghao_obj/objects/obj8.urdf",
            "obj9": "urdf/binghao_obj/objects/obj9.urdf",
            "obj10": "urdf/binghao_obj/objects/obj10.urdf",
            "ball": "urdf/binghao_obj/objects/ball.urdf",
            "pen": "mjcf/open_ai_assets/hand/pen.xml"
        }

        # ── 4.8.3 资产路径字典（第三版：加入novel_obj新奇物体，最终生效版本）
        # 这是实际使用的版本，包含训练集物体 + 测试泛化用的新奇物体
        self.asset_files_dict = {
            # ---- 标准训练物体 ----
            "block": "urdf/objects/cube_multicolor.urdf",
            "obj0": "urdf/binghao_obj/objects/obj0.urdf",
            "obj1": "urdf/binghao_obj/objects/obj1.urdf",
            "obj2": "urdf/binghao_obj/objects/obj2.urdf",
            "obj4": "urdf/binghao_obj/objects/obj4.urdf",
            "obj6": "urdf/binghao_obj/objects/obj6.urdf",
            "obj7": "urdf/binghao_obj/objects/obj7.urdf",
            "obj9": "urdf/binghao_obj/objects/obj9.urdf",
            "obj10": "urdf/binghao_obj/objects/obj10.urdf",
            "ball": "urdf/binghao_obj/objects/ball.urdf",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",

            # ---- novel物体（新奇物体，用于泛化测试）----
            # "novel_obj0": "urdf/binghao_obj/objects/final_calibration.urdf",  # 标定块（已注释）
            "novel_obj1":  "urdf/binghao_obj/objects/final_obj1.urdf",           # 自定义形状1
            "novel_obj2":  "urdf/binghao_obj/objects/final_obj2_cross.urdf",     # 十字形
            "novel_obj3":  "urdf/binghao_obj/objects/final_obj3_flatcan.urdf",   # 扁平罐
            "novel_obj4":  "urdf/binghao_obj/objects/final_obj4_t.urdf",         # T形
            "novel_obj5":  "urdf/binghao_obj/objects/final_obj5_s.urdf",         # S形
            "novel_obj6":  "urdf/binghao_obj/objects/final_obj6_ball.urdf",      # 球形变体
            "novel_obj7":  "urdf/binghao_obj/objects/final_obj7_ball_flatter.urdf",  # 扁球
            "novel_obj8":  "urdf/binghao_obj/objects/final_obj8_cylinder.urdf",  # 圆柱
            "novel_obj9":  "urdf/binghao_obj/objects/final_obj9_irregular_cube.urdf", # 不规则方块
            "novel_obj10": "urdf/binghao_obj/objects/final_obj10_cube_stair.urdf",    # 楼梯方块
            "novel_obj11": "urdf/binghao_obj/objects/final_obj11_cube_ir2.urdf",      # 不规则方块2
            "novel_obj12": "urdf/binghao_obj/objects/final_obj12_cube_extrude1.urdf", # 挤出方块1
            "novel_obj13": "urdf/binghao_obj/objects/final_obj13_cube_extrude2.urdf", # 挤出方块2
            "novel_obj14": "urdf/binghao_obj/objects/final_obj14_cathead.urdf",       # 猫头形状
        }

        # ── 4.8.4 实际参与训练的物体列表（从asset_files_dict中选取子集）
        # 每个环境会按照索引轮流分配不同物体（i % len(used_training_objects)）
        # self.used_training_objects = ['ball', "block"]  # 简化版（调试用）
        self.used_training_objects = [
            "pen", "ball", "block",
            "obj0", "obj1", "obj2", "obj4", "obj6", "obj7", "obj9", "obj10"
        ]
        # 以下两个版本已注释：
        # （仅新奇物体版）
        # self.used_training_objects = ["novel_obj1", ..., "novel_obj14"]
        # （训练+新奇混合版）
        # self.used_training_objects = ["ball", "obj0", ..., "novel_obj14"]

        # ── 4.9 观测空间配置 ──────────────────────────────────────────────────
        # 观测类型，从配置中读取（"full_state" / "point_cloud" / "point_cloud_for_distill"）
        self.obs_type = self.cfg["env"]["observationType"]
        print("Obs type:", self.obs_type)

        # 点云特征维度（来自编码器，如PointNet）
        self.num_point_cloud_feature_dim = 384
        # 单帧观测维度（full_state模式下每帧300维）
        self.one_frame_num_obs = 300

        # 各观测类型对应的总观测维度（3帧堆叠）
        self.num_obs_dict = {
            # 点云观测：111维状态 + 384×3维点云特征
            "point_cloud":             111 + self.num_point_cloud_feature_dim * 3,
            # 蒸馏用点云观测（与上同）
            "point_cloud_for_distill": 111 + self.num_point_cloud_feature_dim * 3,
            # 全状态观测：300维×3帧历史堆叠
            "full_state":              300 * 3,
        }

        # ── 4.10 接触传感器配置 ────────────────────────────────────────────────
        # AllegroHand接触传感器链接名称（FSR = Force Sensing Resistor 力敏电阻）
        # 11个接触点：食指/中指/无名指各3节 + 拇指1节（最终节指尖）
        self.contact_sensor_names = [
            "link_1.0_fsr",          # 食指 第1节
            "link_2.0_fsr",          # 食指 第2节
            "link_3.0_tip_fsr",      # 食指 指尖
            "link_5.0_fsr",          # 中指 第1节
            "link_6.0_fsr",          # 中指 第2节
            "link_7.0_tip_fsr",      # 中指 指尖
            "link_9.0_fsr",          # 无名指 第1节
            "link_10.0_fsr",         # 无名指 第2节
            "link_11.0_tip_fsr",     # 无名指 指尖
            "link_14.0_fsr",         # 拇指 第2节
            "link_15.0_fsr",         # 拇指 指尖（共11个传感器）
        ]

        # ── 4.11 仿真轴和观测控制 ─────────────────────────────────────────────
        self.up_axis = 'z'          # 世界坐标系中"向上"的轴为z轴

        self.use_vel_obs = False    # 是否在观测中加入关节速度（当前未使用，改用unscale位置）
        self.fingertip_obs = True   # 是否观测指尖位置状态

        # 是否使用非对称观测（Actor观测 ≠ Critic观测）
        # 非对称观测：Critic可以看到额外的特权信息（如物体完整状态），Actor只能看到部分感知
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        # Critic状态维度
        num_states = 0
        if self.asymmetric_obs:
            # num_states = 215 + 384 * 3  # 加点云版本
            num_states = 215               # 当前使用215维特权状态

        # 将观测维度和状态维度写回配置（供BaseTask读取）
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states

        if self.is_multi_agent:
            # 多智能体模式：2个智能体，每个控制22个DOF（6个臂关节+16个手指关节）
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 22
        else:
            # 单智能体模式：1个智能体，控制全部44个DOF（两只手合在一起）
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 44

        # 将设备信息写回配置
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        # ── 4.12 相机与点云配置 ───────────────────────────────────────────────
        # 是否启用相机传感器（用于获取深度图生成点云）
        self.enable_camera_sensors = self.cfg["env"]["enableCameraSensors"]
        # 是否开启相机调试模式（实时显示相机图像）
        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        # 是否开启点云调试可视化（需要open3d库）
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)
        # 总并行环境数量
        self.num_envs = cfg["env"]["numEnvs"]

        if self.point_cloud_debug:
            # 仅在点云调试模式下导入open3d（避免强制依赖）
            import open3d as o3d
            from utils.o3dviewer import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False   # 初始化标志
            self.o3d_pc = o3d.geometry.PointCloud()        # Open3D点云对象
        else:
            self.pointCloudVisualizer = None               # 不使用点云可视化

        # ── 4.13 调用父类初始化（BaseTask.__init__）──────────────────────────
        # BaseTask会根据cfg创建gym实例、设备、缓冲区等基础框架
        super().__init__(cfg=self.cfg)

        # ── 4.14 观察者相机设置 ──────────────────────────────────────────────
        if self.viewer is not None:
            # 设置查看器相机位置和朝向（仅在有头模式下生效）
            # cam_pos    = gymapi.Vec3(0.9, -0.65, 1.0)    # 近处视角（已注释）
            cam_pos    = gymapi.Vec3(4.0, -0.65, 1.0)      # 当前：较远的侧视角
            cam_target = gymapi.Vec3(-0.5, -0.65, 0.2)     # 相机目标点（手部中间位置）
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # ──────────────────────────────────────────────────────────────────────
        # 4.15 从GPU获取仿真状态张量（Isaac Gym GPU管道的核心数据结构）
        # ──────────────────────────────────────────────────────────────────────
        # Isaac Gym使用共享GPU内存存储状态，通过acquire获得指针后用gymtorch包装为Tensor
        # 这些张量与仿真内存共享，gym.refresh_xxx后数据自动更新

        # actor根状态张量：每个actor 13维 [px,py,pz, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        # DOF状态张量：每个DOF 2维 [position, velocity]
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # 刚体状态张量：每个刚体 13维（与actor根状态格式相同）
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # 接触力张量：每个刚体3维接触力 [fx, fy, fz]
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        # 雅可比张量：another_hand的末端效应器雅可比矩阵（用于IK计算）
        self.jacobian_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_jacobian_tensor(self.sim, "another_hand")
        )

        # 刷新所有状态张量（确保获取最新仿真状态）
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # ──────────────────────────────────────────────────────────────────────
        # 4.16 AllegroHand 默认关节角度（初始姿态）
        # ──────────────────────────────────────────────────────────────────────
        # another_hand（投手/左手）的默认关节角度
        # num_allegro_hand_dofs = 22（6个臂关节 + 16个手指关节）
        self.another_allegro_hand_default_dof_pos = torch.zeros(
            self.num_allegro_hand_dofs, dtype=torch.float, device=self.device
        )
        # 前6维：xArm6机械臂关节角度（弧度）
        # [-0.0, -0.09, -0.09, 3.141, 2.00, -1.57] 对应机械臂的初始伸展姿态
        self.another_allegro_hand_default_dof_pos[:6] = torch.tensor(
            [-0.0, -0.09, -0.09, 3.141, 2.00, -1.57], dtype=torch.float, device=self.device
        )
        # 后16维：AllegroHand 16个手指关节角度（弧度）
        # 这是一个"持球"姿态（经过调优的默认抓握构型）
        self.another_allegro_hand_default_dof_pos[6:] = to_torch([
            -0.03989830748810656, 1.3495253790945758, 0.8659920759388671, 0.780414711365591,   # 食指
             0.9655586519308622, 1.0139016439397597, 0.8501943208059994, 1.3264760744914152,   # 中指
            -0.20482272250532974, 1.347864170294202, 0.6030536585610538, 0.9181400800651911,   # 无名指
            -0.21341465375119012, 1.7199185039090872, 1.2686849760515697, 0.8245164874462315,  # 拇指
        ], dtype=torch.float, device=self.device)

        # hand（接手/右手）的默认关节角度
        self.allegro_hand_default_dof_pos = torch.zeros(
            self.num_allegro_hand_dofs, dtype=torch.float, device=self.device
        )
        # 前6维：xArm6机械臂关节角度
        # self.allegro_hand_default_dof_pos[:6] = torch.tensor(
        #     [0, 0, -1, 3.14, 0.57, 3.14], ...)  # 另一种姿态（已注释）
        self.allegro_hand_default_dof_pos[:6] = torch.tensor(
            [-0.0, -0.09, -0.09, 3.141, 2.00, -1.57], dtype=torch.float, device=self.device
        )
        ## default qpos（默认接球姿态，与投手相同）
        self.allegro_hand_default_dof_pos[6:] = to_torch([
            -0.03989830748810656, 1.3495253790945758, 0.8659920759388671, 0.780414711365591,
             0.9655586519308622, 1.0139016439397597, 0.8501943208059994, 1.3264760744914152,
            -0.20482272250532974, 1.347864170294202, 0.6030536585610538, 0.9181400800651911,
            -0.21341465375119012, 1.7199185039090872, 1.2686849760515697, 0.8245164874462315,
        ], dtype=torch.float, device=self.device)

        ## hand put（张开/放置姿态，已注释）
        # self.allegro_hand_default_dof_pos[6:] = to_torch(
        #     [0,0,0.7,1.2,0,0.7,0.3,1.2, 0,0,0.7,1.2,0,0,0.7,1.2,], ...)

        ## hand grip（握拳姿态，已注释）
        # self.allegro_hand_default_dof_pos[6:] = to_torch(
        #     [0,0.5,0.7,1.2,1.57,0.3,1.2,0.7,0,0.3,0.7,1.2,0,0.5,0.7,1.2,], ...)

        # ──────────────────────────────────────────────────────────────────────
        # 4.17 DOF状态张量切片（从全局dof_state_tensor中切出对应手的数据）
        # ──────────────────────────────────────────────────────────────────────
        # 将原始DOF张量包装为PyTorch张量
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # 形状：[num_envs, num_dofs_per_env, 2]
        # 切出 hand（右手/接手）的DOF状态，取前 num_allegro_hand_dofs 个
        self.allegro_hand_dof_state = self.dof_state.view(
            self.num_envs, -1, 2
        )[:, :self.num_allegro_hand_dofs]
        # 关节位置：[num_envs, 22]（第0列）
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        # 关节速度：[num_envs, 22]（第1列）
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        # 切出 another_hand（左手/投手）的DOF状态，取第 22~44 个
        self.allegro_hand_another_dof_state = self.dof_state.view(
            self.num_envs, -1, 2
        )[:, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs * 2]
        # 左手关节位置：[num_envs, 22]
        self.allegro_hand_another_dof_pos = self.allegro_hand_another_dof_state[..., 0]
        # 左手关节速度：[num_envs, 22]
        self.allegro_hand_another_dof_vel = self.allegro_hand_another_dof_state[..., 1]

        # ──────────────────────────────────────────────────────────────────────
        # 4.18 刚体状态和根状态张量
        # ──────────────────────────────────────────────────────────────────────
        # 刚体状态：[num_envs, num_bodies_per_env, 13]
        # 每个刚体13维：位置(3) + 四元数(4) + 线速度(3) + 角速度(3)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        # 每个环境中的刚体数量
        self.num_bodies = self.rigid_body_states.shape[1]

        # actor根状态张量：[total_actors, 13]
        # total_actors = num_envs × (手×2 + 物体 + 目标 + 预测目标)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        # 从根状态张量切出各个分量（这些是视图，不占额外内存）
        self.hand_positions    = self.root_state_tensor[:, 0:3]   # 所有actor的位置
        self.hand_orientations = self.root_state_tensor[:, 3:7]   # 所有actor的四元数朝向
        self.hand_linvels      = self.root_state_tensor[:, 7:10]  # 所有actor的线速度
        self.hand_angvels      = self.root_state_tensor[:, 10:13] # 所有actor的角速度
        # 保存初始根状态（用于reset时恢复）
        self.saved_root_tensor = self.root_state_tensor.clone()

        # 接触力张量：[num_envs, num_bodies_per_env * 3]（已展平）
        self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1)

        # ──────────────────────────────────────────────────────────────────────
        # 4.19 控制目标缓冲区
        # ──────────────────────────────────────────────────────────────────────
        # 总DOF数（包含两只手）
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        # 上一步的DOF目标位置（用于移动平均平滑）
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        # 当前步的DOF目标位置（由策略网络输出计算得到）
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        # 物体初始四元数（reset时保存，用于旋转相关计算）
        self.object_init_quat = torch.zeros(
            (self.num_envs, 4), dtype=torch.float, device=self.device
        )

        # ──────────────────────────────────────────────────────────────────────
        # 4.20 常用单位向量（扩展到 num_envs 个环境）
        # ──────────────────────────────────────────────────────────────────────
        # x轴单位向量，形状 [num_envs, 3]，用于四元数旋转变换
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        # y轴单位向量
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        # z轴单位向量
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # ──────────────────────────────────────────────────────────────────────
        # 4.21 成功统计缓冲区
        # ──────────────────────────────────────────────────────────────────────
        # 目标重置缓冲区（goal_dist≈0时触发，通常不会被触发因为goal是静态目标）
        self.reset_goal_buf = self.reset_buf.clone()
        # 每个环境的连续成功次数（达到目标方向对齐次数）
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 滑动平均后的连续成功次数（EMA平滑）
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # ── 接住成功相关统计量 ──────────────────────────────────────────────
        # 本episode是否成功接住（0=未接住/未判定，1=已接住）
        self.catch_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # "接住"判定的距离阈值（接手指到物体的距离 <= 此值才认为贴近）
        self.catch_tolerance = self.cfg["env"].get("catchTolerance", 0.15)  # 默认15cm
        # 累计成功接住的episode总数
        self.total_catch_successes = 0
        # 累计完成的episode总数（attempts）
        self.total_attempts = 0
        # 连续满足接住条件的帧计数器（中断即归零，防止"擦过"算成功）
        self.catch_hold_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 需要连续保持多少帧才算"接稳"（默认5帧，在60Hz下约83ms）
        self.catch_hold_steps = self.cfg["env"].get("catchHoldSteps", 5)
        # 物体速度阈值（m/s），速度过大说明只是弹跳而非稳定接住（当前代码已注释掉此判断）
        self.catch_vel_tolerance = self.cfg["env"].get("catchVelTolerance", 1.0)
        # 平滑后的episode成功率（用于TensorBoard曲线）
        self.smoothed_episode_success_rate = 0.0

        # av_factor需要转为张量（供JIT函数使用）
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        # 开环控制时使用的物体姿态记录（用于对比闭环观测）
        self.object_pose_for_open_loop = torch.zeros_like(
            self.root_state_tensor[self.object_indices, 0:7]
        )

        # 全局计数器
        self.total_successes = 0    # 所有环境的累计成功次数
        self.total_resets = 0       # 所有环境的累计reset次数

        # ──────────────────────────────────────────────────────────────────────
        # 4.22 帧历史堆叠缓冲区（用于时序观测，3帧历史）
        # ──────────────────────────────────────────────────────────────────────
        # 观测帧堆栈（3帧）：每帧 one_frame_num_obs=300 维
        # 最终观测为 300×3=900 维的时序堆叠向量
        self.state_buf_stack_frames = []
        self.obs_buf_stack_frames = []

        for i in range(3):
            # 观测帧缓冲：[num_envs, 300]
            self.obs_buf_stack_frames.append(
                torch.zeros_like(self.obs_buf[:, 0:self.one_frame_num_obs])
            )
            # Critic状态帧缓冲：[num_envs, 215]
            self.state_buf_stack_frames.append(
                torch.zeros_like(self.states_buf[:, 0:215])
            )

        # ──────────────────────────────────────────────────────────────────────
        # 4.23 物体轨迹历史缓冲区（TrajEstimator的输入）
        # ──────────────────────────────────────────────────────────────────────
        # 物体位置历史长度（帧数）
        self.object_seq_len = 20
        # 物体位置历史缓冲区：[num_envs, 20×3=60]
        # 滑动窗口方式更新：最新帧写入最后，之前的向前移位
        self.object_state_stack_frames = torch.zeros(
            (self.num_envs, self.object_seq_len * 3),
            dtype=torch.float, device=self.device
        )

        # 本体感知闭环观测（接手的前22个关节的当前位置，用于闭环控制）
        self.proprioception_close_loop = torch.zeros_like(
            self.allegro_hand_dof_pos[:, 0:22]
        )

        # 找到两只手"link6"（末端执行器基座）在刚体索引中的位置
        # link6 = xArm6末端法兰，用于读取末端执行器位置
        self.another_hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.another_hand_indices[0], "link6", gymapi.DOMAIN_ENV
        )
        print("another_hand_base_rigid_body_index: ", self.another_hand_base_rigid_body_index)
        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.hand_indices[0], "link6", gymapi.DOMAIN_ENV
        )
        print("hand_base_rigid_body_index: ", self.hand_base_rigid_body_index)

        # 施加外力的力张量：[num_envs, num_bodies, 3]（用于扰动测试）
        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )
        # 物体刚体数量（用于确定施力点的索引）
        object_rb_count = self.gym.get_asset_rigid_body_count(self.object_asset)
        # 物体刚体句柄（硬编码索引46，对应场景中物体刚体的位置）
        self.object_rb_handles = 46
        # 随机扰动方向：[num_envs, 6]（xyz位置扰动 + xyz旋转扰动）
        self.perturb_direction = torch_rand_float(
            -1, 1, (self.num_envs, 6), device=self.device
        ).squeeze(-1)

        # 轨迹预测位置（初始化为goal初始位置）
        self.predict_pose = self.goal_init_state[:, 0:3].clone()

        # ──────────────────────────────────────────────────────────────────────
        # 4.24 TensorBoard日志
        # ──────────────────────────────────────────────────────────────────────
        # 从配置中读取算法名称（如"MAPPO"），用于日志目录命名
        self.algorithm_name = self.cfg["env"]["algorithm_name"]
        # 日志保存路径：./logs/allegro_hand_dynamic_handover/{算法名}/success_rate_logs_seed{种子}
        self.log_dir = str(
            './logs/allegro_hand_dynamic_handover/{}/success_rate_logs_seed{}'.format(
                self.algorithm_name, self.cfg["seed"]
            )
        )
        # 创建TensorBoard写入器
        self.writter = SummaryWriter(self.log_dir)

        # ──────────────────────────────────────────────────────────────────────
        # 4.25 轨迹估计网络初始化
        # ──────────────────────────────────────────────────────────────────────
        # 实例化TrajEstimator：输入60维（20帧×3坐标），输出3维（落点xyz）
        self.traj_estimator = TrajEstimator(input_dim=60, output_dim=3).to(self.device)
        # 开启所有参数的梯度（确保网络可训练）
        for param in self.traj_estimator.parameters():
            param.requires_grad_(True)

        # 是否为测试模式（影响轨迹估计网络的加载和训练/推断模式）
        self.is_test = self.cfg["is_test"]

        # Adam优化器（在线训练TrajEstimator用）
        self.traj_estimator_optimizer = torch.optim.Adam(
            self.traj_estimator.parameters(), lr=0.0003
        )
        # 模型保存路径
        self.traj_estimator_save_path = "./traj_e/"
        os.makedirs(self.traj_estimator_save_path, exist_ok=True)

        # BCE损失（当前代码中未实际使用，遗留）
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss()

        if self.is_test:
            # 测试模式：从磁盘加载预训练好的模型，切换为推断模式（不更新参数）
            self.traj_estimator.load_state_dict(
                torch.load("./traj_e/model.pt", map_location='cuda:0')
            )
            self.traj_estimator.eval()
        else:
            # 训练模式：在线学习（每步都会更新网络参数）
            # self.traj_estimator.load_state_dict(torch.load("./traj_e/model_perfect.pt", map_location='cuda:0'))
            self.traj_estimator.train()

        # 全局仿真步数计数器（按环境数累加：每次调用 += num_envs）
        self.total_steps = 0
        # 成功缓冲区（与rew_buf形状相同，用于记录当前帧的成功标志）
        self.success_buf = torch.zeros_like(self.rew_buf)
        # 击中成功缓冲区（预留，当前未使用）
        self.hit_success_buf = torch.zeros_like(self.rew_buf)

    # ==========================================================================
    # 5. 内部状态访问接口
    # ==========================================================================

    def get_internal_state(self):
        """
        获取物体的四元数旋转状态（供外部调用，如评估脚本）

        返回：
            Tensor [num_envs, 4]：每个环境中操作物体的四元数 [qx, qy, qz, qw]
        """
        # object_indices: 每个环境中物体actor的全局索引
        # root_state_tensor[..., 3:7]：四元数部分
        return self.root_state_tensor[self.object_indices, 3:7]

    def get_internal_info(self, key):
        """
        通过key获取内部调试信息

        参数：
            key (str): 信息类型
                'target'  → 调试目标位置列表
                'qpos'    → 调试关节角度列表
                'contact' → 手指接触状态

        返回：
            对应的调试数据，或 None（key未知时）
        """
        if key == 'target':
            return self.debug_target    # 目标位置调试记录
        elif key == 'qpos':
            return self.debug_qpos      # 关节角度调试记录
        elif key == 'contact':
            return self.finger_contacts # 手指接触状态
        return None

    # ==========================================================================
    # 6. 奖励计算
    # ==========================================================================

    def compute_reward(self, actions):
        """
        调用JIT奖励函数计算奖励，并更新统计信息

        核心逻辑：
            1. 调用 compute_hand_reward（@torch.jit.script 加速版）
            2. 更新接住成功率统计（total_attempts, total_catch_successes）
            3. 计算EMA平滑成功率
            4. 写入TensorBoard

        参数：
            actions (Tensor): [num_envs, num_actions]，当前步策略输出的动作
        """
        # 调用JIT奖励函数，批量计算所有环境的奖励、reset信号和成功统计
        (
            self.rew_buf[:],             # 奖励值
            self.reset_buf[:],           # episode重置标志（1=需要重置）
            self.reset_goal_buf[:],      # 目标重置标志（1=需要重新生成目标）
            self.progress_buf[:],        # episode内步数进度
            self.successes[:],           # 旋转对齐成功次数
            self.consecutive_successes[:], # EMA平滑连续成功次数
            self.catch_successes[:],     # 接住成功标志（本局）
            self.catch_hold_counter[:],  # 连续接住帧计数器
        ) = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf,
            self.successes, self.consecutive_successes,
            self.max_episode_length,
            self.object_pos, self.object_rot,
            self.goal_pos, self.goal_rot,
            self.allegro_left_hand_pos,      # 接手（右手）掌心位置
            self.allegro_right_hand_pos,     # 投手（左手）掌心位置（变量名有些混淆）
            self.allegro_hand_another_thmub_pos,  # 左手拇指指尖位置
            self.aux_up_pos,                 # 辅助上方参考点
            self.object_linvel,              # 物体线速度
            self.leeft_hand_ee_rot,          # 接手末端执行器旋转
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps,
            self.actions, self.action_penalty_scale,
            # 各手指尖位置（用于多指接触奖励，当前部分已注释）
            self.allegro_hand_another_ff_pos,  # 左手食指
            self.allegro_hand_another_mf_pos,  # 左手中指
            self.allegro_hand_another_rf_pos,  # 左手无名指
            self.allegro_hand_ff_pos,          # 右手食指
            self.allegro_hand_mf_pos,          # 右手中指
            self.allegro_hand_rf_pos,          # 右手无名指
            self.a_hand_palm_pos,              # 接手掌心（未偏移版本）
            # 默认关节角度（用于计算姿态偏差惩罚）
            unscale(self.another_allegro_hand_default_dof_pos[6:],
                    self.allegro_hand_dof_lower_limits[6:22],
                    self.allegro_hand_dof_upper_limits[6:22]),
            # 当前关节角度（归一化到[-1,1]）
            unscale(self.allegro_hand_another_dof_pos[:, 6:22],
                    self.allegro_hand_dof_lower_limits[6:22],
                    self.allegro_hand_dof_upper_limits[6:22]),
            self.success_tolerance, self.reach_goal_bonus,
            self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,
            (self.object_type == "pen"),   # ignore_z_rot：笔类物体忽略绕轴旋转
            # 接住成功判定参数
            self.catch_successes, self.catch_hold_counter,
            self.catch_tolerance,           # 接住距离阈值(m)
            self.catch_hold_steps,          # 需要连续稳定的帧数
            self.catch_vel_tolerance,       # 物体速度阈值(m/s)
        )

        # 将连续成功次数传入extras（供MAPPO算法读取，用于课程学习判断）
        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        # 按并行环境数累加全局步数
        self.total_steps += self.num_envs

        # ── 成功率统计 ────────────────────────────────────────────────────────
        # 本次调用中触发reset的环境数（即完成的episode数）
        num_attempts_this_call = self.reset_buf.sum().item()
        # 其中"接住成功"的episode数（catch_successes=1 且 reset=1）
        num_catches_this_call = (self.catch_successes * self.reset_buf).sum().item()

        self.total_attempts += num_attempts_this_call
        self.total_catch_successes += num_catches_this_call

        # 全局累计成功率
        success_rate = (
            self.total_catch_successes / self.total_attempts
            if self.total_attempts > 0
            else 0.0
        )

        if num_attempts_this_call > 0:
            # 当前这批reset的原始成功率
            raw_episode_success_rate = num_catches_this_call / num_attempts_this_call
            # EMA平滑系数（=av_factor，越小越平滑）
            alpha = float(self.av_factor.item())
            if self.total_attempts == num_attempts_this_call:
                # 第一次有reset：直接赋值（无历史可平滑）
                self.smoothed_episode_success_rate = raw_episode_success_rate
            else:
                # EMA平滑：保留 (1-alpha) 的历史信息，融入 alpha 的新数据
                self.smoothed_episode_success_rate = (
                    alpha * raw_episode_success_rate
                    + (1 - alpha) * self.smoothed_episode_success_rate
                )

        print('Success Rate:', success_rate,
              'Average episode Success Rate:', self.smoothed_episode_success_rate)

        # ── TensorBoard记录 ───────────────────────────────────────────────────
        self.writter.add_scalar('Total Attempts', float(self.total_attempts), self.total_steps)
        self.writter.add_scalar('Successful Throws and Catches',
                                float(self.total_catch_successes), self.total_steps)
        self.writter.add_scalar('Success Rate', success_rate, self.total_steps)
        self.writter.add_scalar('Average Episode Success Rate',
                                self.smoothed_episode_success_rate, self.total_steps)

        if self.print_success_stat:
            # 打印更详细的连续成功统计（旋转对齐成功，非接住成功）
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))

    # ==========================================================================
    # 7. 观测量计算
    # ==========================================================================

    def compute_observations(self):
        """
        计算当前时刻所有环境的观测量（Actor输入 + Critic状态）

        流程：
            1. 刷新GPU仿真状态张量
            2. 从张量中提取各刚体位置/速度
            3. 更新物体位置历史缓冲区（object_state_stack_frames）
            4. 调用TrajEstimator预测落点（在线训练+推断）
            5. 填充观测缓冲区（obs_buf）和Critic状态缓冲区（states_buf）
        """
        # ── 7.1 刷新GPU状态张量（让张量数据与最新仿真状态同步）──────────────
        self.gym.refresh_dof_state_tensor(self.sim)           # 关节角度和速度
        self.gym.refresh_actor_root_state_tensor(self.sim)    # actor根状态（位置、速度）
        self.gym.refresh_rigid_body_state_tensor(self.sim)    # 所有刚体状态
        self.gym.refresh_net_contact_force_tensor(self.sim)   # 接触力
        self.gym.refresh_jacobian_tensors(self.sim)           # 雅可比矩阵（IK用）

        # ── 7.2 提取手部状态 ──────────────────────────────────────────────────
        # 右手（投手/hand）的根状态（机械臂底座位置）
        self.allegro_right_hand_base_pos = self.root_state_tensor[self.hand_indices, 0:3]
        self.allegro_right_hand_base_rot = self.root_state_tensor[self.hand_indices, 3:7]

        # 右手末端（link6，手腕处）的刚体状态
        # 刚体索引6 = hand的link6
        self.allegro_right_hand_pos = self.rigid_body_states[:, 6, 0:3]
        self.allegro_right_hand_rot = self.rigid_body_states[:, 6, 3:7]

        # 左手（接手/another_hand）末端的刚体状态
        # 刚体索引 6+23=29（another_hand的刚体偏移量为23）
        self.allegro_left_hand_pos = self.rigid_body_states[:, 6 + 23, 0:3]
        self.allegro_left_hand_rot = self.rigid_body_states[:, 6 + 23, 3:7]

        # 保存左手掌心原始位置（未偏移版本，用于奖励函数）
        self.a_hand_palm_pos = self.allegro_left_hand_pos.clone()

        # 将左手掌心位置沿手的局部坐标系偏移，得到更准确的"接触中心"
        # 沿局部y轴偏移0.08m（向前）
        self.allegro_left_hand_pos = self.allegro_left_hand_pos + quat_apply(
            self.allegro_left_hand_rot,
            to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.08
        )
        # 沿局部z轴偏移0.04m（向上）
        self.allegro_left_hand_pos = self.allegro_left_hand_pos + quat_apply(
            self.allegro_left_hand_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04
        )

        # ── 7.3 提取物体状态 ──────────────────────────────────────────────────
        # 物体的完整姿态 [num_envs, 7]：位置(3) + 四元数(4)
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        # 物体位置 [num_envs, 3]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        # 物体四元数朝向 [num_envs, 4]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        # 物体线速度 [num_envs, 3]（m/s）
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        # 物体角速度 [num_envs, 3]（rad/s）
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        # ── 7.4 目标位置（goal状态来自goal_states，不是root_state_tensor）
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos  = self.goal_states[:, 0:3]
        self.goal_rot  = self.goal_states[:, 3:7]

        # ── 7.5 手指关键点位置（用于接触奖励和观测）────────────────────────
        # left/another_hand（投手）拇指指尖（刚体索引 14+23=37）
        self.allegro_hand_another_thmub_pos = self.rigid_body_states[:, 14 + 23, 0:3]
        self.allegro_hand_another_thmub_rot = self.rigid_body_states[:, 14 + 23, 3:7]

        # 右手（接手/hand）各手指指尖位置（用于多指接触判断，当前代码中已注释掉奖励使用）
        self.allegro_hand_another_ff_pos = self.rigid_body_states[:, 10, 0:3]     # 食指
        self.allegro_hand_another_mf_pos = self.rigid_body_states[:, 18, 0:3]     # 中指
        self.allegro_hand_another_rf_pos = self.rigid_body_states[:, 22, 0:3]     # 无名指

        # left/another_hand（投手）各手指指尖
        self.allegro_hand_ff_pos = self.rigid_body_states[:, 10 + 23, 0:3]
        self.allegro_hand_mf_pos = self.rigid_body_states[:, 18 + 23, 0:3]
        self.allegro_hand_rf_pos = self.rigid_body_states[:, 22 + 23, 0:3]

        # 接手（another_hand）末端执行器的旋转（从刚体索引读取）
        self.leeft_hand_ee_rot = self.rigid_body_states[:, self.another_hand_base_rigid_body_index, 3:7]

        # ── 7.6 生成随机噪声（用于域随机化观测扰动）──────────────────────────
        # 63个随机数，用于观测量的随机扰动（提升Sim2Real鲁棒性）
        rand_floats = torch_rand_float(-1.0, 1.0, (self.num_envs, 63), device=self.device)

        # 辅助参考点（固定的空间位置，用于某些相对坐标计算）
        self.aux_up_pos = to_torch(
            [0, -0.52, 0.45], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        # ── 7.7 填充观测和状态缓冲区 ──────────────────────────────────────────
        # 计算 Actor 观测量（full_state 或 sim2real 观测）
        self.compute_sim2real_observation(rand_floats)

        # 如果使用非对称观测，额外计算 Critic 特权状态
        if self.asymmetric_obs:
            self.compute_sim2real_asymmetric_obs(rand_floats)

    def compute_sim2real_observation(self, rand_floats):
        """
        计算 Actor 的观测向量（设计为 Sim2Real 迁移友好的观测）

        观测结构（300维 × 3帧 = 900维总输入）：
            obs[0:22]   → 右手（接手）关节角度（归一化到[-1,1]）
            obs[22:25]  → 目标位置相对于右手底座的偏移（goal_pos - hand_base_pos）
            obs[150:172]→ 左手（投手）关节角度（归一化）
            obs[248:260]→ 物体位置历史（最近4帧×3维，加噪声）
            obs[260:263]→ TrajEstimator预测的落点（xyz）

        帧堆叠：
            最终输出 obs_buf 包含 3 帧历史，每帧300维，共900维

        参数：
            rand_floats (Tensor): [num_envs, 63]，随机噪声，用于观测扰动
        """
        # ── 右手（接手）关节角度观测（前22个DOF，归一化）──────────────────
        # unscale：将 [lower, upper] 范围的关节角度归一化到 [-1, 1]
        self.obs_buf[:, 0:22] = unscale(
            self.allegro_hand_dof_pos,
            self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits
        )
        # 清零臂关节观测（前6个自由度不暴露给策略，避免末端位置信息泄露）
        self.obs_buf[:, 0:6] = 0
        # 保留 joint1 和 joint2 的位置（可能用于末端位置辅助）
        self.obs_buf[:, 1] = self.allegro_hand_dof_pos[:, 1]
        self.obs_buf[:, 2] = self.allegro_hand_dof_pos[:, 2]

        # ── 目标位置相对偏移（接手应该移动的方向）──────────────────────────
        # [num_envs, 3]：目标位置 - 接手底座位置，表示接手还需要移动多远
        self.obs_buf[:, 22:25] = (self.goal_pos - self.allegro_right_hand_base_pos).clone()

        # ── 左手（投手）关节角度观测（obs[150:172]）─────────────────────────
        self.obs_buf[:, 150:172] = unscale(
            self.allegro_hand_another_dof_pos,
            self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits
        )
        # 投手臂关节：直接使用原始角度（不归一化）
        self.obs_buf[:, 150:156] = self.allegro_hand_another_dof_pos[:, :6]
        # 清零某些臂关节（减少观测维度冗余）
        self.obs_buf[:, 150:151] = 0   # joint0 清零
        self.obs_buf[:, 153:154] = 0   # joint3 清零

        # ── 物体位置历史缓冲区更新（滑动窗口）───────────────────────────────
        # object_state_stack_frames: [num_envs, 20×3=60]
        # 更新规则：将旧帧向前移动，新帧写入最后
        for i in range(self.object_seq_len):
            if i == self.object_seq_len - 1:
                # 最后一个槽位：写入当前帧（物体位置相对于右手底座）
                self.object_state_stack_frames[:, (i) * 3:(i + 1) * 3] = (
                    self.object_pos - self.allegro_right_hand_base_pos
                ).clone()
            else:
                # 其他槽位：向前移位（i ← i+1，实现先进先出滑动窗口）
                self.object_state_stack_frames[:, (i) * 3:(i + 1) * 3] = \
                    self.object_state_stack_frames[:, (i + 1) * 3:(i + 2) * 3].clone()

        # ── 轨迹估计网络在线推断和训练 ───────────────────────────────────────
        with TemporaryGrad():
            # 前向推断：输入60维物体历史，输出3维预测落点
            self.predict_pose, self.pose_latent_vector = self.predict_contact_pose(
                self.traj_estimator, self.object_state_stack_frames
            )
            # 反向传播：用真实落点（goal_pos）监督网络，在线更新参数
            self.update_contact_slamer(self.predict_pose)

        # ── 将预测落点写入观测（obs[260:263]）───────────────────────────────
        # .detach()：预测结果作为观测不参与策略梯度计算
        self.obs_buf[:, 260:263] = self.predict_pose[:, 0:3].detach()
        # 注释掉的版本：直接用真实目标位置（相当于作弊/Oracle观测）
        # self.obs_buf[:, 260:263] = (self.goal_pos - self.allegro_right_hand_base_pos).clone()

        # ── 物体最近4帧位置历史写入观测（obs[248:260]，加随机噪声）──────────
        # object_state_stack_frames[:, 36:48] 对应最近4帧（帧12~15）的位置
        # 加 rand_floats * 0.05 噪声提升鲁棒性（约±5cm扰动）
        self.obs_buf[:, 248:260] = (
            self.object_state_stack_frames[:, 36:48].clone()
            + rand_floats[:, 0:12] * 0.05
        )

        # ── 帧历史堆叠（将当前帧加入3帧历史队列）────────────────────────────
        # obs_buf_stack_frames[0] = frame_{t-1}
        # obs_buf_stack_frames[1] = frame_{t-2}
        # 将历史帧移入 obs_buf 的后续300维槽位
        for i in range(len(self.obs_buf_stack_frames) - 1):
            # obs[300:600] ← obs_stack_frames[0]（t-1帧）
            # obs[600:900] ← obs_stack_frames[1]（t-2帧）
            self.obs_buf[:, (i + 1) * self.one_frame_num_obs:(i + 2) * self.one_frame_num_obs] = \
                self.obs_buf_stack_frames[i]
            # 更新帧队列：将当前帧的前300维存入历史
            self.obs_buf_stack_frames[i] = \
                self.obs_buf[:, (i) * self.one_frame_num_obs:(i + 1) * self.one_frame_num_obs].clone()

    def predict_contact_pose(self, traj_estimator, contact_buf):
        """
        调用TrajEstimator进行轨迹落点预测

        参数：
            traj_estimator (TrajEstimator): 轨迹估计网络实例
            contact_buf    (Tensor):        物体位置历史 [num_envs, 60]

        返回：
            predict_pose        (Tensor): [num_envs, 3]，预测落点坐标
            pose_latent_vector  (Tensor): [num_envs, 128]，中间隐层特征
        """
        predict_pose, pose_latent_vector = traj_estimator(contact_buf)
        return predict_pose, pose_latent_vector

    def update_contact_slamer(self, predict_pose):
        """
        用MSE损失在线训练TrajEstimator（SLAM风格的在线学习）

        损失：预测落点 vs. 真实目标位置（相对于接手底座）
        梯度更新：Adam优化器单步更新

        参数：
            predict_pose (Tensor): [num_envs, 3]，TrajEstimator预测的落点

        副作用：
            - 更新 traj_estimator 的参数
            - 将 pos_loss 写入 extras（供算法框架记录）
        """
        # MSE损失：预测落点 vs 真实目标（相对坐标）
        self.pos_loss = F.mse_loss(
            predict_pose[:, 0:3],
            (self.goal_pos - self.allegro_right_hand_base_pos).clone()
        )
        loss = self.pos_loss

        # 标准PyTorch反向传播流程
        self.traj_estimator_optimizer.zero_grad()   # 清零梯度
        loss.backward()                              # 反向传播计算梯度
        self.traj_estimator_optimizer.step()         # Adam参数更新

        # 将损失传入extras（供外部记录，如TensorBoard）
        self.extras['pos_loss'] = self.pos_loss.unsqueeze(0)

    def compute_sim2real_asymmetric_obs(self, rand_floats):
        """
        计算 Critic 的非对称特权状态向量（215维）

        非对称观测设计原理：
            - Actor 只能使用真实部署中可获取的传感器信息（本体感知+预测落点）
            - Critic 在训练时可使用额外的特权信息（完整物体状态、真实目标位置等）
            - Critic的高质量价值估计可以指导Actor学习，无需Actor观测到这些信息

        状态向量布局（215维）：
            [0:22]   → 右手关节角度（归一化）
            [22:44]  → 右手关节速度（×vel_obs_scale=0.2）
            [44:66]  → 右手动作（策略输出）
            [66:88]  → 左手关节角度（归一化）
            [88:110] → 左手关节速度
            [110:132]→ 左手动作
            [132:143]→ 11个接触传感器读数（0/1二值化）
            [143:150]→ 物体姿态（位置+四元数）
            [150:153]→ 物体线速度
            [153:156]→ 物体角速度（×vel_obs_scale）
            [156:163]→ 目标姿态
            [163:167]→ 目标与物体四元数差
            [167:179]→ 域随机化参数（12个随机数）
            [179:191]→ 物体历史位置（加噪声）
            [191:194]→ 相对目标位置（goal_pos - hand_base_pos）

        参数：
            rand_floats (Tensor): [num_envs, 63]，随机噪声
        """
        # ── 右手（接手）关节角度（归一化）──────────────────────────────────
        self.states_buf[:, 0:self.num_allegro_hand_dofs] = unscale(
            self.allegro_hand_dof_pos,
            self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits
        )
        # 右手关节速度（缩放到合理范围）
        self.states_buf[:, self.num_allegro_hand_dofs:2 * self.num_allegro_hand_dofs] = \
            self.vel_obs_scale * self.allegro_hand_dof_vel

        # ── 右手动作（策略输出的前22个动作）──────────────────────────────────
        action_obs_start = 44  # 偏移量
        self.states_buf[:, action_obs_start:action_obs_start + 22] = self.actions[:, :22]

        # ── 左手（投手）关节角度和速度 ──────────────────────────────────────
        another_hand_start = action_obs_start + 22  # = 66
        self.states_buf[:, another_hand_start:self.num_allegro_hand_dofs + another_hand_start] = \
            unscale(self.allegro_hand_another_dof_pos,
                    self.allegro_hand_dof_lower_limits,
                    self.allegro_hand_dof_upper_limits)
        self.states_buf[:, self.num_allegro_hand_dofs + another_hand_start:
                           2 * self.num_allegro_hand_dofs + another_hand_start] = \
            self.vel_obs_scale * self.allegro_hand_another_dof_vel

        # ── 左手动作（策略输出的后22个动作）──────────────────────────────────
        action_another_obs_start = another_hand_start + 44  # = 110
        self.states_buf[:, action_another_obs_start:action_another_obs_start + 22] = \
            self.actions[:, 22:]

        # ── 接触传感器读数（11个传感器，二值化：≥1N则为1，否则为0）──────────
        contact_start = action_another_obs_start + 22  # = 132
        contacts = self.contact_tensor.reshape(self.num_envs, -1, 3)  # 展开为 [envs, 刚体数, 3]
        contacts = contacts[:, self.sensor_handle_indices, :]          # 只取11个传感器刚体
        contacts = torch.norm(contacts, dim=-1)                        # 计算力的模 [envs, 11]
        contacts = torch.where(contacts >= 1.0, 1.0, 0.0)             # 二值化（1N为阈值）
        self.states_buf[:, contact_start:contact_start + 11] = contacts

        # ── 物体完整状态（特权信息：Actor看不到，Critic可以看到）───────────
        obj_obs_start = contact_start + 11  # = 143
        self.states_buf[:, obj_obs_start:obj_obs_start + 7]   = self.object_pose    # 位置+四元数
        self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10]  = self.object_linvel  # 线速度
        self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = \
            self.vel_obs_scale * self.object_angvel  # 角速度（缩放）

        # ── 目标姿态和相对旋转差 ──────────────────────────────────────────────
        goal_obs_start = obj_obs_start + 13  # = 156
        self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        # 四元数乘法：计算物体朝向与目标朝向的旋转差（4维四元数）
        self.states_buf[:, goal_obs_start + 7:goal_obs_start + 11] = \
            quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

        # ── 域随机化参数（让Critic知道当前环境的随机化程度）─────────────────
        randomize_param_start = goal_obs_start + 11  # = 167
        self.states_buf[:, randomize_param_start:randomize_param_start + 12] = rand_floats[:, 0:12]

        # ── 物体历史位置 + 相对目标位置 ──────────────────────────────────────
        offseted_pos_goal_start = randomize_param_start + 12  # = 179
        # 物体最近4帧位置历史（加±5cm随机噪声）
        self.states_buf[:, offseted_pos_goal_start:offseted_pos_goal_start + 12] = \
            self.object_state_stack_frames[:, 36:48].clone() + rand_floats[:, 0:12] * 0.05
        # 目标相对位置（goal_pos - 接手底座位置）
        self.states_buf[:, offseted_pos_goal_start + 12:offseted_pos_goal_start + 15] = \
            (self.goal_pos - self.allegro_right_hand_base_pos).clone()

    # ==========================================================================
    # 8. 仿真世界创建
    # ==========================================================================

    def create_sim(self):
        """
        创建Isaac Gym仿真世界

        调用顺序：
            1. 设置物理参数（dt、up轴）
            2. 创建仿真实例（继承自BaseTask）
            3. 加载物体资产字典
            4. 创建地面
            5. 创建所有并行环境
        """
        # 从sim_params获取时间步长（s）
        self.dt = self.sim_params.dt
        # 设置重力方向（返回up_axis在[x,y,z]中的索引，z轴向上则返回2）
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        # 创建仿真实例（调用父类方法，返回gym.sim句柄）
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        # 加载所有物体的URDF/MJCF资产（构建asset_dict）
        self.create_object_asset_dict(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        )
        # 创建地面平面
        self._create_ground_plane()
        # 创建并行环境实例（包括手、物体、目标marker）
        self._create_envs(
            self.num_envs,
            self.cfg["env"]['envSpacing'],    # 环境间距（m）
            int(np.sqrt(self.num_envs))       # 每行环境数（正方形排列）
        )

    def _create_ground_plane(self):
        """
        创建地面平面（z=0的水平面，法向量朝上）
        """
        plane_params = gymapi.PlaneParams()
        # 法向量：(0,0,1) 表示z轴向上的水平地面
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_object_asset_dict(self, asset_root):
        """
        加载所有训练物体的URDF/MJCF资产，构建资产字典

        每种物体加载三个变体：
            'obj'         → 实际参与物理仿真的物体（有重力）
            'goal'        → 目标标记（无重力，半透明，显示目标位置）
            'predict goal'→ 预测标记（无重力，橙色，显示TrajEstimator预测落点）

        参数：
            asset_root (str): 资产根目录路径（../assets）
        """
        self.object_asset_dict = {}
        print("ENTER ASSET CREATING!")

        for used_objects in self.used_training_objects:
            # 获取该物体对应的URDF文件路径
            object_asset_file = self.asset_files_dict[used_objects]

            # 设置物体资产选项（有重力版本）
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 2000  # 密度 kg/m³（比水重，确保下落稳定）
            # object_asset_options.fix_base_link = True  # 固定基座（调试用，已注释）

            # 加载实际物体资产（受重力）
            self.object_asset = self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options
            )

            # 目标标记：禁用重力（悬浮在空中不动）
            object_asset_options.disable_gravity = True
            goal_asset = self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options
            )

            # 预测标记：同样禁用重力
            predict_goal_asset = self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options
            )

            # 将三个变体存入字典
            self.object_asset_dict[used_objects] = {
                'obj': self.object_asset,
                'goal': goal_asset,
                'predict goal': predict_goal_asset
            }

    def _create_envs(self, num_envs, spacing, num_per_row):
        """
        创建所有并行仿真环境实例

        每个环境包含：
            - 右手（hand）：xArm6 + AllegroHand（接手）
            - 左手（another_hand）：xArm6 + AllegroHand（投手）
            - 物体（object）：被抛接的目标物
            - 目标标记（goal_object）：半透明，标注期望接住位置
            - 预测标记（predict_goal_object）：橙色，显示轨迹预测落点
            - 可选相机传感器（enable_camera_sensors时）

        参数：
            num_envs    (int): 并行环境数量
            spacing     (float): 相邻环境的间距（m）
            num_per_row (int): 每行排列的环境数（正方形网格）
        """
        # 环境边界
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # ── 8.1 加载机械臂+手的URDF资产 ──────────────────────────────────────
        asset_root = "../assets"
        # 左手（接手的另一手）：xArm6 + AllegroHand 左手版URDF
        allegro_hand_asset_file = "urdf/xarm6/xarm6_allegro_left_2023.urdf"
        # 右手（接手）：xArm6 + AllegroHand 右手版URDF（Binghao改版）
        allegro_hand_another_asset_file = "urdf/xarm6/xarm6_allegro_right_2023_binghao.urdf"

        object_asset_file = self.asset_files_dict["ball"]  # 默认物体（实际会被覆盖）

        # 资产加载选项
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False  # 不翻转视觉附件
        asset_options.fix_base_link = True             # 固定机械臂底座（不受重力影响）
        asset_options.collapse_fixed_joints = True     # 合并固定关节（减少计算开销）
        asset_options.disable_gravity = True           # 机械臂本体禁用重力（独立关节控制）
        asset_options.thickness = 0.001                # 碰撞几何厚度（避免穿透）
        asset_options.angular_damping = 0.01           # 角速度阻尼（防止振荡）

        if self.physics_engine == gymapi.SIM_PHYSX:
            # PhysX引擎特有设置：使用PhysX铰链结构（更精确的关节动力学）
            asset_options.use_physx_armature = True
        # DOF驱动模式：力矩控制（DOF_MODE_EFFORT）
        # 后续会在DOF属性中改为位置控制（DOF_MODE_POS）
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        # 加载两只手的资产
        allegro_hand_asset = self.gym.load_asset(
            self.sim, asset_root, allegro_hand_asset_file, asset_options
        )
        allegro_hand_another_asset = self.gym.load_asset(
            self.sim, asset_root, allegro_hand_another_asset_file, asset_options
        )

        # ── 8.2 获取资产基本参数 ──────────────────────────────────────────────
        self.num_allegro_hand_bodies   = self.gym.get_asset_rigid_body_count(allegro_hand_asset)  # 刚体数
        self.num_allegro_hand_shapes   = self.gym.get_asset_rigid_shape_count(allegro_hand_asset) # 碰撞形状数
        self.num_allegro_hand_dofs     = self.gym.get_asset_dof_count(allegro_hand_asset)         # DOF数（=22）
        self.num_allegro_hand_actuators = self.gym.get_asset_dof_count(allegro_hand_asset)        # 执行器数
        self.num_allegro_hand_tendons  = self.gym.get_asset_tendon_count(allegro_hand_asset)      # 腱数（绳索约束）

        print("self.num_allegro_hand_bodies: ", self.num_allegro_hand_bodies)
        print("self.num_allegro_hand_shapes: ", self.num_allegro_hand_shapes)
        print("self.num_allegro_hand_dofs: ", self.num_allegro_hand_dofs)
        print("self.num_allegro_hand_actuators: ", self.num_allegro_hand_actuators)
        print("self.num_allegro_hand_tendons: ", self.num_allegro_hand_tendons)

        # 执行器DOF索引（只控制手指16个DOF，不包含臂关节）
        # 这里设为[0,1,...,15]，但实际在动作写入时会加偏移量
        self.actuated_dof_indices = [i for i in range(16)]

        # ── 8.3 设置DOF属性（驱动模式、刚度、阻尼、力矩限制）────────────────
        allegro_hand_dof_props         = self.gym.get_asset_dof_properties(allegro_hand_asset)
        allegro_hand_another_dof_props = self.gym.get_asset_dof_properties(allegro_hand_another_asset)

        # 用于存储DOF关节限位
        self.allegro_hand_dof_lower_limits = []   # 右手关节下限
        self.allegro_hand_dof_upper_limits = []   # 右手关节上限
        self.a_allegro_hand_dof_lower_limits = [] # 左手关节下限
        self.a_allegro_hand_dof_upper_limits = [] # 左手关节上限
        self.allegro_hand_dof_default_pos = []    # 默认关节位置
        self.allegro_hand_dof_default_vel = []    # 默认关节速度
        self.allegro_hand_dof_stiffness = []      # 关节刚度
        self.allegro_hand_dof_damping = []        # 关节阻尼
        self.allegro_hand_dof_effort = []         # 关节力矩限制
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_allegro_hand_dofs):  # 遍历22个DOF
            # 读取并存储关节限位
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            self.a_allegro_hand_dof_lower_limits.append(allegro_hand_another_dof_props['lower'][i])
            self.a_allegro_hand_dof_upper_limits.append(allegro_hand_another_dof_props['upper'][i])
            self.allegro_hand_dof_default_pos.append(0.0)  # 默认初始位置为0
            self.allegro_hand_dof_default_vel.append(0.0)  # 默认初始速度为0

            # 机械臂各关节的位置控制刚度（PD控制的P增益）
            # 关节顺序：shoulder1, shoulder2, elbow, wrist1, wrist2, wrist3
            self.stiffness = [100, 100, 64, 64, 64, 40]

            # 所有DOF统一使用位置控制模式（目标位置 → PD控制输出力矩）
            allegro_hand_dof_props['driveMode'][i]         = gymapi.DOF_MODE_POS
            allegro_hand_another_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS

            if i < 6:
                # ── 机械臂关节（前6个DOF）：较高刚度，仅设刚度（无速度/力矩限制）
                allegro_hand_dof_props['stiffness'][i]         = self.stiffness[i]
                allegro_hand_another_dof_props['stiffness'][i] = self.stiffness[i]
            else:
                # ── 手指关节（后16个DOF）：较低刚度+阻尼+速度/力矩限制
                allegro_hand_dof_props['velocity'][i] = 3.0    # 最大关节速度 3 rad/s
                allegro_hand_dof_props['stiffness'][i] = 30    # 位置控制刚度（较柔软）
                allegro_hand_dof_props['effort'][i] = 5        # 最大力矩 5 N·m
                allegro_hand_dof_props['damping'][i] = 1       # 阻尼系数（D增益）
                allegro_hand_another_dof_props['velocity'][i] = 3.0
                allegro_hand_another_dof_props['stiffness'][i] = 30
                allegro_hand_another_dof_props['effort'][i] = 5
                allegro_hand_another_dof_props['damping'][i] = 1

        # 将列表转为GPU张量
        self.actuated_dof_indices        = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)
        self.a_allegro_hand_dof_lower_limits = to_torch(self.a_allegro_hand_dof_lower_limits, device=self.device)
        self.a_allegro_hand_dof_upper_limits = to_torch(self.a_allegro_hand_dof_upper_limits, device=self.device)
        self.allegro_hand_dof_default_pos  = to_torch(self.allegro_hand_dof_default_pos, device=self.device)
        self.allegro_hand_dof_default_vel  = to_torch(self.allegro_hand_dof_default_vel, device=self.device)

        # ── 8.4 初始姿态设置 ──────────────────────────────────────────────────
        # 物体资产选项（普通版，受重力）
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500   # 密度 500 kg/m³（比水轻，类似塑料）

        # 物体半径（球形物体）
        self.object_radius = 0.06
        # 创建球形物体资产（半径0.12m，受重力）
        object_asset = self.gym.create_sphere(self.sim, 0.12, object_asset_options)
        # 目标标记（小球，半径0.04m，无重力）
        object_asset_options.disable_gravity = True
        goal_asset = self.gym.create_sphere(self.sim, 0.04, object_asset_options)

        # ── 右手（接手）初始姿态 ──────────────────────────────────────────────
        allegro_hand_start_pose = gymapi.Transform()
        # 位置：沿up_axis方向0.2m处（即z=0.2m）
        allegro_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.2, self.up_axis_idx))
        # 旋转：绕z轴旋转-π/2（≈-90°），使手面向y轴负方向（朝向投手）
        allegro_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, -1.56921)

        # ── 左手（投手）初始姿态 ──────────────────────────────────────────────
        allegro_another_hand_start_pose = gymapi.Transform()
        # 位置：y=-1.35m（距离接手1.35m），z=0.2m
        allegro_another_hand_start_pose.p = gymapi.Vec3(0, -1.35, 0.2)
        # 旋转：绕z轴+π/2（≈+90°），使手面向y轴正方向（朝向接手）
        allegro_another_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.57079)

        # ── 物体初始位置 ──────────────────────────────────────────────────────
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = allegro_hand_start_pose.p.x  # 与接手x位置对齐
        pose_dy, pose_dz = -0.22, 0.15  # 相对偏移（y方向前移0.22m，z方向上移0.15m）
        object_start_pose.p.y = allegro_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = allegro_hand_start_pose.p.z + pose_dz
        # 最终物体位置（硬编码覆盖）
        object_start_pose.p = gymapi.Vec3(0.025, -0.38, 0.449)

        if self.object_type == "pen":
            # pen物体需要更精确的初始高度
            object_start_pose.p.z = allegro_hand_start_pose.p.z + 0.02

        # ── 目标标记初始位置 ──────────────────────────────────────────────────
        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.)  # 目标与物体初始位置的偏移
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z],
            device=self.device
        )
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement  # 目标初始位置
        goal_start_pose.p.z -= 0.0  # 微调z轴（当前为0）

        # ── 8.5 计算聚合模式下的最大刚体数 ──────────────────────────────────
        # 聚合模式（aggregation）将同一环境的actor合并管理，提高碰撞检测效率
        max_agg_bodies = self.num_allegro_hand_bodies * 2 + 2 + 10  # 两只手+物体+目标+余量
        max_agg_shapes = self.num_allegro_hand_shapes * 2 + 2 + 10

        # ── 8.6 初始化各列表（每个环境填充一项）─────────────────────────────
        self.allegro_hands = []          # 手的actor句柄
        self.envs = []                   # 环境句柄
        self.object_init_state = []      # 物体初始状态（13维）
        self.hand_start_states = []      # 手的初始状态
        self.hand_indices = []           # 右手的全局actor索引
        self.another_hand_indices = []   # 左手的全局actor索引
        self.fingertip_indices = []      # 指尖索引（预留）
        self.object_indices = []         # 物体的全局actor索引
        self.goal_object_indices = []    # 目标标记的全局actor索引
        self.predict_goal_object_indices = []  # 预测标记的全局actor索引

        # ── 8.7 逐环境创建 ───────────────────────────────────────────────────
        for i in range(num_envs):
            # 创建一个环境实例（Isaac Gym隔离的物理空间）
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                # 开始聚合（将本环境的所有actor打包管理）
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # ── 创建两只手的actor ────────────────────────────────────────────
            # collision_filter=-1：使用MJCF/URDF中定义的自碰撞过滤规则
            # 右手（接手）
            allegro_hand_actor = self.gym.create_actor(
                env_ptr, allegro_hand_asset, allegro_hand_start_pose,
                "hand", i, -1, 0
            )
            # 左手（投手）
            allegro_hand_another_actor = self.gym.create_actor(
                env_ptr, allegro_hand_another_asset, allegro_another_hand_start_pose,
                "another_hand", i, -1, 0
            )

            # 记录手的初始13维状态 [px,py,pz, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
            self.hand_start_states.append([
                allegro_hand_start_pose.p.x, allegro_hand_start_pose.p.y, allegro_hand_start_pose.p.z,
                allegro_hand_start_pose.r.x, allegro_hand_start_pose.r.y, allegro_hand_start_pose.r.z,
                allegro_hand_start_pose.r.w,
                0, 0, 0, 0, 0, 0  # 初始速度全零
            ])

            # 应用DOF属性（刚度、阻尼等）
            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_actor, allegro_hand_dof_props)
            # 获取右手在仿真全局的actor索引（用于后续张量索引）
            hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_another_actor, allegro_hand_another_dof_props)
            another_hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_another_actor, gymapi.DOMAIN_SIM)
            self.another_hand_indices.append(another_hand_idx)

            # 刚体颜色/纹理随机化（暂时预留，未实际使用）
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, allegro_hand_actor)
            hand_rigid_body_index = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]

            # ── 创建操作物体 ──────────────────────────────────────────────────
            # 按环境索引轮流选择物体种类（i % 物体种数）
            index = i % len(self.used_training_objects)
            select_obj = self.used_training_objects[index]
            # 创建物体actor（从asset_dict取对应URDF）
            object_handle = self.gym.create_actor(
                env_ptr, self.object_asset_dict[select_obj]['obj'],
                object_start_pose, "object", i, 0, 0
            )

            # 记录物体初始13维状态
            self.object_init_state.append([
                object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                object_start_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            # 获取物体的全局actor索引
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # 设置物体质量（×1倍，即保持不变）
            lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            for lego_body_prop in lego_body_props:
                lego_body_prop.mass *= 1
            self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, lego_body_props)

            # 设置物体恢复系数为0（完全非弹性碰撞，防止接住后弹飞）
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            for object_shape_prop in object_shape_props:
                object_shape_prop.restitution = 0
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            # 设置手的碰撞恢复系数为0
            hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, allegro_hand_actor)
            for hand_shape_prop in hand_shape_props:
                hand_shape_prop.restitution = 0.
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, hand_shape_props)

            # ── 创建目标标记（goal object，无重力，指示期望接住位置）──────────
            # collision_group = i+num_envs（不与物体发生碰撞，只做可视化）
            goal_handle = self.gym.create_actor(
                env_ptr, self.object_asset_dict[select_obj]['goal'],
                goal_start_pose, "goal_object", i + self.num_envs, 0, 0
            )
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            # ── 创建预测标记（橙色，显示TrajEstimator预测落点）─────────────
            predict_goal_handle = self.gym.create_actor(
                env_ptr, self.object_asset_dict[select_obj]['predict goal'],
                goal_start_pose, "predict_goal_object", i + self.num_envs * 2, 0, 0
            )
            predict_goal_object_idx = self.gym.get_actor_index(env_ptr, predict_goal_handle, gymapi.DOMAIN_SIM)
            self.predict_goal_object_indices.append(predict_goal_object_idx)
            # 设置预测标记颜色为橙色（RGB: 0.8, 0.4, 0.0）
            self.gym.set_rigid_body_color(
                env_ptr, predict_goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.4, 0.)
            )

            # ── 相机传感器（可选）────────────────────────────────────────────
            if self.enable_camera_sensors:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                # 相机位置和目标点（侧视深度相机）
                self.gym.set_camera_location(
                    camera_handle, env_ptr,
                    gymapi.Vec3(0, -0.3, 0.43),  # 相机位置
                    gymapi.Vec3(0, -0.55, 0)      # 相机朝向目标点
                )
                # 获取深度图GPU张量
                camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH
                )
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                # 相机视图矩阵逆矩阵（用于将相机坐标转换为世界坐标）
                cam_vinv = torch.inverse(
                    torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle))
                ).to(self.device)
                # 相机投影矩阵（用于深度图到点云的反投影）
                cam_proj = torch.tensor(
                    self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle),
                    device=self.device
                )

            # 设置非方块物体为蓝色（区别于默认颜色）
            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                # 结束聚合
                self.gym.end_aggregate(env_ptr)

            # 记录环境和手的句柄
            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_hand_actor)

            if self.enable_camera_sensors:
                # 记录相机相关数据（每个环境一组）
                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)

        # ── 8.8 接触传感器句柄（仅取最后一个环境的传感器作为代表）────────────
        # 注意：只取最后一个env_ptr（所有环境传感器命名相同，索引规律相同）
        sensor_handles = [
            self.gym.find_actor_rigid_body_handle(
                env_ptr, allegro_hand_another_actor, sensor_name
            )
            for sensor_name in self.contact_sensor_names
        ]
        # 转为long型张量，用于后续contact_tensor的索引
        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)

        # ── 8.9 物体刚体质量（从最后一个环境读取）───────────────────────────
        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        # ── 8.10 张量化初始状态（列表 → GPU张量）────────────────────────────
        # 物体初始13维状态 → [num_envs, 13]
        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)

        # 目标状态从物体初始状态clone
        self.goal_states     = self.object_init_state.clone()
        self.goal_pose       = self.goal_states[:, 0:7]   # 目标姿态（位置+四元数）
        self.goal_pos        = self.goal_states[:, 0:3]   # 目标位置
        self.goal_rot        = self.goal_states[:, 3:7]   # 目标旋转
        self.goal_init_state = self.goal_states.clone()   # 保存目标初始状态

        # 手的初始13维状态 → [num_envs, 13]
        self.hand_start_states = to_torch(
            self.hand_start_states, device=self.device
        ).view(self.num_envs, 13)

        # 索引列表 → long型GPU张量
        self.hand_indices              = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.another_hand_indices      = to_torch(self.another_hand_indices, dtype=torch.long, device=self.device)
        self.object_indices            = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices       = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.predict_goal_object_indices = to_torch(self.predict_goal_object_indices, dtype=torch.long, device=self.device)

        # ── 8.11 初始化其他控制变量 ───────────────────────────────────────────
        self.init_object_tracking = True      # 物体轨迹追踪初始化标志
        self.test_for_robot_controller = False # 机器人控制器测试模式（未使用）

        # PD控制器增益（用于某些备用控制模式）
        self.p_gain_val = 100.0               # P增益（位置）
        self.d_gain_val = 4.0                 # D增益（速度）
        # 所有环境的PD增益张量：[num_envs, num_dofs×2]
        self.p_gain = torch.ones(
            (self.num_envs, self.num_allegro_hand_dofs * 2),
            device=self.device, dtype=torch.float
        ) * self.p_gain_val
        self.d_gain = torch.ones(
            (self.num_envs, self.num_allegro_hand_dofs * 2),
            device=self.device, dtype=torch.float
        ) * self.d_gain_val

        # PD控制器内部状态缓冲区
        self.pd_previous_dof_pos = torch.zeros(
            (self.num_envs, self.num_allegro_hand_dofs * 2),
            device=self.device, dtype=torch.float
        ) * self.p_gain_val
        self.pd_dof_pos = torch.zeros(
            (self.num_envs, self.num_allegro_hand_dofs * 2),
            device=self.device, dtype=torch.float
        ) * self.p_gain_val

        # 调试数据记录列表
        self.debug_target = []   # 调试目标位置记录
        self.debug_qpos = []     # 调试关节角度记录

    # ==========================================================================
    # 9. 目标重置
    # ==========================================================================

    def reset_target_pose(self, env_ids, apply_reset=False):
        """
        重置指定环境的目标接住位置（goal object）

        目标位置带随机扰动，使接手不能简单地记忆固定位置，
        而必须根据物体飞行轨迹动态调整接球位置。

        参数：
            env_ids     (Tensor): 需要重置目标的环境索引
            apply_reset (bool):  是否立即将状态写入仿真（单独调用goal reset时为True）
        """
        # 生成4个随机数（用于位置和旋转随机化）
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        # ── 重置目标位置为初始值 ──────────────────────────────────────────────
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]

        # ── 添加位置随机扰动 ──────────────────────────────────────────────────
        # x轴随机扰动 ±5cm
        self.goal_states[env_ids, 0] += rand_floats[:, 0] * 0.05
        # y轴：接手附近位置（-0.55m ± 5cm），接手在y≈-1.35m，物体飞行到y≈-0.55m处
        self.goal_states[env_ids, 1] -= 0.55 + rand_floats[:, 1] * 0.05
        # z轴：上移0.1m（让接住位置略高于初始位置）
        self.goal_states[env_ids, 2] += 0.1

        # ── 更新根状态张量中目标的位置 ──────────────────────────────────────
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = \
            self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor

        # ── 随机化目标朝向 ────────────────────────────────────────────────────
        # 用欧拉角生成随机四元数（±π随机旋转）
        quat = quat_from_euler_xyz(
            torch.sign(rand_floats[:, 0]) * 3.1415,
            torch.sign(rand_floats[:, 1]) * 3.1415 + 1.571,
            torch.sign(rand_floats[:, 2]) * 3.14
        )
        self.root_state_tensor[self.goal_object_indices[env_ids], 3] = quat[:, 0]
        self.root_state_tensor[self.goal_object_indices[env_ids], 4] = quat[:, 1]
        self.root_state_tensor[self.goal_object_indices[env_ids], 5] = quat[:, 2]
        self.root_state_tensor[self.goal_object_indices[env_ids], 6] = quat[:, 3]
        # 清零速度
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = \
            torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            # 仅重置goal时，立即将状态写入GPU仿真（indexed版本只更新指定actor）
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(goal_object_indices),
                len(env_ids)
            )
        # 清零goal_reset标志
        self.reset_goal_buf[env_ids] = 0

    # ==========================================================================
    # 10. 环境重置
    # ==========================================================================

    def reset(self, env_ids, goal_env_ids):
        """
        完整重置指定环境（episode结束时调用）

        重置内容：
            1. 域随机化（如果启用）
            2. 目标位置重置
            3. 物体位置/旋转重置（带随机噪声）
            4. 两只手的关节角度/速度重置到默认姿态
            5. 清零所有统计缓冲区
            6. 将状态写入GPU仿真

        参数：
            env_ids      (Tensor): 需要完整重置的环境索引
            goal_env_ids (Tensor): 需要重置目标的环境索引
        """
        # ── 10.1 域随机化 ─────────────────────────────────────────────────────
        if self.randomize:
            # 随机化物理参数（质量、摩擦系数、关节动力学参数等）
            self.apply_randomizations(self.randomization_params)

        # 随机化扰动方向（用于后续force perturbation测试）
        self.perturb_direction[env_ids] = torch_rand_float(
            -1, 1, (len(env_ids), 6), device=self.device
        ).squeeze(-1)

        # ── 10.2 生成随机数 ───────────────────────────────────────────────────
        # 为位置、角度等随机化生成足够多的随机数
        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), self.num_allegro_hand_dofs * 2 + 5), device=self.device
        )

        # ── 10.3 重置目标位置 ─────────────────────────────────────────────────
        self.reset_target_pose(env_ids)  # 不立即apply（后面统一写入）

        # ── 10.4 重置物体状态 ─────────────────────────────────────────────────
        # 从初始状态恢复物体（位置、旋转、速度）
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        # 添加位置噪声（x,y方向）
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = \
            self.object_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        # 添加z方向噪声
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = \
            self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        # 随机化物体初始旋转
        quat = quat_from_euler_xyz(
            torch.sign(rand_floats[:, 3]) * 3.1415,
            torch.sign(rand_floats[:, 4]) * 3.1415 + 1.571,
            torch.sign(rand_floats[:, 5]) * 3.14
        )
        self.root_state_tensor[self.object_indices[env_ids], 3] = quat[:, 0]
        self.root_state_tensor[self.object_indices[env_ids], 4] = quat[:, 1]
        self.root_state_tensor[self.object_indices[env_ids], 5] = quat[:, 2]
        self.root_state_tensor[self.object_indices[env_ids], 6] = quat[:, 3]
        # 清零物体速度（静止开始）
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = \
            torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        # 记录物体初始位姿用于开环控制对比
        self.object_pose_for_open_loop[env_ids] = \
            self.root_state_tensor[self.object_indices[env_ids], 0:7].clone()

        # 合并所有需要更新的object类actor索引
        object_indices = torch.unique(torch.cat([
            self.object_indices[env_ids],
            self.goal_object_indices[env_ids],
            self.predict_goal_object_indices[env_ids],
            self.goal_object_indices[goal_env_ids]
        ]).to(torch.int32))

        # ── 10.5 重置两只手的关节状态 ────────────────────────────────────────
        pos         = self.allegro_hand_default_dof_pos          # 接手默认位姿
        another_pos = self.another_allegro_hand_default_dof_pos  # 投手默认位姿

        # 直接写入DOF位置缓冲区
        self.allegro_hand_dof_pos[env_ids, :]         = pos
        self.allegro_hand_another_dof_pos[env_ids, :] = another_pos

        # 写入关节速度（默认速度+随机噪声）
        self.allegro_hand_dof_vel[env_ids, :] = \
            self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_allegro_hand_dofs:
                                                      5 + self.num_allegro_hand_dofs * 2]
        self.allegro_hand_another_dof_vel[env_ids, :] = \
            self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_allegro_hand_dofs:
                                                      5 + self.num_allegro_hand_dofs * 2]

        # 同步控制目标缓冲区（prev_targets 和 cur_targets 也重置为默认位姿）
        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs]  = pos
        self.prev_targets[env_ids, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs * 2] = another_pos
        self.cur_targets[env_ids, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs * 2] = another_pos

        # ── 10.6 将重置状态写入GPU仿真 ───────────────────────────────────────
        hand_indices         = self.hand_indices[env_ids].to(torch.int32)
        another_hand_indices = self.another_hand_indices[env_ids].to(torch.int32)
        all_hand_indices     = torch.unique(
            torch.cat([hand_indices, another_hand_indices]).to(torch.int32)
        )

        # 写入位置控制目标（即将执行的关节目标角度）
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices)
        )

        # 合并所有需要更新的actor索引
        all_indices = torch.unique(
            torch.cat([all_hand_indices, object_indices]).to(torch.int32)
        )

        # 写入DOF状态（关节角度和速度）
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices)
        )

        # 写入actor根状态（位置、旋转、速度）
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(all_indices),
            len(all_indices)
        )

        # ── 10.7 清零所有统计缓冲区 ──────────────────────────────────────────
        self.progress_buf[env_ids]      = 0   # 步数计数器归零
        self.reset_buf[env_ids]         = 0   # reset标志清零
        self.successes[env_ids]         = 0   # 成功次数清零
        self.catch_successes[env_ids]   = 0   # 接住成功标志清零
        self.catch_hold_counter[env_ids] = 0  # 连续接住帧计数清零

        # 本体感知重置为当前关节位置
        self.proprioception_close_loop[env_ids] = \
            self.allegro_hand_dof_pos[env_ids, 0:22].clone()

        # 清零物体位置历史缓冲区（新episode从零开始记录轨迹）
        self.object_state_stack_frames[env_ids] = \
            torch.zeros_like(self.object_state_stack_frames[env_ids])

        # 重置物体轨迹追踪标志
        self.init_object_tracking = True
        # 清除调试线条（如果viewer存在）
        self.gym.clear_lines(self.viewer)

    # ==========================================================================
    # 11. 物理步前处理（动作写入）
    # ==========================================================================

    def pre_physics_step(self, actions):
        """
        物理仿真步之前的处理：将策略输出的动作转换为关节目标并写入仿真

        调用时序：
            MAPPO输出动作 → pre_physics_step → [物理仿真N步] → post_physics_step

        动作空间（44维，多智能体时每个agent 22维）：
            actions[:, 0:6]  → 右手臂关节（actions[:,1]和[:,2]用相对控制）
            actions[:, 6:22] → 右手手指关节（绝对位置控制，缩放到关节限位）
            actions[:, 22:28]→ 左手臂关节（相对控制，步长×0.1）
            actions[:, 28:44]→ 左手手指关节（绝对位置控制）

        参数：
            actions (Tensor): [num_envs, 44]，策略网络输出的归一化动作[-1,1]
        """
        self.actions = actions.clone().to(self.device)

        # ── 11.1 检查哪些环境需要reset ───────────────────────────────────────
        # nonzero() 返回reset_buf=1的环境索引
        env_ids      = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # ── 11.2 处理目标和环境重置 ───────────────────────────────────────────
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            # 只有目标需要reset（无episode结束），直接重置目标位置
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        elif len(goal_env_ids) > 0:
            # 目标reset会在完整reset中一并处理
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            # 完整episode reset
            self.reset(env_ids, goal_env_ids)

        # ── 11.3 计算当前步的关节目标位置 ────────────────────────────────────

        # 右手手指关节（actions[:,6:22]）：缩放到关节角度范围[lower, upper]
        # scale函数：[-1,1] → [lower_limit, upper_limit]
        self.cur_targets[:, self.actuated_dof_indices + 6] = scale(
            self.actions[:, 6:22],
            self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6],
            self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6]
        )

        # 左手手指关节（actions[:,28:44]）：同样缩放到关节范围
        self.cur_targets[:, self.actuated_dof_indices + 28] = scale(
            self.actions[:, 28:44],
            self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6],
            self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6]
        )

        # 右手臂关节1和2（actions[:,1:3]）：相对控制（在前一步目标上增减，步长×2）
        # 注意：步长较大（×2），允许臂关节快速移动
        self.cur_targets[:, [1, 2]] = self.prev_targets[:, [1, 2]] + self.actions[:, [1, 2]] * 2

        # 左手臂关节（actions[:,22:28]）：相对控制，步长×0.1（小步长，精细控制）
        self.cur_targets[:, 22:28] = self.prev_targets[:, 22:28] + self.actions[:, 22:28] * 0.1

        # ── 11.4 动作平滑（移动平均）─────────────────────────────────────────
        # 防止相邻步动作跳变导致机械抖动
        # cur = alpha * cur + (1-alpha) * prev
        # alpha = act_moving_average（配置文件中设置，典型值0.3~0.7）
        self.cur_targets[:, self.actuated_dof_indices + 6] = (
            self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices + 6]
            + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices + 6]
        )
        self.cur_targets[:, self.actuated_dof_indices + 28] = (
            self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices + 22]
            + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices + 22]
        )

        # ── 11.5 关节限位裁剪 ────────────────────────────────────────────────
        # 右手（前22个DOF）
        self.cur_targets[:, 0:22] = tensor_clamp(
            self.cur_targets[:, 0:22],
            self.allegro_hand_dof_lower_limits[:],
            self.allegro_hand_dof_upper_limits[:]
        )
        # 左手（22:44个DOF）
        self.cur_targets[:, 22:44] = tensor_clamp(
            self.cur_targets[:, 22:44],
            self.a_allegro_hand_dof_lower_limits[:],
            self.a_allegro_hand_dof_upper_limits[:]
        )

        # ── 11.6 更新前一步目标并写入仿真 ────────────────────────────────────
        self.prev_targets[:, :] = self.cur_targets[:, :]
        # 将位置控制目标写入GPU仿真（Isaac Gym会根据目标角度计算PD力矩）
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets)
        )

        # ── 11.7 更新预测目标标记（橙色球）的位置 ────────────────────────────
        # 预测目标的位置 = TrajEstimator预测落点 + 右手底座位置
        # 随着episode进行，逐渐向真实目标位置靠拢（线性插值动画效果）
        self.root_state_tensor[self.predict_goal_object_indices, 0:3] = (
            self.predict_pose[:, 0:3].detach()
            + self.root_state_tensor[self.hand_indices, 0:3]
            - (
                (self.predict_pose[:, 0:3].detach() + self.root_state_tensor[self.hand_indices, 0:3])
                - self.goal_pos
              ) * torch.clamp(self.progress_buf[0] * random.random() / 10, 0, 1)
            # progress_buf[0]*random/10：随episode进行从0到1的权重，使预测球逐渐收敛到真实目标
        )
        # 预测标记与物体旋转同步（保持朝向一致）
        self.root_state_tensor[self.predict_goal_object_indices, 3:7] = \
            self.root_state_tensor[self.object_indices, 3:7].detach()

        # ── 11.8 更新所有object类actor的根状态 ───────────────────────────────
        object_indices = torch.unique(torch.cat([
            self.object_indices,
            self.goal_object_indices,
            self.predict_goal_object_indices
        ]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices.to(torch.int32)),
            len(object_indices.to(torch.int32))
        )

        # ── 11.9 定期保存TrajEstimator模型 ──────────────────────────────────
        # 每200个episode保存一次（total_steps是环境数×步数，故除以200×episode长度）
        if self.total_steps % (200 * (self.max_episode_length - 1)) == 0:
            iter = int(self.total_steps / (200 * (self.max_episode_length - 1)))
            if not self.is_test:
                torch.save(
                    self.traj_estimator.state_dict(),
                    self.traj_estimator_save_path + "/model.pt"
                )

        # ── 11.10 外力扰动（当前已关闭）────────────────────────────────────
        self.apply_force = False   # 开关：False=不施加扰动
        if self.apply_force == True:
            # 清零力张量
            self.rb_forces[:] = torch.zeros_like(self.rb_forces)

            # 找出物体在有效范围内的环境索引
            self.apply_force_env_id = torch.where(
                self.root_state_tensor[self.object_indices, 1] < 10.1,
                torch.where(
                    self.root_state_tensor[self.object_indices, 1] > -10.9, 1, 0
                ), 0
            ).nonzero(as_tuple=False).squeeze(-1)

            # 设置扰动方向（z轴向上推力）
            self.perturb_direction[self.apply_force_env_id, 0] = 0
            self.perturb_direction[self.apply_force_env_id, 1] = 0
            self.perturb_direction[self.apply_force_env_id, 2] = 1

            # 施加1.5N的推力
            self.rb_forces[self.apply_force_env_id, self.object_rb_handles, 0:3] = \
                torch.ones(
                    self.rb_forces[self.apply_force_env_id, self.object_rb_handles, 0:3].shape,
                    device=self.device
                ) * 1.5 * self.perturb_direction[self.apply_force_env_id, 0:3].squeeze(-1)
            # 将力应用到仿真（局部坐标系）
            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE
            )

            # 通过颜色变化可视化是否施加了扰动（红=施加，蓝=未施加）
            if 0 in self.apply_force_env_id:
                self.gym.set_rigid_body_color(
                    self.envs[0], self.object_indices[0], 0,
                    gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.3, 0.3)   # 红色
                )
            else:
                self.gym.set_rigid_body_color(
                    self.envs[0], self.object_indices[0], 0,
                    gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 1)   # 蓝色
                )

    # ==========================================================================
    # 12. 物理步后处理
    # ==========================================================================

    def post_physics_step(self):
        """
        每次物理仿真步之后调用：更新进度、计算观测和奖励

        调用时序：
            [物理仿真] → post_physics_step → [策略推断] → pre_physics_step
        """
        # 更新episode步数进度（+1每个物理步）
        self.progress_buf += 1
        # 更新域随机化计数器（某些随机化按间隔触发）
        self.randomize_buf += 1

        # 计算新的观测量（刷新状态、填充obs_buf和states_buf）
        self.compute_observations()
        # 计算奖励、检测reset条件、更新统计
        self.compute_reward(self.actions)

        # ── 调试可视化（仅在viewer存在且debug_viz开启时）─────────────────────
        if self.viewer and self.debug_viz:
            # 清除上一帧的调试线条
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                # 在拇指指尖处绘制坐标轴（用于调试接手手指位置）
                self.add_debug_lines(
                    self.envs[i],
                    self.allegro_hand_another_thmub_pos[i],
                    self.allegro_hand_another_thmub_rot[i],
                    line_width=2
                )

    def add_debug_lines(self, env, pos, rot, line_width=1):
        """
        在指定位置绘制RGB三轴坐标系（调试用）

        绘制3条线段：
            红线 → 局部x轴方向（0.2m长）
            绿线 → 局部y轴方向（0.2m长）
            蓝线 → 局部z轴方向（0.2m长）

        参数：
            env       : Isaac Gym环境句柄
            pos       (Tensor): [3]，坐标轴原点（世界坐标）
            rot       (Tensor): [4]，旋转四元数（决定局部坐标轴方向）
            line_width (int):   线宽（像素）
        """
        # 沿局部x/y/z轴各延伸0.2m，得到轴端点的世界坐标
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()  # 原点

        # 绘制x轴（红色 RGB: 0.85, 0.1, 0.1）
        self.gym.add_lines(self.viewer, env, line_width,
                           [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]],
                           [0.85, 0.1, 0.1])
        # 绘制y轴（绿色 RGB: 0.1, 0.85, 0.1）
        self.gym.add_lines(self.viewer, env, line_width,
                           [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]],
                           [0.1, 0.85, 0.1])
        # 绘制z轴（蓝色 RGB: 0.1, 0.1, 0.85）
        self.gym.add_lines(self.viewer, env, line_width,
                           [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]],
                           [0.1, 0.1, 0.85])


# =============================================================================
# 13. JIT加速函数：核心奖励计算
# =============================================================================

@torch.jit.script
def compute_hand_reward(
    rew_buf,              # [num_envs] 奖励缓冲区（将被更新）
    reset_buf,            # [num_envs] episode重置标志
    reset_goal_buf,       # [num_envs] 目标重置标志
    progress_buf,         # [num_envs] 当前episode内步数
    successes,            # [num_envs] 旋转对齐成功次数
    consecutive_successes,# [1]        滑动平均连续成功次数
    max_episode_length: float,  # episode最大步数
    object_pos,           # [num_envs, 3] 物体位置
    object_rot,           # [num_envs, 4] 物体旋转（四元数）
    target_pos,           # [num_envs, 3] 目标位置（goal）
    target_rot,           # [num_envs, 4] 目标旋转
    allegro_left_hand_pos,  # [num_envs, 3] 接手（左手）掌心位置
    allegro_right_hand_pos, # [num_envs, 3] 投手（右手）位置
    allegro_another_hand_thmub_pos, # [num_envs, 3] 左手拇指位置
    aux_up_pos,           # [num_envs, 3] 辅助参考点
    object_vel,           # [num_envs, 3] 物体速度
    leeft_hand_ee_rot,    # [num_envs, 4] 接手末端旋转
    dist_reward_scale: float,  # 距离奖励系数
    rot_reward_scale: float,   # 旋转奖励系数
    rot_eps: float,            # 旋转奖励稳定性epsilon
    actions,              # [num_envs, num_actions] 当前动作
    action_penalty_scale: float,  # 动作惩罚系数
    allegro_hand_another_ff_pos,  # [num_envs, 3] 右手食指位置（当前未用于奖励）
    allegro_hand_another_mf_pos,  # [num_envs, 3] 右手中指位置
    allegro_hand_another_rf_pos,  # [num_envs, 3] 右手无名指位置
    allegro_hand_ff_pos,          # [num_envs, 3] 左手食指位置
    allegro_hand_mf_pos,          # [num_envs, 3] 左手中指位置
    allegro_hand_rf_pos,          # [num_envs, 3] 左手无名指位置
    a_hand_palm_pos,      # [num_envs, 3] 接手掌心（未偏移）
    hand_init_qpos,       # [16] 手指默认关节角度（归一化）
    hand_qpos,            # [num_envs, 16] 当前手指关节角度（归一化）
    success_tolerance: float,      # 旋转对齐成功容差（rad）
    reach_goal_bonus: float,       # 到达目标的奖励加成
    fall_dist: float,              # 掉落距离判定阈值
    fall_penalty: float,           # 掉落惩罚
    max_consecutive_successes: int, # 最大连续成功次数（触发reset）
    av_factor: float,              # EMA平滑系数
    ignore_z_rot: bool,            # 是否忽略绕z轴旋转（pen类物体）
    catch_successes,               # [num_envs] 本局是否接住成功
    catch_hold_counter,            # [num_envs] 连续稳定帧计数器
    catch_tolerance: float,        # 接住距离阈值（m）
    catch_hold_steps: int,         # 需要连续稳定的帧数
    catch_vel_tolerance: float,    # 物体速度阈值（m/s，当前未使用）
):
    """
    计算所有并行环境的奖励、重置信号和成功统计

    奖励函数设计：
        1. 主奖励：基于物体位置到目标距离的指数奖励
           reward_dist = exp(-4 × 3 × dist)
           （指数形式使接近目标时奖励急剧增加，形成密集引导）

        2. 速度奖励：物体在有效y坐标范围内，-y方向速度越快越好
           （激励投手用力抛出，而非轻放）

        3. 动作惩罚：动作向量的L2范数平方
           （抑制不必要的大幅动作，鼓励平滑运动）

        4. 到达目标bonus：当物体到达目标位置时给予额外奖励

        5. 掉落处理：物体z<0.15m时episode结束，不给额外掉落惩罚（已注释）

    接住成功判定：
        连续 catch_hold_steps 帧满足以下条件：
            - 接手（左手）到物体的距离 ≤ catch_tolerance（0.15m）
            - 物体高度 > 0.15m（未掉落）
        episode结束时（reset触发时）检查counter，满足则记为成功

    返回：
        reward, resets, goal_resets, progress_buf,
        successes, cons_successes, catch_successes, catch_hold_counter
    """
    # ── 13.1 距离计算 ─────────────────────────────────────────────────────────

    # 物体到目标位置的L2距离（主要优化目标）
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)  # [num_envs]

    # 接手（左手掌心）到物体的距离（用于接住判定）
    left_hand_dist = torch.norm(allegro_left_hand_pos - object_pos, p=2, dim=-1)  # [num_envs]

    # ── 13.2 接住条件判断 ──────────────────────────────────────────────────────
    # 判定条件：
    #   (1) 接手到物体距离 ≤ catch_tolerance（15cm）
    #   (2) 物体高度 > 0.15m（未落地）
    # 注：物体速度阈值判断已注释掉（too strict）
    catch_condition = (left_hand_dist <= catch_tolerance) & (object_pos[:, 2] > 0.15)

    # ── 13.3 连续帧计数器更新 ──────────────────────────────────────────────────
    # 策略：
    #   - 满足接住条件：计数+1
    #   - 不满足（物体离开手/掉落）：立即归零（要求"连续"稳定，不允许断开后重新累积）
    catch_hold_counter = torch.where(
        catch_condition,
        catch_hold_counter + 1,
        torch.zeros_like(catch_hold_counter),
    )

    # ── 13.4 旋转对齐容差调整 ──────────────────────────────────────────────────
    if ignore_z_rot:
        # pen类物体：放宽旋转容差到2倍（因为绕轴旋转对抓握无影响）
        success_tolerance = 2.0 * success_tolerance

    # ── 13.5 旋转对齐距离计算 ──────────────────────────────────────────────────
    # 四元数差：q_diff = object_rot × conj(target_rot)
    # 当两个四元数完全对齐时，q_diff = [0,0,0,1]（单位四元数）
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # 从四元数差的向量部分提取旋转角度（弧度）
    # rot_dist = 2 × arcsin(|q_diff[0:3]|)（由四元数到轴角的换算公式）
    rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    )

    # ── 13.6 奖励项计算 ────────────────────────────────────────────────────────
    dist_rew = goal_dist  # 距离误差（越小越好）

    # 动作惩罚：动作向量L2范数平方（抑制大幅动作）
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # 速度奖励：激励物体以足够速度向-y方向飞行（从投手到接手）
    # clamp限制在[-0.1, 0.1]，防止过大的速度奖励主导总奖励
    object_vel_reward = torch.clamp(-object_vel[:, 1], -0.1, 0.1)
    # 仅在物体y坐标处于 [-0.85, -0.65] 之间时激活速度奖励
    # （物体飞行中段才给速度奖励，防止在起点或落点给奖励）
    object_vel_reward = torch.where(
        object_pos[:, 1] < -0.65,
        torch.where(-0.85 < object_pos[:, 1], object_vel_reward, torch.zeros_like(object_vel_reward)),
        torch.zeros_like(object_vel_reward)
    )

    # ── 13.7 总奖励计算 ────────────────────────────────────────────────────────
    # 公式：exp(-12 × goal_dist) + vel_reward - 0.001 × action_penalty
    # 指数距离奖励：距离接近时奖励急剧增加（在0.1m时奖励≈0.3，在0时奖励=1）
    reward = (torch.exp(-4 * (3 * dist_rew)) + object_vel_reward) - 0.001 * action_penalty

    # ── 13.8 旋转对齐成功判断（目标reset）────────────────────────────────────
    # 当goal_dist=0时触发目标reset（实际上很难达到，goal是固定的接住位置）
    goal_resets = torch.where(
        torch.abs(goal_dist) <= 0,
        torch.ones_like(reset_goal_buf),
        reset_goal_buf
    )
    # 累加旋转对齐成功次数
    successes = successes + goal_resets

    # 到达目标位置时给予bonus
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # ── 13.9 Episode终止条件 ──────────────────────────────────────────────────
    # 物体掉落（z < 0.15m）→ episode结束
    resets = torch.where(object_pos[:, 2] <= 0.15, torch.ones_like(reset_buf), reset_buf)

    if max_consecutive_successes > 0:
        # 当连续成功次数达到上限时，重置progress（继续episode直到超时）
        progress_buf = torch.where(
            torch.abs(rot_dist) <= success_tolerance,
            torch.zeros_like(progress_buf),
            progress_buf
        )
        # 连续成功次数达上限 → episode结束
        resets = torch.where(
            successes >= max_consecutive_successes,
            torch.ones_like(resets),
            resets
        )

    # 超过最大步数 → episode结束
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    if max_consecutive_successes > 0:
        # 超时未完成目标 → 额外惩罚（鼓励在有限时间内完成任务）
        reward = torch.where(
            progress_buf >= max_episode_length,
            reward + 0.5 * fall_penalty,
            reward
        )

    # ── 13.10 接住成功判定（仅在episode结束时评估）────────────────────────────
    # 设计原理：
    #   - 在 resets=1 的帧（episode刚结束），检查 catch_hold_counter 是否 >= catch_hold_steps
    #   - 掉落reset（z<0.15m）：catch_condition=False → counter已归零 → 判为失败✓
    #   - 超时reset（物体稳定在手中）：counter仍在累积 → 可能满足 → 判为成功✓
    #   - 非reset帧：保持 catch_successes 原值（单局内不重复计数）
    catch_successes = torch.where(
        resets == 1,      # 本帧是否触发reset
        torch.where(
            catch_hold_counter >= catch_hold_steps,   # 接住计数是否达标
            torch.ones_like(catch_successes),         # 达标 → 成功（1）
            torch.zeros_like(catch_successes),        # 未达标 → 失败（0）
        ),
        catch_successes,  # 非reset帧：保持不变
    )

    # ── 13.11 滑动平均连续成功次数更新 ───────────────────────────────────────
    num_resets = torch.sum(resets)  # 本批次reset的环境数
    # 本批次reset环境中累积的旋转对齐成功次数
    finished_cons_successes = torch.sum(successes * resets.float())

    # EMA更新：cons = av_factor × (成功次数/reset数) + (1-av_factor) × 旧cons
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes  # 没有reset时不更新
    )

    return reward, resets, goal_resets, progress_buf, successes, cons_successes, catch_successes, catch_hold_counter


# =============================================================================
# 14. JIT加速函数：旋转随机化工具
# =============================================================================

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """
    生成随机旋转四元数（绕x轴和y轴各随机旋转）

    参数：
        rand0 (Tensor): [num_envs]，绕x轴旋转的随机系数（范围[-1,1]，乘以π得旋转角）
        rand1 (Tensor): [num_envs]，绕y轴旋转的随机系数
        x_unit_tensor (Tensor): [num_envs, 3]，x轴单位向量
        y_unit_tensor (Tensor): [num_envs, 3]，y轴单位向量

    返回：
        Tensor [num_envs, 4]：随机旋转四元数（绕x轴旋转 × 绕y轴旋转）
    """
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),  # 绕x轴随机旋转 [-π, π]
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)   # 绕y轴随机旋转 [-π, π]
    )


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    """
    生成笔类物体的随机旋转四元数（限制最大偏转角，防止笔竖直）

    参数：
        rand0     (Tensor): [num_envs]，随机系数
        rand1     (Tensor): [num_envs]，随机系数
        max_angle (float):  最大偏转角（弧度）
        x_unit_tensor, y_unit_tensor, z_unit_tensor: 单位向量

    返回：
        Tensor [num_envs, 4]：适合笔类物体的随机旋转四元数
    """
    # 绕x轴旋转π/2+随机偏移（使笔从垂直变为水平，再加小扰动）
    # 绕z轴旋转（给笔一个随机的水平朝向）
    rot = quat_mul(
        quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
        quat_from_angle_axis(rand0 * np.pi, z_unit_tensor)
    )
    return rot


# =============================================================================
# 15. 辅助函数：IK控制
# =============================================================================

def orientation_error(desired, current):
    """
    计算四元数旋转误差（用于末端执行器方向控制）

    基于旋转误差公式：
        e = q_r[0:3] × sign(q_r[3])
        其中 q_r = desired × conj(current)

    当 desired ≈ current 时，误差向量趋近于零向量。

    参数：
        desired (Tensor): [num_envs, 4]，期望四元数
        current (Tensor): [num_envs, 4]，当前四元数

    返回：
        Tensor [num_envs, 3]：旋转误差向量（方向误差）
    """
    cc = quat_conjugate(current)                     # 当前四元数的共轭（等价于逆）
    q_r = quat_mul(desired, cc)                      # 相对旋转四元数
    # 取向量部分，并乘以标量部分的符号（保证误差方向一致性）
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik(j_eef, device, dpose, num_envs):
    """
    阻尼最小二乘逆运动学（Damped Least Squares IK）

    求解末端执行器的关节速度，使末端位姿误差最小化：
        Δθ = J^T (J J^T + λ²I)^{-1} Δx

    这是经典的阻尼伪逆法（Levenberg–Marquardt）：
        - λ为阻尼因子，防止雅可比矩阵奇异时解发散
        - 比纯伪逆更稳定，在奇异位形附近仍能给出合理解

    参数：
        j_eef    (Tensor): [num_envs, 6, num_dofs]，末端执行器雅可比矩阵
                           6行 = 3线速度 + 3角速度
        device   (str):    计算设备
        dpose    (Tensor): [num_envs, 6, 1]，末端位姿误差（位置误差+旋转误差）
        num_envs (int):    并行环境数

    返回：
        Tensor [num_envs, num_dofs]：各关节速度增量（用于更新关节目标）
    """
    # 阻尼因子（经验值0.05，越大越稳定但响应越慢）
    damping = 0.05

    # 雅可比转置：[num_envs, num_dofs, 6]
    j_eef_T = torch.transpose(j_eef, 1, 2)

    # 阻尼矩阵：λ²I，形状 [6, 6]
    lmbda = torch.eye(6, device=device) * (damping ** 2)

    # 阻尼最小二乘求解：Δθ = J^T (J J^T + λ²I)^{-1} Δx
    # torch.inverse：矩阵求逆（J J^T + λ²I 为6×6矩阵，可逆性有保证）
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)

    return u   # [num_envs, num_dofs]：各关节速度增量
