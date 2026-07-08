# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from matplotlib.pyplot import axis
import numpy as np
import os
import random
import torch
import pickle

from torch.utils.tensorboard import SummaryWriter

from utils.torch_jit_utils import *
# from isaacgym.torch_utils import *

from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

import matplotlib.pyplot as plt
from PIL import Image as Im
import cv2
from torch import nn
import torch.nn.functional as F

# 轨迹预测器模型，继承自nn.Module
class TrajEstimator(nn.Module):
    # 初始化方法，设置输入维度和输出维度
    def __init__(self, input_dim, output_dim) :
        super(TrajEstimator, self).__init__()
        # 定义神经网络结构，四层全连接层
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, output_dim)

        # 定义激活函数为ELU
        self.activate_func = nn.ELU()

    # 前向传播函数
    def forward(self, inputs):
        # 依次经过各层和激活函数
        x = self.activate_func(self.linear1(inputs))
        x = self.activate_func(self.linear2(x))
        x = self.activate_func(self.linear3(x))
        # 最后一层输出
        outputs = self.output_layer(x)

        # 返回输出值和最后一层的特征表示
        return outputs, x

# 临时开启梯度计算的上下文管理器
class TemporaryGrad(object):
    # 进入上下文时保存当前的梯度状态并开启梯度计算
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    # 退出上下文时恢复原来的梯度状态
    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.prev)

# Allegro手动态交接任务类，继承自BaseTask
class AllegroHandDynamicHandover(BaseTask):
    # 初始化方法，加载配置、设置模拟器参数和任务属性
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        # 保存配置和参数
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        # 读取任务相关配置
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # 读取奖励相关权重和容差
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        # 定义观测值的缩放因子
        self.vel_obs_scale = 0.2  # 速度观测的缩放因子
        self.force_torque_obs_scale = 10.0  # 力矩观测的缩放因子

        # 读取重置时的噪声设置
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        # 读取控制和显示相关的配置
        self.allegro_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # 读取训练过程监控相关的配置
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        # 计算并更新 episode 最大长度
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        # 设置物体类型及对应的资源文件
        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen", "ycb/banana", "ycb/can", "ycb/mug", "ycb/brick"]

        self.ignore_z = (self.object_type == "pen")

        # 定义可用的物体及其资源路径字典
        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "ball": "urdf/objects/ball.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "ycb/banana": "urdf/ycb/011_banana/011_banana.urdf",
            "ycb/can": "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
            "ycb/mug": "urdf/ycb/025_mug/025_mug.urdf",
            "ycb/brick": "urdf/ycb/061_foam_brick/061_foam_brick.urdf"
        }

        # 更新为包含更多Binghao提供的物体
        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "obj0": "urdf/binghao_obj/objects/obj0.urdf",
            "obj1": "urdf/binghao_obj/objects/obj1.urdf",
            "obj2": "urdf/binghao_obj/objects/obj2.urdf",
            # "obj3": "urdf/binghao_obj/objects/obj3.urdf",
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

        # 更新为包含更多新奇物体(novel objects)
        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "obj0": "urdf/binghao_obj/objects/obj0.urdf",
            "obj1": "urdf/binghao_obj/objects/obj1.urdf",
            "obj2": "urdf/binghao_obj/objects/obj2.urdf",
            # "obj3": "urdf/binghao_obj/objects/obj3.urdf",
            "obj4": "urdf/binghao_obj/objects/obj4.urdf",
            # "obj5": "urdf/binghao_obj/objects/obj5.urdf",
            "obj6": "urdf/binghao_obj/objects/obj6.urdf",
            "obj7": "urdf/binghao_obj/objects/obj7.urdf",
            # "obj8": "urdf/binghao_obj/objects/obj8.urdf",
            "obj9": "urdf/binghao_obj/objects/obj9.urdf",
            "obj10": "urdf/binghao_obj/objects/obj10.urdf",
            "ball": "urdf/binghao_obj/objects/ball.urdf",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",

            # "novel_obj0": "urdf/binghao_obj/objects/final_calibration.urdf",
            "novel_obj1": "urdf/binghao_obj/objects/final_obj1.urdf",
            "novel_obj2": "urdf/binghao_obj/objects/final_obj2_cross.urdf",
            "novel_obj3": "urdf/binghao_obj/objects/final_obj3_flatcan.urdf",
            "novel_obj4": "urdf/binghao_obj/objects/final_obj4_t.urdf",
            "novel_obj5": "urdf/binghao_obj/objects/final_obj5_s.urdf",
            "novel_obj6": "urdf/binghao_obj/objects/final_obj6_ball.urdf",
            "novel_obj7": "urdf/binghao_obj/objects/final_obj7_ball_flatter.urdf",
            "novel_obj8": "urdf/binghao_obj/objects/final_obj8_cylinder.urdf",
            "novel_obj9": "urdf/binghao_obj/objects/final_obj9_irregular_cube.urdf",
            "novel_obj10": "urdf/binghao_obj/objects/final_obj10_cube_stair.urdf",
            "novel_obj11": "urdf/binghao_obj/objects/final_obj11_cube_ir2.urdf",
            "novel_obj12": "urdf/binghao_obj/objects/final_obj12_cube_extrude1.urdf",
            "novel_obj13": "urdf/binghao_obj/objects/final_obj13_cube_extrude2.urdf",
            "novel_obj14": "urdf/binghao_obj/objects/final_obj14_cathead.urdf",
        }


        # 设置实际用于训练的物体列表
        self.used_training_objects = ["pen", "ball", "block", "obj0", "obj1", "obj2", "obj4", "obj6", "obj7", "obj9", "obj10"]
        # self.used_training_objects = ["novel_obj1", "novel_obj2", "novel_obj3", "novel_obj4", "novel_obj5", "novel_obj6", "novel_obj7", "novel_obj8", "novel_obj9", "novel_obj10", "novel_obj11", "novel_obj12", "novel_obj13", "novel_obj14"]

        # self.used_training_objects = ["ball", "obj0", "obj1", "obj2", "obj4", "obj6", "obj7", "obj9", "obj10",
        #                               "novel_obj1", "novel_obj2", "novel_obj3", "novel_obj4", "novel_obj5", "novel_obj6",
        #                               "novel_obj7", "novel_obj8", "novel_obj9", "novel_obj10", "novel_obj11", "novel_obj12", "novel_obj13", "novel_obj14"]

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        print("Obs type:", self.obs_type)

        self.num_point_cloud_feature_dim = 384
        self.one_frame_num_obs = 300
        self.num_obs_dict = {
            "point_cloud": 111 + self.num_point_cloud_feature_dim * 3,
            "point_cloud_for_distill": 111 + self.num_point_cloud_feature_dim * 3,
            "full_state": 300 * 3,
        }

        self.contact_sensor_names = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr",
                                     "link_5.0_fsr", "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr",
                                     "link_10.0_fsr", "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr"]

        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            # num_states = 215 + 384 * 3
            num_states = 215

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 22

        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 44

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.enable_camera_sensors = self.cfg["env"]["enableCameraSensors"]
        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)
        self.num_envs = cfg["env"]["numEnvs"]

        if self.point_cloud_debug:
            import open3d as o3d
            from utils.o3dviewer import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.o3d_pc = o3d.geometry.PointCloud()
        else :
            self.pointCloudVisualizer = None

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            # cam_pos = gymapi.Vec3(0.9, -0.65, 1.0)
            cam_pos = gymapi.Vec3(4.0, -0.65, 1.0)
            cam_target = gymapi.Vec3(-0.5, -0.65, 0.2)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "another_hand"))

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.another_allegro_hand_default_dof_pos = torch.zeros(self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
        self.another_allegro_hand_default_dof_pos[:6] = torch.tensor([-0.0, -0.09, -0.09, 3.141, 2.00, -1.57], dtype=torch.float, device=self.device)
        self.another_allegro_hand_default_dof_pos[6:] = to_torch([-0.03989830748810656, 1.3495253790945758, 0.8659920759388671, 0.780414711365591,
0.9655586519308622, 1.0139016439397597, 0.8501943208059994, 1.3264760744914152,
                                                        -0.20482272250532974, 1.347864170294202, 0.6030536585610538, 0.9181400800651911,
                                                        -0.21341465375119012, 1.7199185039090872, 1.2686849760515697, 0.8245164874462315, ], dtype=torch.float, device=self.device)

        self.allegro_hand_default_dof_pos = torch.zeros(self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
        # self.allegro_hand_default_dof_pos[:6] = torch.tensor([0, 0, -1, 3.14, 0.57, 3.14], dtype=torch.float, device=self.device)
        self.allegro_hand_default_dof_pos[:6] = torch.tensor([-0.0, -0.09, -0.09, 3.141, 2.00, -1.57], dtype=torch.float, device=self.device)
        # self.allegro_hand_default_dof_pos[6:] = to_torch([0.0, -0.174, 0.785, 0.785,
        #                                     0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)

        ## default qpos
        self.allegro_hand_default_dof_pos[6:] = to_torch([-0.03989830748810656, 1.3495253790945758, 0.8659920759388671, 0.780414711365591,
0.9655586519308622, 1.0139016439397597, 0.8501943208059994, 1.3264760744914152,
                                                        -0.20482272250532974, 1.347864170294202, 0.6030536585610538, 0.9181400800651911,
                                                        -0.21341465375119012, 1.7199185039090872, 1.2686849760515697, 0.8245164874462315, ], dtype=torch.float, device=self.device)

        ## hand put
        # self.allegro_hand_default_dof_pos[6:] = to_torch([0,0,0.7,1.2,0,0.7,0.3,1.2, 0,0,0.7,1.2,0,0,0.7,1.2,], dtype=torch.float, device=self.device)

        ## hand grip
        # self.allegro_hand_default_dof_pos[6:] = to_torch([0,0.5,0.7,1.2,1.57,0.3,1.2,0.7,0,0.3,0.7,1.2,0,0.5,0.7,1.2,], dtype=torch.float, device=self.device)


        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.allegro_hand_another_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2]
        self.allegro_hand_another_dof_pos = self.allegro_hand_another_dof_state[..., 0]
        self.allegro_hand_another_dof_vel = self.allegro_hand_another_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()

        self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.object_init_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.catch_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # 本局是否已被接住过（catch）
        self.catch_tolerance = self.cfg["env"].get("catchTolerance", 0.15)  # 判定“接住”的距离阈值(m)，可在cfg里配置
        self.total_catch_successes = 0   # 累计成功接住的局数
        self.total_attempts = 0          # 累计的 episode attempts（完成的局数）
        self.catch_hold_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # 连续满足接住条件的帧数
        self.catch_hold_steps = self.cfg["env"].get("catchHoldSteps", 5)      # 需要连续保持多少帧才算“接稳”，可按控制频率调整（比如60Hz下10帧≈0.17s）
        self.catch_vel_tolerance = self.cfg["env"].get("catchVelTolerance", 1.0)  # 物体速度阈值(m/s)，速度太快说明只是擦过/弹跳，不算稳稳接住
        self.smoothed_episode_success_rate = 0.0

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.object_pose_for_open_loop = torch.zeros_like(self.root_state_tensor[self.object_indices, 0:7])

        self.total_successes = 0
        self.total_resets = 0

        self.state_buf_stack_frames = []
        self.obs_buf_stack_frames = []

        for i in range(3):
            self.obs_buf_stack_frames.append(torch.zeros_like(self.obs_buf[:, 0:self.one_frame_num_obs]))
            self.state_buf_stack_frames.append(torch.zeros_like(self.states_buf[:, 0:215]))

        self.object_seq_len = 20
        self.object_state_stack_frames = torch.zeros((self.num_envs, self.object_seq_len * 3), dtype=torch.float, device=self.device)

        self.proprioception_close_loop = torch.zeros_like(self.allegro_hand_dof_pos[:, 0:22])
        self.another_hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.another_hand_indices[0], "link6", gymapi.DOMAIN_ENV)
        print("another_hand_base_rigid_body_index: ", self.another_hand_base_rigid_body_index)
        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.hand_indices[0], "link6", gymapi.DOMAIN_ENV)
        print("hand_base_rigid_body_index: ", self.hand_base_rigid_body_index)
        # with open("./demo_throw.pkl", "rb") as f:
        #     self.demo_throw = pickle.load(f)

        # print(self.demo_throw)
        # # self.demo_throw = to_torch(self.demo_throw['qpos'], dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        # self.demo_throw = to_torch(self.demo_throw['qpos'], dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        object_rb_count = self.gym.get_asset_rigid_body_count(self.object_asset)
        self.object_rb_handles = 46
        self.perturb_direction = torch_rand_float(-1, 1, (self.num_envs, 6), device=self.device).squeeze(-1)

        self.predict_pose = self.goal_init_state[:, 0:3].clone()

        self.algorithm_name = self.cfg["env"]["algorithm_name"]
        self.log_dir = str(
            './logs/allegro_hand_dynamic_handover/{}/success_rate_logs_seed{}'.format(self.algorithm_name, self.cfg["seed"]))
        self.writter = SummaryWriter(self.log_dir)

    # 创建模拟环境
    def create_sim(self):
        # 设置时间步长
        self.dt = self.sim_params.dt
        # 设置模拟器的轴向，通常为'z'轴向上
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        # 创建物理模拟器实例
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # 加载物体资产字典
        self.create_object_asset_dict(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets'))
        # 创建地面
        self._create_ground_plane()
        # 创建多个环境实例
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    # 创建地面平面
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # 设置地面法向量为(0, 0, 1)即垂直于z轴
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # 在模拟中添加地面
        self.gym.add_ground(self.sim, plane_params)

    # 加载物体资产字典，为每个训练物体加载其实体、目标和预测目标资产
    def create_object_asset_dict(self, asset_root):
        self.object_asset_dict = {}
        print("ENTER ASSET CREATING!")
        for used_objects in self.used_training_objects:
            object_asset_file = self.asset_files_dict[used_objects]
            object_asset_options = gymapi.AssetOptions()
            # 设置密度
            object_asset_options.density = 2000
            # 加载物体实体资产
            self.object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

            # 为目标资产禁用重力
            object_asset_options.disable_gravity = True

            # 加载目标资产
            goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

            # 加载预测目标资产
            predict_goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

            # 存入字典
            self.object_asset_dict[used_objects] = {'obj': self.object_asset, 'goal': goal_asset, 'predict goal': predict_goal_asset}


    # 创建多个环境实例，设置机器人手、物体和目标的初始状态
    def _create_envs(self, num_envs, spacing, num_per_row):
        # 设置环境边界
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets"
        # 定义机器人URDF文件路径
        allegro_hand_asset_file = "urdf/xarm6/xarm6_allegro_left_2023.urdf"
        allegro_hand_another_asset_file = "urdf/xarm6/xarm6_allegro_right_2023_binghao.urdf"

        object_asset_file = self.asset_files_dict["ball"]

        # 配置机器人手的资产参数
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        # 加载左右手资产
        allegro_hand_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_asset_file, asset_options)
        allegro_hand_another_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_another_asset_file, asset_options)

        # 获取机器人手的物理属性
        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_actuators = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_tendons = self.gym.get_asset_tendon_count(allegro_hand_asset)

        print("self.num_allegro_hand_bodies: ", self.num_allegro_hand_bodies)
        print("self.num_allegro_hand_shapes: ", self.num_allegro_hand_shapes)
        print("self.num_allegro_hand_dofs: ", self.num_allegro_hand_dofs)
        print("self.num_allegro_hand_actuators: ", self.num_allegro_hand_actuators)
        print("self.num_allegro_hand_tendons: ", self.num_allegro_hand_tendons)

        self.actuated_dof_indices = [i for i in range(16)]

        # 设置机器人手DOF（自由度）属性
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)
        allegro_hand_another_dof_props = self.gym.get_asset_dof_properties(allegro_hand_another_asset)

        # 初始化各DOF属性列表
        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []
        self.a_allegro_hand_dof_lower_limits = []
        self.a_allegro_hand_dof_upper_limits = []
        self.allegro_hand_dof_default_pos = []
        self.allegro_hand_dof_default_vel = []
        self.allegro_hand_dof_stiffness = []
        self.allegro_hand_dof_damping = []
        self.allegro_hand_dof_effort = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        # 遍历所有DOF以设置其属性
        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            self.a_allegro_hand_dof_lower_limits.append(allegro_hand_another_dof_props['lower'][i])
            self.a_allegro_hand_dof_upper_limits.append(allegro_hand_another_dof_props['upper'][i])
            self.allegro_hand_dof_default_pos.append(0.0)
            self.allegro_hand_dof_default_vel.append(0.0)

            # 预定义刚度值
            self.stiffness = [100, 100, 64, 64, 64, 40]

            # 设置驱动模式为位置控制
            allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            allegro_hand_another_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            # 如果是前6个DOF（底座关节），设置特定的刚度
            if i < 6:
                allegro_hand_dof_props['stiffness'][i] = self.stiffness[i]
                allegro_hand_another_dof_props['stiffness'][i] = self.stiffness[i]

            # 否则（手指关节），设置velocity、stiffness、effort、damping参数
            else:
                allegro_hand_dof_props['velocity'][i] = 3.0
                allegro_hand_dof_props['stiffness'][i] = 30
                allegro_hand_dof_props['effort'][i] = 5
                allegro_hand_dof_props['damping'][i] = 1
                allegro_hand_another_dof_props['velocity'][i] = 3.0
                allegro_hand_another_dof_props['stiffness'][i] = 30
                allegro_hand_another_dof_props['effort'][i] = 5
                allegro_hand_another_dof_props['damping'][i] = 1

        # 将配置好的DOF属性转换为torch tensor
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)
        self.a_allegro_hand_dof_lower_limits = to_torch(self.a_allegro_hand_dof_lower_limits, device=self.device)
        self.a_allegro_hand_dof_upper_limits = to_torch(self.a_allegro_hand_dof_upper_limits, device=self.device)
        self.allegro_hand_dof_default_pos = to_torch(self.allegro_hand_dof_default_pos, device=self.device)
        self.allegro_hand_dof_default_vel = to_torch(self.allegro_hand_dof_default_vel, device=self.device)

        # 加载待操作物体和目标资产
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500

        self.object_radius = 0.06
        # 创建球形物体作为默认物体
        object_asset = self.gym.create_sphere(self.sim, 0.12, object_asset_options)

        # 设置目标资产的重力属性
        object_asset_options.disable_gravity = True
        goal_asset = self.gym.create_sphere(self.sim, 0.04, object_asset_options)

        # 设置机器手的初始位姿
        allegro_hand_start_pose = gymapi.Transform()
        allegro_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.2, self.up_axis_idx))
        allegro_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, -1.56921)

        # 设置另一只手的初始位姿
        allegro_another_hand_start_pose = gymapi.Transform()
        allegro_another_hand_start_pose.p = gymapi.Vec3(0, -1.35, 0.2)
        allegro_another_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.57079)

        # 设置物体的初始位姿
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = allegro_hand_start_pose.p.x
        pose_dy, pose_dz = -0.22, 0.15

        object_start_pose.p.y = allegro_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = allegro_hand_start_pose.p.z + pose_dz
        object_start_pose.p = gymapi.Vec3(0.025, -0.38, 0.449)

        # 如果物体是“笔”，调整Z轴偏移
        if self.object_type == "pen":
            object_start_pose.p.z = allegro_hand_start_pose.p.z + 0.02

        # 设置目标位姿偏移
        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.0

        # 计算聚合（aggregate）大小，用于优化物理模拟性能
        max_agg_bodies = self.num_allegro_hand_bodies * 2 + 2 + 10
        max_agg_shapes = self.num_allegro_hand_shapes * 2 + 2 + 10

        self.allegro_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        # 初始化各种追踪索引列表
        self.hand_indices = []
        self.another_hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.predict_goal_object_indices = []

        # 遍历创建所有环境
        for i in range(self.num_envs):
            # 创建环境实例
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # 开启聚合模式，减少物理模拟中的计算复杂度
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 添加左手和右手角色
            allegro_hand_actor = self.gym.create_actor(env_ptr, allegro_hand_asset, allegro_hand_start_pose, "hand", i, -1, 0)
            allegro_hand_another_actor = self.gym.create_actor(env_ptr, allegro_hand_another_asset, allegro_another_hand_start_pose, "another_hand", i, -1, 0)

            # 保存手的初始状态
            self.hand_start_states.append([allegro_hand_start_pose.p.x, allegro_hand_start_pose.p.y, allegro_hand_start_pose.p.z,
                                           allegro_hand_start_pose.r.x, allegro_hand_start_pose.r.y, allegro_hand_start_pose.r.z, allegro_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            # 设置手的DOF属性并获取其在模拟器中的唯一索引
            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_actor, allegro_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_another_actor, allegro_hand_another_dof_props)
            another_hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_another_actor, gymapi.DOMAIN_SIM)
            self.another_hand_indices.append(another_hand_idx)

            # 添加物体角色
            index = i % len(self.used_training_objects)
            select_obj = self.used_training_objects[index]
            object_handle = self.gym.create_actor(env_ptr, self.object_asset_dict[select_obj]['obj'], object_start_pose, "object", i, 0, 0)

            # 保存物体初始状态
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # 调整物体的质量属性
            lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            for lego_body_prop in lego_body_props:
                lego_body_prop.mass *= 1
            self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, lego_body_props)

            # 调整物体的碰撞恢复系数(restitution)，设为0表示无弹性碰撞
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            for object_shape_prop in object_shape_props:
                object_shape_prop.restitution = 0
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            # 调整手的碰撞恢复系数
            hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, allegro_hand_actor)
            for hand_shape_prop in hand_shape_props:
                hand_shape_prop.restitution = 0.
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, hand_shape_props)

            # 添加目标物体
            goal_handle = self.gym.create_actor(env_ptr, self.object_asset_dict[select_obj]['goal'], goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            # 添加预测目标物体并设置特定颜色
            predict_goal_handle = self.gym.create_actor(env_ptr, self.object_asset_dict[select_obj]['predict goal'], goal_start_pose, "predict_goal_object", i + self.num_envs * 2, 0, 0)
            predict_goal_object_idx = self.gym.get_actor_index(env_ptr, predict_goal_handle, gymapi.DOMAIN_SIM)
            self.predict_goal_object_indices.append(predict_goal_object_idx)
            self.gym.set_rigid_body_color(env_ptr, predict_goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.4, 0.))
            # self.gym.set_actor_scale(env_ptr, predict_goal_handle, 0.01)

            # 如果启用了相机传感器，创建并设置相机
            if self.enable_camera_sensors:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0, -0.3, 0.43), gymapi.Vec3(0, -0.55, 0))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                # 计算相机的视图矩阵和投影矩阵逆矩阵
                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)

            # 设置物体和目标的颜色（如果不是方块）
            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            # 结束环境聚合
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_hand_actor)

            # 如果启用了相机，保存相机相关数据
            if self.enable_camera_sensors:
                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)

        # 查找接触传感器的句柄索引
        sensor_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, allegro_hand_another_actor, sensor_name) for sensor_name in
                          self.contact_sensor_names]

        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)

        # 获取物体的刚体质量
        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        # 将物理初始状态列表转换为 Tensor
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        # 将各种实体索引转换为 Tensor
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.another_hand_indices = to_torch(self.another_hand_indices, dtype=torch.long, device=self.device)

        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.predict_goal_object_indices = to_torch(self.predict_goal_object_indices, dtype=torch.long, device=self.device)

        # 初始化控制相关变量和PD控制器增益
        self.init_object_tracking = True
        self.test_for_robot_controller = False

        self.p_gain_val = 100.0
        self.d_gain_val = 4.0
        self.p_gain = torch.ones((self.num_envs, self.num_allegro_hand_dofs * 2), device=self.device, dtype=torch.float) * self.p_gain_val
        self.d_gain = torch.ones((self.num_envs, self.num_allegro_hand_dofs * 2), device=self.device, dtype=torch.float) * self.d_gain_val

        self.pd_previous_dof_pos = torch.zeros((self.num_envs, self.num_allegro_hand_dofs * 2), device=self.device, dtype=torch.float) * self.p_gain_val
        self.pd_dof_pos = torch.zeros((self.num_envs, self.num_allegro_hand_dofs * 2), device=self.device, dtype=torch.float) * self.p_gain_val

        # 初始化轨迹估计器模型
        self.traj_estimator = TrajEstimator(input_dim=60, output_dim=3).to(self.device)
        for param in self.traj_estimator.parameters():
            param.requires_grad_(True)

        self.is_test = self.cfg["is_test"]

        # 设置轨迹估计器的优化器和损失函数
        self.traj_estimator_optimizer = torch.optim.Adam(self.traj_estimator.parameters(), lr=0.0003)
        self.traj_estimator_save_path = "./traj_e/"
        os.makedirs(self.traj_estimator_save_path, exist_ok=True)
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss()

        # 根据测试/训练模式加载模型或初始化训练
        if self.is_test:
            self.traj_estimator.load_state_dict(torch.load("./traj_e/model.pt", map_location='cuda:0'))
            self.traj_estimator.eval()
        else:
            self.traj_estimator.train()

        # 初始化统计变量
        self.total_steps = 0
        self.success_buf = torch.zeros_like(self.rew_buf)
        self.hit_success_buf = torch.zeros_like(self.rew_buf)

    # 获取物体内部状态（旋转四元数）
    def get_internal_state(self):
        return self.root_state_tensor[self.object_indices, 3:7]

    # 获取内部信息（用于调试）
    def get_internal_info(self, key):
        if key == 'target':
            return self.debug_target
        elif key == 'qpos':
            return self.debug_qpos
        elif key == 'contact':
            return self.finger_contacts
        return None

    # 计算奖励函数，核心逻辑在于评估两手交接物体的质量
    def compute_reward(self, actions):
        # 调用具体的奖励计算工具函数，计算本步骤的奖励、重置标志、进度等
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:], self.catch_successes[:], self.catch_hold_counter[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.allegro_left_hand_pos, self.allegro_right_hand_pos, self.allegro_hand_another_thmub_pos, self.aux_up_pos, self.object_linvel, self.leeft_hand_ee_rot,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale, self.allegro_hand_another_ff_pos, self.allegro_hand_another_mf_pos, self.allegro_hand_another_rf_pos, self.allegro_hand_ff_pos, self.allegro_hand_mf_pos, self.allegro_hand_rf_pos, self.a_hand_palm_pos, unscale(self.another_allegro_hand_default_dof_pos[6:],
                                            self.allegro_hand_dof_lower_limits[6:22], self.allegro_hand_dof_upper_limits[6:22]), unscale(self.allegro_hand_another_dof_pos[:, 6:22] ,
                                            self.allegro_hand_dof_lower_limits[6:22], self.allegro_hand_dof_upper_limits[6:22]),
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen"),
            self.catch_successes, self.catch_hold_counter, self.catch_tolerance, self.catch_hold_steps, self.catch_vel_tolerance,
        )

        # 保存额外信息
        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        # 更新总步数计数
        self.total_steps += self.num_envs

        # 计算本局中完成（reset）的局数
        num_attempts_this_call = self.reset_buf.sum().item()
        # 计算在这些完成的局中，被判定为“接住过”的局数
        num_catches_this_call = (self.catch_successes * self.reset_buf).sum().item()

        # 更新全局统计数据
        self.total_attempts += num_attempts_this_call
        self.total_catch_successes += num_catches_this_call

        # 计算总的成功率
        success_rate = self.total_catch_successes / self.total_attempts if self.total_attempts > 0 else 0.0

        # 如果有局完成，更新平滑后的成功率
        if num_attempts_this_call > 0:
            # 计算当前这一步的真实成功率
            raw_episode_success_rate = num_catches_this_call / num_attempts_this_call
            # 平滑系数 alpha
            alpha = float(self.av_factor.item())
            # 如果是刚开始，直接赋值；否则进行指数平滑
            if self.total_attempts == num_attempts_this_call:
                self.smoothed_episode_success_rate = raw_episode_success_rate
            else:
                self.smoothed_episode_success_rate = (alpha * raw_episode_success_rate) + ((1 - alpha) * self.smoothed_episode_success_rate)

        print('Success Rate:', success_rate, 'Average episode Success Rate:', self.smoothed_episode_success_rate)

        # 使用 TensorBoard 记录统计信息
        self.writter.add_scalar('Total Attempts', float(self.total_attempts), self.total_steps)
        self.writter.add_scalar('Successful Throws and Catches', float(self.total_catch_successes), self.total_steps)
        self.writter.add_scalar('Success Rate', success_rate, self.total_steps)
        self.writter.add_scalar('Average Episode Success Rate', self.smoothed_episode_success_rate, self.total_steps)

        # 如果配置要求，打印详细的成功统计
        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    # 计算观测空间，刷新物理状态并提取关键的身体和物体位姿
    def compute_observations(self):
        # 刷新物理状态张量
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # 获取右手基座的位姿
        self.allegro_right_hand_base_pos = self.root_state_tensor[self.hand_indices, 0:3]
        self.allegro_right_hand_base_rot = self.root_state_tensor[self.hand_indices, 3:7]

        # 获取右手的末端执行器位姿
        self.allegro_right_hand_pos = self.rigid_body_states[:, 6, 0:3]
        self.allegro_right_hand_rot = self.rigid_body_states[:, 6, 3:7]

        # 获取左手的末端执行器位姿
        self.allegro_left_hand_pos = self.rigid_body_states[:, 6 + 23, 0:3]
        self.allegro_left_hand_rot = self.rigid_body_states[:, 6 + 23, 3:7]

        # 保存左手手掌基座位置
        self.a_hand_palm_pos = self.allegro_left_hand_pos.clone()

        # 根据手的朝向对末端位置进行空间坐标转换，计算更精确的接触点
        self.allegro_left_hand_pos = self.allegro_left_hand_pos + quat_apply(self.allegro_left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.allegro_left_hand_pos = self.allegro_left_hand_pos + quat_apply(self.allegro_left_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04)

        # 获取物体的位姿、线速度和角速度
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        # 获取目标的位姿
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # 获取手指关键点的位姿
        self.allegro_hand_another_thmub_pos = self.rigid_body_states[:, 14 + 23, 0:3]
        self.allegro_hand_another_thmub_rot = self.rigid_body_states[:, 14 + 23, 3:7]

        self.allegro_hand_another_ff_pos = self.rigid_body_states[:, 10, 0:3]
        self.allegro_hand_another_mf_pos = self.rigid_body_states[:, 18, 0:3]
        self.allegro_hand_another_rf_pos = self.rigid_body_states[:, 22, 0:3]

        self.allegro_hand_ff_pos = self.rigid_body_states[:, 10 + 23, 0:3]
        self.allegro_hand_mf_pos = self.rigid_body_states[:, 18 + 23, 0:3]
        self.allegro_hand_rf_pos = self.rigid_body_states[:, 22 + 23, 0:3]

        # 获取左手末端执行器的旋转四元数
        self.leeft_hand_ee_rot = self.rigid_body_states[:, self.another_hand_base_rigid_body_index, 3:7]

        # 生成用于观测噪声的随机数
        rand_floats = torch_rand_float(-1.0, 1.0, (self.num_envs, 63), device=self.device)

        # 定义辅助参考点位置
        self.aux_up_pos = to_torch([0, -0.52, 0.45], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # 计算 Sim2Real 的观测数据
        self.compute_sim2real_observation(rand_floats)

        # 如果需要不对称观测（例如用于训练 Critic 网络），计算相应的观测数据
        if self.asymmetric_obs:
            self.compute_sim2real_asymmetric_obs(rand_floats)

    # 计算 Sim2Real 观测数据，包含多帧物体轨迹栈的处理和接触位置预测
    def compute_sim2real_observation(self, rand_floats):
        # 填充右手关节位置观测 (0~22)
        self.obs_buf[:, 0:22] = unscale(self.allegro_hand_dof_pos,
                                            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)

        # 隐藏部分观测信息以模拟现实情况
        self.obs_buf[:, 0:6] = 0
        self.obs_buf[:, 1] = self.allegro_hand_dof_pos[:, 1]
        self.obs_buf[:, 2] = self.allegro_hand_dof_pos[:, 2]

        # 填充目标位置观测 (22~25)
        self.obs_buf[:, 22:25] = (self.goal_pos - self.allegro_right_hand_base_pos).clone()

        # 填充左手关节位置观测 (150~172)
        self.obs_buf[:, 150:172] = unscale(self.allegro_hand_another_dof_pos,
                                                            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.obs_buf[:, 150:156] = self.allegro_hand_another_dof_pos[:, :6]
        self.obs_buf[:, 150:151] = 0
        self.obs_buf[:, 153:154] = 0

        # 更新物体轨迹帧栈
        for i in range(self.object_seq_len):
            if i == self.object_seq_len - 1:
                self.object_state_stack_frames[:, (i)*3:(i+1)*3] = (self.object_pos - self.allegro_right_hand_base_pos).clone()
            else:
                self.object_state_stack_frames[:, (i)*3:(i+1)*3] = self.object_state_stack_frames[:, (i+1)*3:(i+2)*3].clone()

        # 使用临时上下文管理器开启梯度计算，进行轨迹预测
        with TemporaryGrad():
            self.predict_pose, self.pose_latent_vector = self.predict_contact_pose(self.traj_estimator, self.object_state_stack_frames)
            # 更新接触点优化器
            self.update_contact_slamer(self.predict_pose)

        # 填充预测的接触位姿和带有随机噪声的物体状态
        self.obs_buf[:, 260:263] = self.predict_pose[:, 0:3].detach()
        self.obs_buf[:, 248:260] = self.object_state_stack_frames[:, 36:48].clone() + rand_floats[:, 0:12] * 0.05

        # 更新观测缓冲区历史帧栈
        for i in range(len(self.obs_buf_stack_frames) - 1):
            self.obs_buf[:, (i+1) * self.one_frame_num_obs:(i+2) * self.one_frame_num_obs] = self.obs_buf_stack_frames[i]
            self.obs_buf_stack_frames[i] = self.obs_buf[:, (i) * self.one_frame_num_obs:(i+1) * self.one_frame_num_obs].clone()

    # 使用轨迹预测器预测接触位姿
    def predict_contact_pose(self, traj_estimator, contact_buf):
        predict_pose, pose_latent_vector = traj_estimator(contact_buf)
        return predict_pose, pose_latent_vector

    # 使用均方误差(MSE)更新接触点估计器
    def update_contact_slamer(self, predict_pose):
        # 计算预测位置与实际目标位置的损失
        self.pos_loss = F.mse_loss(predict_pose[:, 0:3], (self.goal_pos - self.allegro_right_hand_base_pos).clone())
        loss = self.pos_loss
        # 反向传播更新网络参数
        self.traj_estimator_optimizer.zero_grad()
        loss.backward()
        self.traj_estimator_optimizer.step()
        self.extras['pos_loss'] = self.pos_loss.unsqueeze(0)

    # 计算不对称观测数据（用于训练 Critic 网络）
    def compute_sim2real_asymmetric_obs(self, rand_floats):
        # 填充右手关节状态（位置和速度）
        self.states_buf[:, 0:self.num_allegro_hand_dofs] = unscale(self.allegro_hand_dof_pos,
                                                    self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.states_buf[:, self.num_allegro_hand_dofs:2*self.num_allegro_hand_dofs] = self.vel_obs_scale * self.allegro_hand_dof_vel

        # 填充动作历史记录
        action_obs_start = 44
        self.states_buf[:, action_obs_start:action_obs_start + 22] = self.actions[:, :22]

        # 填充左手关节状态
        another_hand_start = action_obs_start + 22
        self.states_buf[:, another_hand_start:self.num_allegro_hand_dofs + another_hand_start] = unscale(self.allegro_hand_another_dof_pos,
                                                            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.states_buf[:, self.num_allegro_hand_dofs + another_hand_start:2*self.num_allegro_hand_dofs + another_hand_start] = self.vel_obs_scale * self.allegro_hand_another_dof_vel

        # 填充左手动作历史
        action_another_obs_start = another_hand_start + 44
        self.states_buf[:, action_another_obs_start:action_another_obs_start + 22] = self.actions[:, 22:]

        # 填充左手接触传感器数据
        contact_start = action_another_obs_start + 22
        contacts = self.contact_tensor.reshape(self.num_envs, -1, 3)
        contacts = contacts[:, self.sensor_handle_indices, :]
        contacts = torch.norm(contacts, dim=-1)
        contacts = torch.where(contacts >= 1.0, 1.0, 0.0)
        self.states_buf[:, contact_start:contact_start + 11] = contacts

        # 填充物体状态（位姿、线速度、角速度）
        obj_obs_start = contact_start + 11
        self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

        # 填充目标位置状态及目标与物体的旋转差值
        goal_obs_start = obj_obs_start + 13
        self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        self.states_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

        # 填充随机化参数及物体历史位置偏移
        randomize_param_start = goal_obs_start + 11
        self.states_buf[:, randomize_param_start:randomize_param_start + 12] = rand_floats[:, 0:12]

        offseted_pos_goal_start = randomize_param_start + 12
        self.states_buf[:, offseted_pos_goal_start:offseted_pos_goal_start + 12] = self.object_state_stack_frames[:, 36:48].clone() + rand_floats[:, 0:12] * 0.05
        self.states_buf[:, offseted_pos_goal_start + 12:offseted_pos_goal_start + 15] = (self.goal_pos - self.allegro_right_hand_base_pos).clone()

    # 重置目标物体的位姿
    def reset_target_pose(self, env_ids, apply_reset=False):
        # 生成随机噪声
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        # 设置目标基础状态
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 0] += rand_floats[:, 0] * 0.05
        self.goal_states[env_ids, 1] -= 0.55 + rand_floats[:, 1] * 0.05
        self.goal_states[env_ids, 2] += 0.1

        # 更新根状态张量
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        # 生成目标随机旋转
        quat = quat_from_euler_xyz(torch.sign(rand_floats[:, 0]) * 3.1415, torch.sign(rand_floats[:, 1]) * 3.1415 + 1.571, torch.sign(rand_floats[:, 2]) * 3.14)
        self.root_state_tensor[self.goal_object_indices[env_ids], 3] = quat[:, 0]
        self.root_state_tensor[self.goal_object_indices[env_ids], 4] = quat[:, 1]
        self.root_state_tensor[self.goal_object_indices[env_ids], 5] = quat[:, 2]
        self.root_state_tensor[self.goal_object_indices[env_ids], 6] = quat[:, 3]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        # 如果需要应用重置，更新模拟器状态
        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    # 重置环境状态（在训练过程中重置机器人、物体和目标位置）
    def reset(self, env_ids, goal_env_ids):
        # 如果启用了随机化，应用任务参数随机化
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # 生成环境扰动方向
        self.perturb_direction[env_ids] = torch_rand_float(-1, 1, (len(env_ids), 6), device=self.device).squeeze(-1)

        # 生成随机噪声
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_allegro_hand_dofs * 2 + 5), device=self.device)

        # 随机化目标物体的位姿
        self.reset_target_pose(env_ids)

        # 重置物体状态
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        # 加入物体位置噪声
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        # 设置物体随机姿态
        quat = quat_from_euler_xyz(torch.sign(rand_floats[:, 3]) * 3.1415, torch.sign(rand_floats[:, 4]) * 3.1415 + 1.571, torch.sign(rand_floats[:, 5]) * 3.14)
        self.root_state_tensor[self.object_indices[env_ids], 3] = quat[:, 0]
        self.root_state_tensor[self.object_indices[env_ids], 4] = quat[:, 1]
        self.root_state_tensor[self.object_indices[env_ids], 5] = quat[:, 2]
        self.root_state_tensor[self.object_indices[env_ids], 6] = quat[:, 3]
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        # 更新开环控制用的物体位姿记录
        self.object_pose_for_open_loop[env_ids] = self.root_state_tensor[self.object_indices[env_ids], 0:7].clone()

        # 收集需要重置的所有物体相关索引
        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.predict_goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))

        # 重置机器人手自由度(DOF)状态
        pos = self.allegro_hand_default_dof_pos
        another_pos = self.another_allegro_hand_default_dof_pos

        self.allegro_hand_dof_pos[env_ids, :] = pos
        self.allegro_hand_another_dof_pos[env_ids, :] = another_pos

        # 加入关节速度噪声
        self.allegro_hand_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_allegro_hand_dofs:5+self.num_allegro_hand_dofs*2]

        self.allegro_hand_another_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_allegro_hand_dofs:5+self.num_allegro_hand_dofs*2]

        # 重置控制目标
        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = pos

        self.prev_targets[env_ids, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2] = another_pos
        self.cur_targets[env_ids, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2] = another_pos

        # 设置模拟器中手和物体的状态
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        another_hand_indices = self.another_hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices,
                                                 another_hand_indices]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                                 object_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))

        # 重置进度、标志位等
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.catch_successes[env_ids] = 0
        self.catch_hold_counter[env_ids] = 0

        self.proprioception_close_loop[env_ids] = self.allegro_hand_dof_pos[env_ids, 0:22].clone()
        self.object_state_stack_frames[env_ids] = torch.zeros_like(self.object_state_stack_frames[env_ids])

        self.init_object_tracking = True
        self.gym.clear_lines(self.viewer)

    # 在物理步进前的控制处理
    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # 检查需要重置的环境
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # 如果只有目标需要重置，调用设置接口
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # 如果需要重置环境，调用 reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        # 应用动作，计算驱动目标(targets)
        self.cur_targets[:, self.actuated_dof_indices + 6] = scale(self.actions[:, 6:22],
                                                            self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
        self.cur_targets[:, self.actuated_dof_indices + 28] = scale(self.actions[:, 28:44],
                                                            self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])

        # 基础位置平滑移动
        self.cur_targets[:, [1, 2]] = self.prev_targets[:, [1, 2]] + self.actions[:, [1, 2]] * 2
        self.cur_targets[:, 22:28] = self.prev_targets[:, 22:28] + self.actions[:, 22:28] * 0.1

        # 动作移动平均平滑
        self.cur_targets[:, self.actuated_dof_indices + 6] = self.act_moving_average * self.cur_targets[:,
                                                                                                    self.actuated_dof_indices + 6] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices + 6]

        self.cur_targets[:, self.actuated_dof_indices + 28] = self.act_moving_average * self.cur_targets[:,
                                                                                                    self.actuated_dof_indices + 22] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices + 22]

        # 限制关节角度在物理允许范围内
        self.cur_targets[:, 0:22] = tensor_clamp(self.cur_targets[:, 0:22],
                                                self.allegro_hand_dof_lower_limits[:],
                                                self.allegro_hand_dof_upper_limits[:])

        self.cur_targets[:, 22:44] = tensor_clamp(self.cur_targets[:, 22:44],
                                                self.a_allegro_hand_dof_lower_limits[:],
                                                self.a_allegro_hand_dof_upper_limits[:])

        self.prev_targets[:, :] = self.cur_targets[:, :]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # 更新预测目标物体的位姿（用于可视化预测结果）
        self.root_state_tensor[self.predict_goal_object_indices, 0:3] = self.predict_pose[:, 0:3].detach() + self.root_state_tensor[self.hand_indices, 0:3] - ((self.predict_pose[:, 0:3].detach() + self.root_state_tensor[self.hand_indices, 0:3]) - self.goal_pos) * torch.clamp(self.progress_buf[0] * random.random() / 10, 0, 1)
        self.root_state_tensor[self.predict_goal_object_indices, 3:7] = self.root_state_tensor[self.object_indices, 3:7].detach()

        object_indices = torch.unique(torch.cat([self.object_indices,
                                                 self.goal_object_indices,
                                                 self.predict_goal_object_indices]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices.to(torch.int32)), len(object_indices.to(torch.int32)))

        # 定期保存轨迹估计器模型
        if self.total_steps % (200 * (self.max_episode_length - 1)) == 0:
            if not self.is_test:
                torch.save(self.traj_estimator.state_dict(), self.traj_estimator_save_path + "/model.pt")

        # （可选）施加额外的力，模拟扰动
        self.apply_force = False
        if self.apply_force == True:
            self.rb_forces[:] = torch.zeros_like(self.rb_forces)
            # 根据物体高度判断是否应用
            self.apply_force_env_id = torch.where(self.root_state_tensor[self.object_indices, 1] < 10.1,
                                        torch.where(self.root_state_tensor[self.object_indices, 1] > -10.9, 1, 0,), 0).nonzero(as_tuple=False).squeeze(-1)
            # 设置扰动方向
            self.perturb_direction[self.apply_force_env_id, 0] = 0
            self.perturb_direction[self.apply_force_env_id, 1] = 0
            self.perturb_direction[self.apply_force_env_id, 2] = 1

            # 应用力
            self.rb_forces[self.apply_force_env_id, self.object_rb_handles, 0:3] = torch.ones(
                self.rb_forces[self.apply_force_env_id, self.object_rb_handles, 0:3].shape, device=self.device) * 1.5 * self.perturb_direction[self.apply_force_env_id, 0:3].squeeze(-1)
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    # 物理步进后的处理（用于更新统计信息和可视化）
    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        # 计算下一帧的观测和奖励
        self.compute_observations()
        self.compute_reward(self.actions)

        # 如果启用了调试可视化，绘制坐标轴
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.allegro_hand_another_thmub_pos[i], self.allegro_hand_another_thmub_rot[i], line_width=2)


    # 添加用于可视化调试的线（绘制坐标轴）
    def add_debug_lines(self, env, pos, rot, line_width=1):
        # 计算坐标轴的三个方向点
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        # 绘制三轴线条
        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, line_width, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, line_width, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, line_width, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################

# 使用 Torch JIT 编译的奖励计算函数
@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, allegro_left_hand_pos, allegro_right_hand_pos, allegro_another_hand_thmub_pos, aux_up_pos, object_vel, leeft_hand_ee_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float, allegro_hand_another_ff_pos, allegro_hand_another_mf_pos, allegro_hand_another_rf_pos, allegro_hand_ff_pos, allegro_hand_mf_pos, allegro_hand_rf_pos, a_hand_palm_pos, hand_init_qpos, hand_qpos,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool,
    catch_successes, catch_hold_counter, catch_tolerance: float, catch_hold_steps: int, catch_vel_tolerance: float,
):
    # 计算手到物体的距离
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    left_hand_dist =  torch.norm(allegro_left_hand_pos - object_pos, p=2, dim=-1)

    # 判断接触条件：距离足够近且物体高度符合要求
    catch_condition = (left_hand_dist <= catch_tolerance) & (object_pos[:, 2] > 0.15)

    # 维护连续接触帧计数器，如果不满足则立即归零
    catch_hold_counter = torch.where(
        catch_condition,
        catch_hold_counter + 1,
        torch.zeros_like(catch_hold_counter),
    )

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # 计算物体与目标的旋转距离
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist
    # 动作惩罚
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # 物体速度奖励（鼓励速度向特定方向优化）
    object_vel_reward = torch.clamp(-object_vel[:, 1], -0.1, 0.1)
    object_vel_reward = torch.where(object_pos[:, 1] < -0.65,
                                    torch.where(-0.85 < object_pos[:, 1], object_vel_reward, torch.zeros_like(object_vel_reward)), torch.zeros_like(object_vel_reward))

    # 合成总奖励
    reward = (torch.exp(-4*((3 * dist_rew))) + object_vel_reward) - 0.001 * action_penalty

    # 更新任务完成成功计数
    goal_resets = torch.where(torch.abs(goal_dist) <= 0, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # 若达成目标添加奖励
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # 检查终止条件
    resets = torch.where(object_pos[:, 2] <= 0.15, torch.ones_like(reset_buf), reset_buf)

    # 如果配置了最大连续成功数，达到后强制重置
    if max_consecutive_successes > 0:
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # 应用超时重置的惩罚
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    # 根据 reset 时是否仍持有物体判定“接住”是否成功
    catch_successes = torch.where(
        resets == 1,
        torch.where(
            catch_hold_counter >= catch_hold_steps,
            torch.ones_like(catch_successes),
            torch.zeros_like(catch_successes),
        ),
        catch_successes,
    )

    # 计算连续成功平滑均值
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes, catch_successes, catch_hold_counter

# 随机化姿态（通过轴角旋转）
@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))

# 随机化笔的姿态
@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

# 计算姿态误差
def orientation_error(desired, current):
	cc = quat_conjugate(current)
	q_r = quat_mul(desired, cc)
	return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

# 基于雅可比矩阵的逆运动学控制
def control_ik(j_eef, device, dpose, num_envs):
    damping = 0.05
    # 求解阻尼最小二乘法(Damped Least Squares)
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u
