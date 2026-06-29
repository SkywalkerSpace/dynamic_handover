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

from torch import nn
import torch.nn.functional as F


# 轨迹估计器神经网络：用于预测物体的目标接触位姿，输入为物体位置历史序列，输出为预测的目标 3D 位置
class TrajEstimator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TrajEstimator, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, output_dim)

        self.activate_func = nn.ELU()

    def forward(self, inputs):
        # 前向传播：通过三层 MLP + ELU 激活提取特征，最终线性层输出预测位姿
        x = self.activate_func(self.linear1(inputs))
        x = self.activate_func(self.linear2(x))
        x = self.activate_func(self.linear3(x))
        outputs = self.output_layer(x)

        return outputs, x  # 同时返回输出和倒数第二层特征（可用于辅助损失或蒸馏）


# 临时启用梯度计算的上下文管理器（用于在推理阶段临时开启反向传播以更新轨迹估计器）
class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.prev)


# Allegro 双手动态递物主任务类，继承 BaseTask，实现递物场景的完整仿真逻辑
class AllegroHandDynamicHandover(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.allegro_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(
                round(self.reset_time/(control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in [
            "block", "egg", "pen", "ycb/banana", "ycb/can", "ycb/mug", "ycb/brick"]

        self.ignore_z = (self.object_type == "pen")

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

        # new novel objects
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

        # self.used_training_objects = ['ball', "block"]
        self.used_training_objects = [
            "pen", "ball", "block", "obj0", "obj1", "obj2", "obj4", "obj6", "obj7", "obj9", "obj10"]

        # self.used_training_objects = ["novel_obj1", "novel_obj2", "novel_obj3", "novel_obj4", "novel_obj5", "novel_obj6",
        #                               "novel_obj7", "novel_obj8", "novel_obj9", "novel_obj10", "novel_obj11", "novel_obj12", 
        #                               "novel_obj13", "novel_obj14"]

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

        self.contact_sensor_names = [
            "link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr",
            "link_5.0_fsr", "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr",
            "link_10.0_fsr", "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr"
        ]

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
        else:
            self.pointCloudVisualizer = None

        # 调用基类 BaseTask 的 __init__：创建仿真实例、分配缓冲区、准备 GPU 张量
        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            # 设置观察窗口的摄像机位置和朝向
            # cam_pos = gymapi.Vec3(0.9, -0.65, 1.0)
            # cam_target = gymapi.Vec3(-0.5, -0.65, 0.2)

            cam_pos = gymapi.Vec3(5.0, -0.5, 2.0)
            cam_target = gymapi.Vec3(-0.5, -0.65, 0.2)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

        # 获取 GPU 上的仿真状态张量句柄（Actor 根状态、自由度状态、刚体状态、接触力、雅可比矩阵）
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(
            self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.jacobian_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_jacobian_tensor(self.sim, "another_hand"))

        # 首次刷新张量缓存，使 wrap_tensor 后的 PyTorch 张量获取有效数据
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 左手（another hand）默认关节位置：前6维为机械臂（xArm6），后16维为 Allegro 手指
        self.another_allegro_hand_default_dof_pos = torch.zeros(
            self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
        self.another_allegro_hand_default_dof_pos[:6] = torch.tensor(
            [-0.0, -0.09, -0.09, 3.141, 2.00, -1.57], dtype=torch.float, device=self.device)
        self.another_allegro_hand_default_dof_pos[6:] = to_torch(
            [-0.03989830748810656, 1.3495253790945758, 0.8659920759388671, 0.780414711365591,
             0.9655586519308622, 1.0139016439397597, 0.8501943208059994, 1.3264760744914152,
             -0.20482272250532974, 1.347864170294202, 0.6030536585610538, 0.9181400800651911,
             -0.21341465375119012, 1.7199185039090872, 1.2686849760515697, 0.8245164874462315,
             ], dtype=torch.float, device=self.device)

        # 右手默认关节位置（与左手对称）
        self.allegro_hand_default_dof_pos = torch.zeros(
            self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
        # self.allegro_hand_default_dof_pos[:6] = torch.tensor([0, 0, -1, 3.14, 0.57, 3.14], dtype=torch.float, device=self.device)
        self.allegro_hand_default_dof_pos[:6] = torch.tensor(
            [-0.0, -0.09, -0.09, 3.141, 2.00, -1.57], dtype=torch.float, device=self.device)
        # self.allegro_hand_default_dof_pos[6:] = to_torch([0.0, -0.174, 0.785, 0.785,
        #                                     0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)

        # default qpos
        self.allegro_hand_default_dof_pos[6:] = to_torch(
            [-0.03989830748810656, 1.3495253790945758, 0.8659920759388671, 0.780414711365591,
             0.9655586519308622, 1.0139016439397597, 0.8501943208059994, 1.3264760744914152,
             -0.20482272250532974, 1.347864170294202, 0.6030536585610538, 0.9181400800651911,
             -0.21341465375119012, 1.7199185039090872, 1.2686849760515697, 0.8245164874462315,
             ], dtype=torch.float, device=self.device)

        # hand put
        # self.allegro_hand_default_dof_pos[6:] = to_torch([0,0,0.7,1.2,0,0.7,0.3,1.2, 0,0,0.7,1.2,0,0,0.7,1.2,], dtype=torch.float, device=self.device)

        # hand grip
        # self.allegro_hand_default_dof_pos[6:] = to_torch([0,0.5,0.7,1.2,1.57,0.3,1.2,0.7,0,0.3,0.7,1.2,0,0.5,0.7,1.2,], dtype=torch.float, device=self.device)

        # 将原始 GPU 张量包装为 PyTorch 张量切片，方便按环境和自由度维度索引
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # 右手的关节状态切片：位置和速度
        self.allegro_hand_dof_state = self.dof_state.view(
            self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        # 左手的关节状态切片
        self.allegro_hand_another_dof_state = self.dof_state.view(
            self.num_envs, -1, 2)[:, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2]
        self.allegro_hand_another_dof_pos = self.allegro_hand_another_dof_state[..., 0]
        self.allegro_hand_another_dof_vel = self.allegro_hand_another_dof_state[..., 1]

        # 刚体状态张量：每个刚体的 [位置(3) + 四元数(4) + 线速度(3) + 角速度(3)] = 13维
        self.rigid_body_states = gymtorch.wrap_tensor(
            rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        # Actor 根状态张量切片：位置、朝向、线速度、角速度
        self.root_state_tensor = gymtorch.wrap_tensor(
            actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()  # 保存初始状态副本用于重置

        # 接触力张量：每个刚体受到的净接触力 (3维)
        self.contact_tensor = gymtorch.wrap_tensor(
            contact_tensor).view(self.num_envs, -1)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        # 前一帧和当前帧的关节目标位置（用于指数移动平均平滑控制指令）
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.object_init_quat = torch.zeros(
            (self.num_envs, 4), dtype=torch.float, device=self.device)

        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(
            1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(
            self.av_factor, dtype=torch.float, device=self.device)
        self.object_pose_for_open_loop = torch.zeros_like(
            self.root_state_tensor[self.object_indices, 0:7])

        self.total_successes = 0
        self.total_resets = 0

        self.state_buf_stack_frames = []
        self.obs_buf_stack_frames = []

        for i in range(3):
            self.obs_buf_stack_frames.append(torch.zeros_like(
                self.obs_buf[:, 0:self.one_frame_num_obs]))
            self.state_buf_stack_frames.append(
                torch.zeros_like(self.states_buf[:, 0:215]))

        self.object_seq_len = 20
        self.object_state_stack_frames = torch.zeros(
            (self.num_envs, self.object_seq_len * 3), dtype=torch.float, device=self.device)

        self.proprioception_close_loop = torch.zeros_like(
            self.allegro_hand_dof_pos[:, 0:22])
        self.another_hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.another_hand_indices[0], "link6", gymapi.DOMAIN_ENV)
        print("another_hand_base_rigid_body_index: ",
              self.another_hand_base_rigid_body_index)
        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.hand_indices[0], "link6", gymapi.DOMAIN_ENV)
        print("hand_base_rigid_body_index: ", self.hand_base_rigid_body_index)
        # with open("./demo_throw.pkl", "rb") as f:
        #     self.demo_throw = pickle.load(f)

        # print(self.demo_throw)
        # # self.demo_throw = to_torch(self.demo_throw['qpos'], dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        # self.demo_throw = to_torch(self.demo_throw['qpos'], dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        object_rb_count = self.gym.get_asset_rigid_body_count(
            self.object_asset)
        self.object_rb_handles = 46
        self.perturb_direction = torch_rand_float(
            -1, 1, (self.num_envs, 6), device=self.device).squeeze(-1)

        self.predict_pose = self.goal_init_state[:, 0:3].clone()

        self.log_dir = str(
            './logs/allegro_hand_dynamic_handover/mappo/success_logs_seed22')
        self.writter = SummaryWriter(self.log_dir)

    def create_sim(self):
        """ 创建物理仿真实例，并在其中构建地面、加载物体资产、创建并行环境。 """
        self.dt = self.sim_params.dt
        # 根据配置的向上轴方向设置重力，返回轴索引 (z=2, y=1)
        self.up_axis_idx = self.set_sim_params_up_axis(
            self.sim_params, self.up_axis)

        # 调用基类方法创建仿真实例
        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                      self.physics_engine, self.sim_params)
        # 加载所有训练用物体的 URDF/MJCF 资产并建立资产字典
        self.create_object_asset_dict(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '../../assets'))
        # 创建地面平面
        self._create_ground_plane()
        # 创建所有并行仿真环境（包含双手、物体、目标）
        self._create_envs(
            self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        """ 创建水平地面，法线朝向 z 轴正方向。 """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_object_asset_dict(self, asset_root):
        """ 预加载所有训练用物体的物理资产，每个物体生成三个资产：实体(obj)、目标指示器(goal)、预测目标指示器(predict goal)。 """
        self.object_asset_dict = {}
        print("ENTER ASSET CREATING!")
        for used_objects in self.used_training_objects:
            object_asset_file = self.asset_files_dict[used_objects]
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 2000
            # 加载物体实体资产（受重力影响）
            self.object_asset = self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options)

            # 加载目标指示器资产（禁用重力，仅用于可视化显示目标位置）
            object_asset_options.disable_gravity = True
            goal_asset = self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options)

            # 加载预测目标指示器资产（显示轨迹估计器的预测结果）
            predict_goal_asset = self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options)

            self.object_asset_dict[used_objects] = {
                'obj': self.object_asset, 'goal': goal_asset, 'predict goal': predict_goal_asset}

    def _create_envs(self, num_envs, spacing, num_per_row):
        """ 创建所有并行仿真环境：每个环境包含左右双手、操作物体、目标指示器和预测目标指示器。 """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets"
        # 右手 URDF（名称“left”为原始命名惯例）
        allegro_hand_asset_file = "urdf/xarm6/xarm6_allegro_left_2023.urdf"
        allegro_hand_another_asset_file = "urdf/xarm6/xarm6_allegro_right_2023_binghao.urdf"  # 左手 URDF

        object_asset_file = self.asset_files_dict["ball"]

        # 加载双手机械臂资产，设置物理属性（固定基座、禁用重力、拆叠固定关节等）
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        # asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        # asset_options.override_com = True
        # asset_options.override_inertia = True
        # asset_options.vhacd_enabled = True
        # asset_options.vhacd_params = gymapi.VhacdParams()
        # asset_options.vhacd_params.resolution = 3000000
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        # 加载右手和左手的机械臂+手指联合资产
        allegro_hand_asset = self.gym.load_asset(
            self.sim, asset_root, allegro_hand_asset_file, asset_options)
        allegro_hand_another_asset = self.gym.load_asset(
            self.sim, asset_root, allegro_hand_another_asset_file, asset_options)

        # 获取资产的刚体/形状/自由度/驱动器/肌腱数量
        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(
            allegro_hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(
            allegro_hand_asset)
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(
            allegro_hand_asset)
        self.num_allegro_hand_actuators = self.gym.get_asset_dof_count(
            allegro_hand_asset)
        self.num_allegro_hand_tendons = self.gym.get_asset_tendon_count(
            allegro_hand_asset)

        print("self.num_allegro_hand_bodies: ", self.num_allegro_hand_bodies)
        print("self.num_allegro_hand_shapes: ", self.num_allegro_hand_shapes)
        print("self.num_allegro_hand_dofs: ", self.num_allegro_hand_dofs)
        print("self.num_allegro_hand_actuators: ",
              self.num_allegro_hand_actuators)
        print("self.num_allegro_hand_tendons: ", self.num_allegro_hand_tendons)

        self.actuated_dof_indices = [i for i in range(16)]

        # 获取并配置双手的关节物理属性（包括驱动模式、刚度、阻尼、力矩上限等）
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(
            allegro_hand_asset)
        allegro_hand_another_dof_props = self.gym.get_asset_dof_properties(
            allegro_hand_another_asset)

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

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(
                allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(
                allegro_hand_dof_props['upper'][i])
            self.a_allegro_hand_dof_lower_limits.append(
                allegro_hand_another_dof_props['lower'][i])
            self.a_allegro_hand_dof_upper_limits.append(
                allegro_hand_another_dof_props['upper'][i])
            self.allegro_hand_dof_default_pos.append(0.0)
            self.allegro_hand_dof_default_vel.append(0.0)

            self.stiffness = [100, 100, 64, 64, 64, 40]

            allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            allegro_hand_another_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if i < 6:
                allegro_hand_dof_props['stiffness'][i] = self.stiffness[i]
                allegro_hand_another_dof_props['stiffness'][i] = self.stiffness[i]

            else:
                allegro_hand_dof_props['velocity'][i] = 3.0
                allegro_hand_dof_props['stiffness'][i] = 30
                allegro_hand_dof_props['effort'][i] = 5
                allegro_hand_dof_props['damping'][i] = 1
                allegro_hand_another_dof_props['velocity'][i] = 3.0
                allegro_hand_another_dof_props['stiffness'][i] = 30
                allegro_hand_another_dof_props['effort'][i] = 5
                allegro_hand_another_dof_props['damping'][i] = 1

        self.actuated_dof_indices = to_torch(
            self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.allegro_hand_dof_lower_limits = to_torch(
            self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(
            self.allegro_hand_dof_upper_limits, device=self.device)
        self.a_allegro_hand_dof_lower_limits = to_torch(
            self.a_allegro_hand_dof_lower_limits, device=self.device)
        self.a_allegro_hand_dof_upper_limits = to_torch(
            self.a_allegro_hand_dof_upper_limits, device=self.device)
        self.allegro_hand_dof_default_pos = to_torch(
            self.allegro_hand_dof_default_pos, device=self.device)
        self.allegro_hand_dof_default_vel = to_torch(
            self.allegro_hand_dof_default_vel, device=self.device)

        # 加载操作物体和目标资产（这里创建了默认的球体作为备用）
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500

        self.object_radius = 0.06
        object_asset = self.gym.create_sphere(
            self.sim, 0.12, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.create_sphere(
            self.sim, 0.04, object_asset_options)

        allegro_hand_start_pose = gymapi.Transform()
        allegro_hand_start_pose.p = gymapi.Vec3(
            *get_axis_params(0.2, self.up_axis_idx))
        allegro_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, -1.56921)

        allegro_another_hand_start_pose = gymapi.Transform()
        allegro_another_hand_start_pose.p = gymapi.Vec3(0, -1.35, 0.2)
        allegro_another_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.57079)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = allegro_hand_start_pose.p.x
        pose_dy, pose_dz = -0.22, 0.15
        # pose_dy, pose_dz = -0.3, 0.45

        object_start_pose.p.y = allegro_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = allegro_hand_start_pose.p.z + pose_dz
        object_start_pose.p = gymapi.Vec3(0.025, -0.38, 0.449)

        if self.object_type == "pen":
            object_start_pose.p.z = allegro_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.0

        # compute aggregate size
        max_agg_bodies = self.num_allegro_hand_bodies * 2 + 2 + 10
        max_agg_shapes = self.num_allegro_hand_shapes * 2 + 2 + 10

        self.allegro_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.another_hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.predict_goal_object_indices = []

        for i in range(self.num_envs):
            # 创建第 i 个并行环境实例
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                # 开启聚合模式：将同一环境内的所有 Actor 打包以优化 GPU 碰撞检测
                self.gym.begin_aggregate(
                    env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 创建右手和左手 Actor（collision filter = -1 使用资产自带的碰撞过滤）
            allegro_hand_actor = self.gym.create_actor(
                env_ptr, allegro_hand_asset, allegro_hand_start_pose, "hand", i, -1, 0)
            allegro_hand_another_actor = self.gym.create_actor(
                env_ptr, allegro_hand_another_asset, allegro_another_hand_start_pose, "another_hand", i, -1, 0)

            self.hand_start_states.append(
                [allegro_hand_start_pose.p.x, allegro_hand_start_pose.p.y, allegro_hand_start_pose.p.z,
                 allegro_hand_start_pose.r.x, allegro_hand_start_pose.r.y, allegro_hand_start_pose.r.z, allegro_hand_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])

            # 设置右手关节属性并记录其全局索引
            self.gym.set_actor_dof_properties(
                env_ptr, allegro_hand_actor, allegro_hand_dof_props)
            hand_idx = self.gym.get_actor_index(
                env_ptr, allegro_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # 设置左手关节属性并记录其全局索引
            self.gym.set_actor_dof_properties(
                env_ptr, allegro_hand_another_actor, allegro_hand_another_dof_props)
            another_hand_idx = self.gym.get_actor_index(
                env_ptr, allegro_hand_another_actor, gymapi.DOMAIN_SIM)
            self.another_hand_indices.append(another_hand_idx)

            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(
                env_ptr, allegro_hand_actor)
            hand_rigid_body_index = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [
                12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]

            # 按环境编号循环选择训练物体（实现多物体在不同环境中的随机分配）
            index = i % len(self.used_training_objects)
            select_obj = self.used_training_objects[index]
            # 创建操作物体 Actor
            object_handle = self.gym.create_actor(
                env_ptr, self.object_asset_dict[select_obj]['obj'], object_start_pose, "object", i, 0, 0)

            # object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append(
                [object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                 object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            lego_body_props = self.gym.get_actor_rigid_body_properties(
                env_ptr, object_handle)
            for lego_body_prop in lego_body_props:
                lego_body_prop.mass *= 1
            self.gym.set_actor_rigid_body_properties(
                env_ptr, object_handle, lego_body_props)

            object_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, object_handle)
            for object_shape_prop in object_shape_props:
                object_shape_prop.restitution = 0
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, object_handle, object_shape_props)

            hand_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, allegro_hand_actor)
            for hand_shape_prop in hand_shape_props:
                hand_shape_prop.restitution = 0.
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, object_handle, hand_shape_props)

            # 创建目标指示器 Actor（用于可视化显示目标位置，不与物体交互）
            goal_handle = self.gym.create_actor(
                env_ptr, self.object_asset_dict[select_obj]['goal'], goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(
                env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            # 创建预测目标指示器 Actor（显示 TrajEstimator 的预测结果）
            predict_goal_handle = self.gym.create_actor(
                env_ptr, self.object_asset_dict[select_obj]['predict goal'], goal_start_pose, "predict_goal_object", i + self.num_envs * 2, 0, 0)
            predict_goal_object_idx = self.gym.get_actor_index(
                env_ptr, predict_goal_handle, gymapi.DOMAIN_SIM)
            self.predict_goal_object_indices.append(predict_goal_object_idx)
            self.gym.set_rigid_body_color(
                env_ptr, predict_goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.4, 0.))
            # self.gym.set_actor_scale(env_ptr, predict_goal_handle, 0.01)

            if self.enable_camera_sensors:
                camera_handle = self.gym.create_camera_sensor(
                    env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(
                    0, -0.3, 0.43), gymapi.Vec3(0, -0.55, 0))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(
                    self.sim, env_ptr, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(
                    self.sim, env_ptr, camera_handle), device=self.device)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_hand_actor)

            if self.enable_camera_sensors:
                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)

        sensor_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, allegro_hand_another_actor, sensor_name) for sensor_name in
                          self.contact_sensor_names]

        self.sensor_handle_indices = to_torch(
            sensor_handles, dtype=torch.int64)

        object_rb_props = self.gym.get_actor_rigid_body_properties(
            env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(
            self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.hand_indices = to_torch(
            self.hand_indices, dtype=torch.long, device=self.device)
        self.another_hand_indices = to_torch(
            self.another_hand_indices, dtype=torch.long, device=self.device)

        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(
            self.goal_object_indices, dtype=torch.long, device=self.device)
        self.predict_goal_object_indices = to_torch(
            self.predict_goal_object_indices, dtype=torch.long, device=self.device)

        self.init_object_tracking = True
        self.test_for_robot_controller = False

        self.p_gain_val = 100.0
        self.d_gain_val = 4.0
        self.p_gain = torch.ones((self.num_envs, self.num_allegro_hand_dofs * 2),
                                 device=self.device, dtype=torch.float) * self.p_gain_val
        self.d_gain = torch.ones((self.num_envs, self.num_allegro_hand_dofs * 2),
                                 device=self.device, dtype=torch.float) * self.d_gain_val

        self.pd_previous_dof_pos = torch.zeros(
            (self.num_envs, self.num_allegro_hand_dofs * 2), device=self.device, dtype=torch.float) * self.p_gain_val
        self.pd_dof_pos = torch.zeros((self.num_envs, self.num_allegro_hand_dofs * 2),
                                      device=self.device, dtype=torch.float) * self.p_gain_val

        self.debug_target = []
        self.debug_qpos = []

        # 初始化轨迹估计器网络（输入：20帧×3维物体位置历史 = 60维，输出：3维预测目标位置）
        self.traj_estimator = TrajEstimator(
            input_dim=60, output_dim=3).to(self.device)
        for param in self.traj_estimator.parameters():
            param.requires_grad_(True)

        self.is_test = self.cfg["is_test"]

        # 轨迹估计器的 Adam 优化器
        self.traj_estimator_optimizer = torch.optim.Adam(
            self.traj_estimator.parameters(), lr=0.0003)
        self.traj_estimator_save_path = "./traj_e/"
        os.makedirs(self.traj_estimator_save_path, exist_ok=True)
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss()

        if self.is_test:
            # 测试模式：加载训练好的轨迹估计器权重
            self.traj_estimator.load_state_dict(torch.load(
                "./traj_e/model.pt", map_location='cuda:0'))
            self.traj_estimator.eval()
        else:
            # self.traj_estimator.load_state_dict(torch.load("./traj_e/model_perfect.pt", map_location='cuda:0'))
            self.traj_estimator.train()

        self.total_steps = 0
        self.success_attempts = 0
        self.total_attempts = 0
        self.success_buf = torch.zeros_like(self.rew_buf)
        self.hit_success_buf = torch.zeros_like(self.rew_buf)
        self.reset_no_fall_buf = torch.zeros_like(self.rew_buf)
        self.grasp_success_buf = torch.zeros_like(self.rew_buf)
        self.grasp_episode_success_buf = torch.zeros_like(self.rew_buf)
        self.finger_contacts = torch.zeros(
            (self.num_envs, len(self.contact_sensor_names)), dtype=torch.float, device=self.device)

    def get_internal_state(self):
        return self.root_state_tensor[self.object_indices, 3:7]

    def get_internal_info(self, key):
        if key == 'target':
            return self.debug_target
        elif key == 'qpos':
            return self.debug_qpos
        elif key == 'contact':
            return self.finger_contacts
        return None

    def compute_reward(self, actions):
        # 计算当前环境步骤下的综合奖励（包含位置靠近度、方向旋转匹配度、动作控制惩罚、任务成功奖励与掉落惩罚项）
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:], self.reset_no_fall_buf[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.allegro_left_hand_pos, self.allegro_right_hand_pos, 
            self.allegro_hand_another_thmub_pos, self.aux_up_pos, self.object_linvel, self.leeft_hand_ee_rot,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale, self.allegro_hand_another_ff_pos, self.allegro_hand_another_mf_pos, 
            self.allegro_hand_another_rf_pos, self.allegro_hand_ff_pos, self.allegro_hand_mf_pos, self.allegro_hand_rf_pos, self.a_hand_palm_pos, 
            unscale(self.another_allegro_hand_default_dof_pos[6:],
            self.allegro_hand_dof_lower_limits[6:22], self.allegro_hand_dof_upper_limits[6:22]), unscale(self.allegro_hand_another_dof_pos[:, 6:22],
            self.allegro_hand_dof_lower_limits[6:22], self.allegro_hand_dof_upper_limits[6:22]),
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (
                self.object_type == "pen")
        )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes
        self.extras['reset_no_fall'] = self.reset_no_fall_buf

        # 更偏向“接住并抓稳”的判据：
        # 1) 这一回合确实是成功回合（reset_no_fall）
        # 2) 至少有多个触觉/接触点参与
        # 3) 物体离手掌足够近
        # 4) 物体速度足够低，说明不是刚碰到就飞走
        # 5) 物体高度仍然保持在合理范围内，避免把下落过程误判为成功

        # contact_count = torch.sum(self.finger_contacts > 0.5, dim=-1)
        palm_obj_dist = torch.norm(
            self.a_hand_palm_pos - self.object_pos, p=2, dim=-1)
        object_speed = torch.norm(self.object_linvel, p=2, dim=-1)
        # min_contact_count = 1
        max_palm_obj_dist = 0.14
        max_object_speed = 0.25
        min_object_height = 0.12
        stable_grasp = (
            # (contact_count >= min_contact_count) &
            (palm_obj_dist < max_palm_obj_dist) &
            (object_speed < max_object_speed) &
            (self.object_pos[:, 2] > min_object_height)
        )
        new_grasp_success = stable_grasp & (self.grasp_episode_success_buf == 0)
        self.grasp_episode_success_buf = torch.where(
            new_grasp_success,
            torch.ones_like(self.grasp_episode_success_buf),
            self.grasp_episode_success_buf)
        self.grasp_success_buf[:] = self.grasp_episode_success_buf
        self.extras['grasp_success'] = self.grasp_success_buf

        self.total_steps += self.num_envs
        current_attempts = int(self.reset_buf.sum().item())
        current_successes = int(new_grasp_success.sum().item())
        self.total_attempts += current_attempts
        self.success_attempts += current_successes

        success_rate = float(self.success_attempts / self.total_attempts) if self.total_attempts > 0 else 0.0
        self.writter.add_scalar('Total Attempts', float(self.total_attempts), self.total_steps)
        self.writter.add_scalar('Successful Throws and Catches', float(self.success_attempts), self.total_steps)
        self.writter.add_scalar('Success Rate', success_rate, self.total_steps)

        print('total_steps', self.total_steps,
            'total_attempts', self.total_attempts,
            'success_attempts', self.success_attempts,
            'grasp_success_buf', self.grasp_success_buf.sum().item(),
            'success rate', success_rate)

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + \
                (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print('successes', self.successes.sum().item(), 'resets', self.reset_buf.sum().item(),
                  'total_resets', self.total_resets.item(), 'total_successes', self.total_successes.item())
            print("Direct average consecutive successes = {:.3f}".format(
                direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.3f}".format(
                    self.total_successes/self.total_resets))

    def compute_observations(self):
        # 刷新仿真器内部的状态张量缓存，包括关节状态、物体刚体根状态、触力张量及雅可比矩阵
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.allegro_right_hand_base_pos = self.root_state_tensor[self.hand_indices, 0:3]
        self.allegro_right_hand_base_rot = self.root_state_tensor[self.hand_indices, 3:7]

        self.allegro_right_hand_pos = self.rigid_body_states[:, 6, 0:3]
        self.allegro_right_hand_rot = self.rigid_body_states[:, 6, 3:7]

        self.allegro_left_hand_pos = self.rigid_body_states[:, 6 + 23, 0:3]
        self.allegro_left_hand_rot = self.rigid_body_states[:, 6 + 23, 3:7]

        self.a_hand_palm_pos = self.allegro_left_hand_pos.clone()

        self.allegro_left_hand_pos = self.allegro_left_hand_pos + \
            quat_apply(self.allegro_left_hand_rot, to_torch(
                [0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.allegro_left_hand_pos = self.allegro_left_hand_pos + \
            quat_apply(self.allegro_left_hand_rot, to_torch(
                [0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.allegro_hand_another_thmub_pos = self.rigid_body_states[:, 14 + 23, 0:3]
        self.allegro_hand_another_thmub_rot = self.rigid_body_states[:, 14 + 23, 3:7]

        self.allegro_hand_another_ff_pos = self.rigid_body_states[:, 10, 0:3]
        self.allegro_hand_another_mf_pos = self.rigid_body_states[:, 18, 0:3]
        self.allegro_hand_another_rf_pos = self.rigid_body_states[:, 22, 0:3]

        self.allegro_hand_ff_pos = self.rigid_body_states[:, 10 + 23, 0:3]
        self.allegro_hand_mf_pos = self.rigid_body_states[:, 18 + 23, 0:3]
        self.allegro_hand_rf_pos = self.rigid_body_states[:, 22 + 23, 0:3]

        self.leeft_hand_ee_rot = self.rigid_body_states[
            :, self.another_hand_base_rigid_body_index, 3:7]

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0,
                                       (self.num_envs, 63), device=self.device)

        self.aux_up_pos = to_torch(
            [0, -0.52, 0.45], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.compute_sim2real_observation(rand_floats)

        if self.asymmetric_obs:
            self.compute_sim2real_asymmetric_obs(rand_floats)

    def compute_sim2real_observation(self, rand_floats):
        """ 构建 sim-to-real 兼容的观测张量，将关节位置归一化后按智能体分段填充到 obs_buf 中。 """
        # obs 通道 0~150：右手（hand）的观测数据
        # 归一化右手全部22个关节位置到 [-1, 1]
        self.obs_buf[:, 0:22] = unscale(
            self.allegro_hand_dof_pos,
            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)

        self.obs_buf[:, 0:6] = 0
        self.obs_buf[:, 1] = self.allegro_hand_dof_pos[:, 1]
        self.obs_buf[:, 2] = self.allegro_hand_dof_pos[:, 2]

        self.obs_buf[:, 22:25] = (
            self.goal_pos - self.allegro_right_hand_base_pos).clone()

        # obs 通道 150~300：左手（another hand）的观测数据
        # 归一化左手全部22个关节位置
        self.obs_buf[:, 150:172] = unscale(
            self.allegro_hand_another_dof_pos,
            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        # 覆盖左手臂部分关键关节为原始值（sim2real 需要）
        self.obs_buf[:, 150:156] = self.allegro_hand_another_dof_pos[:, :6]
        self.obs_buf[:, 150:151] = 0
        self.obs_buf[:, 153:154] = 0

        # 维护物体位置的滑动窗口序列（长度为 object_seq_len=20，每帧3维）
        for i in range(self.object_seq_len):
            if i == self.object_seq_len - 1:
                self.object_state_stack_frames[:, (i)*3:(i+1)*3] = (
                    self.object_pos - self.allegro_right_hand_base_pos).clone()
            else:
                self.object_state_stack_frames[:, (i)*3:(
                    i+1)*3] = self.object_state_stack_frames[:, (i+1)*3:(i+2)*3].clone()

        # 在临时梯度模式下运行轨迹估计器，并在线更新其参数
        with TemporaryGrad():
            # 用物体位置序列预测目标接触位置
            self.predict_pose, self.pose_latent_vector = self.predict_contact_pose(
                self.traj_estimator, self.object_state_stack_frames)
            # 用真实目标位置监督轨迹估计器进行在线学习
            self.update_contact_slamer(self.predict_pose)

        # 将预测位姿和物体近期位置历史写入观测缓冲区
        self.obs_buf[:, 260:263] = self.predict_pose[:, 0:3].detach()
        self.obs_buf[:, 248:260] = self.object_state_stack_frames[
            :, 36:48].clone() + rand_floats[:, 0:12] * 0.05

        # 帧堆叠：将当前帧观测压入栈中，构成多帧观测输入
        for i in range(len(self.obs_buf_stack_frames) - 1):
            self.obs_buf[:, (i+1) * self.one_frame_num_obs:(i+2) *
                         self.one_frame_num_obs] = self.obs_buf_stack_frames[i]
            self.obs_buf_stack_frames[i] = self.obs_buf[:, (
                i) * self.one_frame_num_obs:(i+1) * self.one_frame_num_obs].clone()

    def predict_contact_pose(self, traj_estimator, contact_buf):
        """ 调用轨迹估计器网络进行前向传播，预测物体的目标接触位置。 """
        predict_pose, pose_latent_vector = traj_estimator(contact_buf)
        return predict_pose, pose_latent_vector

    def update_contact_slamer(self, predict_pose):
        """ 在线更新轨迹估计器：用预测位置与真实目标位置的 MSE 损失进行梯度下降。 """
        self.pos_loss = F.mse_loss(
            predict_pose[:, 0:3], (self.goal_pos - self.allegro_right_hand_base_pos).clone())
        loss = self.pos_loss
        # 清零梯度 → 反向传播 → 更新参数
        self.traj_estimator_optimizer.zero_grad()
        loss.backward()
        self.traj_estimator_optimizer.step()
        self.extras['pos_loss'] = self.pos_loss.unsqueeze(0)

    def compute_sim2real_asymmetric_obs(self, rand_floats):
        """ 构建非对称全局状态（用于 CTDE 的 Critic），包含双手完整信息、触觉、物体位姿和目标位置。 """
        # 右手关节位置（归一化）
        self.states_buf[:, 0:self.num_allegro_hand_dofs] = unscale(
            self.allegro_hand_dof_pos,
            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        # 右手关节速度
        self.states_buf[
            :, self.num_allegro_hand_dofs:2 *
            self.num_allegro_hand_dofs] = self.vel_obs_scale * self.allegro_hand_dof_vel

        # 右手上一步的动作
        action_obs_start = 44
        self.states_buf[:, action_obs_start:action_obs_start +
                        22] = self.actions[:, :22]

        # 左手关节位置（归一化）和速度
        another_hand_start = action_obs_start + 22
        self.states_buf[
            :, another_hand_start:self.num_allegro_hand_dofs + another_hand_start] = unscale(self.allegro_hand_another_dof_pos,
                                                                                             self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.states_buf[
            :, self.num_allegro_hand_dofs + another_hand_start:2*self.num_allegro_hand_dofs +
            another_hand_start] = self.vel_obs_scale * self.allegro_hand_another_dof_vel

        # 左手上一步的动作
        action_another_obs_start = another_hand_start + 44
        self.states_buf[
            :, action_another_obs_start:action_another_obs_start +
            22] = self.actions[:, 22:]

        # 左手触觉接触信息：提取指定传感器位置的接触力范数，二值化为接触/未接触
        contact_start = action_another_obs_start + 22
        contacts = self.contact_tensor.reshape(self.num_envs, -1, 3)
        contacts = contacts[:, self.sensor_handle_indices, :]  # 选取 11 个指尖传感器
        contacts = torch.norm(contacts, dim=-1)
        contacts = torch.where(contacts >= 1.0, 1.0, 0.0)  # 阈值为 1.0 N
        self.finger_contacts = contacts
        self.states_buf[:, contact_start:contact_start + 11] = contacts

        # 物体的位姿、线速度和角速度
        obj_obs_start = contact_start + 11
        self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.states_buf[:, obj_obs_start +
                        7:obj_obs_start + 10] = self.object_linvel
        self.states_buf[:, obj_obs_start + 10:obj_obs_start +
                        13] = self.vel_obs_scale * self.object_angvel

        # 目标位姿和物体与目标之间的旋转差异
        goal_obs_start = obj_obs_start + 13
        self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        self.states_buf[:, goal_obs_start + 7:goal_obs_start +
                        11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

        # 随机化噪声参数和带噪声的物体位置序列
        randomize_param_start = goal_obs_start + 11
        self.states_buf[
            :, randomize_param_start:randomize_param_start +
            12] = rand_floats[:, 0:12]

        offseted_pos_goal_start = randomize_param_start + 12
        self.states_buf[
            :, offseted_pos_goal_start:offseted_pos_goal_start +
            12] = self.object_state_stack_frames[:, 36:48].clone() + rand_floats[:, 0:12] * 0.05
        # 真实目标位置（相对于右手基座）
        self.states_buf[
            :, offseted_pos_goal_start + 12:offseted_pos_goal_start +
            15] = (self.goal_pos - self.allegro_right_hand_base_pos).clone()

    def reset_target_pose(self, env_ids, apply_reset=False):
        """ 重置目标位置：在初始位置基础上添加随机扰动，并更新目标指示器的根状态。 """
        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), 4), device=self.device)

        # 在初始目标位置基础上添加随机偏移
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]

        self.goal_states[env_ids, 0] += rand_floats[:, 0] * 0.05
        # self.goal_states[env_ids, 1] -= 0.45 + rand_floats[:, 1] * 0.15
        # self.goal_states[env_ids, 2] += 0.1

        self.goal_states[env_ids, 1] -= 0.55 + rand_floats[:, 1] * 0.05
        self.goal_states[env_ids, 2] += 0.1

        # self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[
            self.goal_object_indices[env_ids],
            0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        quat = quat_from_euler_xyz(torch.sign(rand_floats[:, 0]) * 3.1415, torch.sign(
            rand_floats[:, 1]) * 3.1415 + 1.571, torch.sign(rand_floats[:, 2]) * 3.14)
        self.root_state_tensor[
            self.goal_object_indices[env_ids], 3] = quat[:, 0]
        self.root_state_tensor[
            self.goal_object_indices[env_ids], 4] = quat[:, 1]
        self.root_state_tensor[
            self.goal_object_indices[env_ids], 5] = quat[:, 2]
        self.root_state_tensor[
            self.goal_object_indices[env_ids], 6] = quat[:, 3]
        self.root_state_tensor[
            self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            # 如果需要立即生效，则将目标指示器的新状态写入仿真器
            goal_object_indices = self.goal_object_indices[env_ids].to(
                torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(
                    self.root_state_tensor),
                gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):
        # 环境重置逻辑：主要用于重置环境观测、清除探索轨迹并更新动力学随机化参数
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.perturb_direction[env_ids] = torch_rand_float(
            -1, 1, (len(env_ids), 6), device=self.device).squeeze(-1)

        # generate random values
        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids),
                        self.num_allegro_hand_dofs * 2 + 5), device=self.device)

        # 随机化并重新设置环境的初始目标姿态
        self.reset_target_pose(env_ids)

        # self.root_state_tensor[self.another_hand_indices[env_ids], 2] = -0.05 + rand_floats[:, 4] * 0.01

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]
                               ] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        quat = quat_from_euler_xyz(torch.sign(rand_floats[:, 3]) * 3.1415, torch.sign(
            rand_floats[:, 4]) * 3.1415 + 1.571, torch.sign(rand_floats[:, 5]) * 3.14)
        self.root_state_tensor[self.object_indices[env_ids], 3] = quat[:, 0]
        self.root_state_tensor[self.object_indices[env_ids], 4] = quat[:, 1]
        self.root_state_tensor[self.object_indices[env_ids], 5] = quat[:, 2]
        self.root_state_tensor[self.object_indices[env_ids], 6] = quat[:, 3]
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13])

        self.object_pose_for_open_loop[env_ids] = self.root_state_tensor[self.object_indices[env_ids], 0:7].clone(
        )

        object_indices = torch.unique(
            torch.cat([self.object_indices[env_ids],
                       self.goal_object_indices[env_ids],
                       self.predict_goal_object_indices[env_ids],
                       self.goal_object_indices[goal_env_ids]]).to(torch.int32))

        # reset shadow hand
        pos = self.allegro_hand_default_dof_pos
        another_pos = self.another_allegro_hand_default_dof_pos

        self.allegro_hand_dof_pos[env_ids, :] = pos
        self.allegro_hand_another_dof_pos[env_ids, :] = another_pos

        self.allegro_hand_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * \
            rand_floats[:, 5+self.num_allegro_hand_dofs:5 +
                        self.num_allegro_hand_dofs*2]

        self.allegro_hand_another_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * \
            rand_floats[:, 5+self.num_allegro_hand_dofs:5 +
                        self.num_allegro_hand_dofs*2]

        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = pos

        self.prev_targets[env_ids,
                          self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2] = another_pos
        self.cur_targets[env_ids,
                         self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2] = another_pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        another_hand_indices = self.another_hand_indices[env_ids].to(
            torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices,
                                                   another_hand_indices]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(
                                                            self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              object_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.grasp_episode_success_buf[env_ids] = 0

        self.proprioception_close_loop[env_ids] = self.allegro_hand_dof_pos[env_ids, 0:22].clone(
        )

        self.object_state_stack_frames[env_ids] = torch.zeros_like(
            self.object_state_stack_frames[env_ids])

        # if 1 in env_ids:
        self.init_object_tracking = True
        self.gym.clear_lines(self.viewer)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # 如果部分环境的目标需要重置，则更新对应的目标属性参数
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # 如果是整条环境重置，reset() 内部会统一写入设备
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        # 重置已经达到最大时间步长或满足终止条件的环境
        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        # 将策略输出的手指动作从 [-1,1] 缩放到真实关节范围
        self.cur_targets[:, self.actuated_dof_indices + 6] = scale(
            self.actions[:, 6:22],
            self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
        self.cur_targets[:, self.actuated_dof_indices + 28] = scale(
            self.actions[:, 28:44],
            self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])

        # 机械臂关节使用增量控制（相对前一帧的位移）
        self.cur_targets[:, [1, 2]] = self.prev_targets[
            :,
            [1, 2]] + self.actions[:, [1, 2]] * 2
        self.cur_targets[:, 22:28] = self.prev_targets[
            :,
            22:28] + self.actions[:, 22:28] * 0.1

        # 对手指关节目标位置应用指数移动平均平滑（EMA），减少抖动
        self.cur_targets[:, self.actuated_dof_indices + 6] = self.act_moving_average * self.cur_targets[
            :,
            self.actuated_dof_indices + 6] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices + 6]

        self.cur_targets[:, self.actuated_dof_indices + 28] = self.act_moving_average * self.cur_targets[
            :,
            self.actuated_dof_indices + 22] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices + 22]

        # 裁剪关节目标到安全范围内
        self.cur_targets[:, 0:22] = tensor_clamp(
            self.cur_targets[:, 0:22],
            self.allegro_hand_dof_lower_limits[:],
            self.allegro_hand_dof_upper_limits[:])

        self.cur_targets[:, 22:44] = tensor_clamp(
            self.cur_targets[:, 22:44],
            self.a_allegro_hand_dof_lower_limits[:],
            self.a_allegro_hand_dof_upper_limits[:])

        # 保存当前帧目标并将关节位置目标写入仿真器
        self.prev_targets[:, :] = self.cur_targets[:, :]
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # 更新预测目标指示器的可视化位置（逐步插值趋近真实目标）
        self.root_state_tensor[self.predict_goal_object_indices, 0:3] = self.predict_pose[:, 0:3].detach() + self.root_state_tensor[self.hand_indices, 0:3] - (
            (self.predict_pose[:, 0:3].detach() + self.root_state_tensor[self.hand_indices, 0:3]) - self.goal_pos) * torch.clamp(self.progress_buf[0] * random.random() / 10, 0, 1)
        self.root_state_tensor[self.predict_goal_object_indices,
                               3:7] = self.root_state_tensor[self.object_indices, 3:7].detach()

        # 将物体和目标指示器的更新后根状态批量写入仿真器
        object_indices = torch.unique(
            torch.cat([self.object_indices,
                       self.goal_object_indices,
                       self.predict_goal_object_indices]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(
                self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices.to(torch.int32)), len(object_indices.to(torch.int32)))

        # 定期保存轨迹估计器的模型权重（每 200 个 episode 保存一次）
        if self.total_steps % (200 * (self.max_episode_length - 1)) == 0:
            iter = int(self.total_steps /
                       (200 * (self.max_episode_length - 1)))
            if not self.is_test:
                torch.save(self.traj_estimator.state_dict(),
                           self.traj_estimator_save_path + "/model.pt")

        self.apply_force = False
        if self.apply_force == True:
            # apply new force
            self.rb_forces[:] = torch.zeros_like(self.rb_forces)

            self.apply_force_env_id = torch.where(
                self.root_state_tensor[self.object_indices, 1] < 10.1,
                torch.where(self.root_state_tensor[self.object_indices, 1] > -10.9, 1, 0,), 0).nonzero(as_tuple=False).squeeze(-1)

            self.perturb_direction[self.apply_force_env_id, 0] = 0
            self.perturb_direction[self.apply_force_env_id, 1] = 0
            self.perturb_direction[self.apply_force_env_id, 2] = 1

            self.rb_forces[self.apply_force_env_id, self.object_rb_handles, 0:3] = torch.ones(
                self.rb_forces[self.apply_force_env_id, self.object_rb_handles, 0:3].shape, device=self.device) * 1.5 * self.perturb_direction[self.apply_force_env_id, 0:3].squeeze(-1)
            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

            if 0 in self.apply_force_env_id:
                self.gym.set_rigid_body_color(
                    self.envs[0], self.object_indices[0], 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.3, 0.3))
            else:
                self.gym.set_rigid_body_color(
                    self.envs[0], self.object_indices[0], 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 1))

    def post_physics_step(self):
        # 物理引擎步进后置处理：步长计数自增，并实时计算最新的智能体观测张量和环境奖励
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)
        # self.gym.clear_lines(self.viewer)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        # self.add_debug_lines(self.envs[0], self.a_hand_palm_pos[0], self.allegro_left_hand_rot[0], line_width=2)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(
                    self.envs[i], self.allegro_hand_another_thmub_pos[i],
                    self.allegro_hand_another_thmub_rot[i], line_width=2)
                # self.add_debug_lines(self.envs[i], self.allegro_left_hand_pos[i], self.allegro_left_hand_rot[i])

    def add_debug_lines(self, env, pos, rot, line_width=1):
        """ 在可视化窗口中绘制三轴坐标系调试线（红=X轴, 绿=Y轴, 蓝=Z轴）。 """
        posx = (pos + quat_apply(rot,
                to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot,
                to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot,
                to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, line_width, [
                           # X轴红色
                           p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, line_width, [
                           # Y轴绿色
                           p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, line_width, [
                           # Z轴蓝色
                           p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

#####################################################################
### ===== JIT 编译的奖励和辅助函数（纯张量运算，无 Python 开销） =====###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, allegro_left_hand_pos, 
    allegro_right_hand_pos, allegro_another_hand_thmub_pos, aux_up_pos, object_vel, leeft_hand_ee_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float, allegro_hand_another_ff_pos, allegro_hand_another_mf_pos, 
    allegro_hand_another_rf_pos, allegro_hand_ff_pos, allegro_hand_mf_pos, allegro_hand_rf_pos, a_hand_palm_pos, hand_init_qpos, hand_qpos,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    """
    计算递物任务的综合奖励，包含：
    - 物体到目标的距离奖励（指数衰减形式）
    - 物体在递物方向上的速度奖励
    - 动作幅度惩罚（鼓励平滑控制）
    - 成功到达目标的额外奖励
    - 物体掉落的惩罚
    同时更新环境的重置标志和连续成功计数。
    """
    # 计算物体与目标之间的欧氏距离
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

    # 拇指到物体的距离（可选用于辅助奖励）
    thmub_dist = torch.norm(
        allegro_another_hand_thmub_pos - object_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # 计算物体当前朝向与目标朝向之间的旋转距离
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * \
        torch.asin(torch.clamp(torch.norm(
            quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist

    # 动作幅度惩罚：抑制过大的控制信号
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # 物体沿递物方向（-y轴）的速度奖励：仅在物体处于传递区间时给予
    object_vel_reward = torch.clamp(
        -object_vel[:, 1], -0.1, 0.1)
    object_vel_reward = torch.where(
        object_pos[:, 1] < -0.65,
        torch.where(-0.85 < object_pos[:, 1], object_vel_reward, torch.zeros_like(object_vel_reward)), torch.zeros_like(object_vel_reward))

    # 综合奖励 = 距离奖励（指数衰减） + 速度奖励 - 动作惩罚
    reward = (torch.exp(-4*((3 * dist_rew))) +
              object_vel_reward) - 0.001 * action_penalty

    # 判断哪些环境已到达目标
    goal_resets = torch.where(
        torch.abs(goal_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # 成功奖励：到达目标时给予额外奖励
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # 掉落惩罚：物体 z 坐标低于 0.15 时判定为掉落
    # reward = torch.where(object_pos[:, 2] <= 0.2, reward + fall_penalty, reward)

    # 终止条件检查：物体掉落
    resets = torch.where(object_pos[:, 2] <= 0.15,
                         torch.ones_like(reset_buf), reset_buf)

    if max_consecutive_successes > 0:
        # 如果旋转误差在容差范围内，重置进度计数器
        progress_buf = torch.where(torch.abs(
            rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        # 达到最大连续成功次数时终止 episode
        resets = torch.where(
            successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    # 达到最大 episode 长度时终止
    resets = torch.where(progress_buf >= max_episode_length,
                         torch.ones_like(resets), resets)

    # 超时未完成的额外惩罚
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length,
                             reward + 0.5 * fall_penalty, reward)

    # 计算连续成功的滑动平均统计
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0, av_factor*finished_cons_successes /
        num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    reset_no_fall = torch.where(
        (resets == 1) & (object_pos[:, 2] > 0.15) &
        (successes >= max_consecutive_successes) &
        (goal_resets == 1),
        torch.ones_like(resets), torch.zeros_like(resets))

    return reward, resets, goal_resets, progress_buf, successes, cons_successes, reset_no_fall


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """ 通过两个随机角度绕 X 轴和 Y 轴组合旋转，生成随机朝向四元数。 """
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    """ 专用于笔类物体的旋转随机化（限制最大角度偏移）。 """
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot


def orientation_error(desired, current):
    """ 计算当前朝向与期望朝向之间的旋转误差（返回旋转向量形式）。 """
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik(j_eef, device, dpose, num_envs):
    """ 逆运动学控制器：使用阻尼最小二乘法（DLS）从末端执行器位姿误差求解关节增量。 """
    # 阻尼系数（damping）防止雅可比矩阵奇异
    damping = 0.05
    # 求解阻尼最小二乘：u = J^T * (J*J^T + λI)^{-1} * Δpose
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda)
         @ dpose).view(num_envs, -1)
    return u
