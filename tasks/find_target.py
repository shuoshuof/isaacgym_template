from isaacgym import gymutil, gymtorch, gymapi
import random
import torch
import math
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import *
from typing import Tuple
from isaacgym.terrain_utils import *

from tasks.visualization_utils import *

class FindTarget(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # TODO : 归一化系数和action的系数需要调, dof_vel_scale 似乎也有问题
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        # self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang


        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # TODO: set in config file
        visualize_env_boundaries(self.gym,self.viewer,self.env_handles, self.cfg["env"]['envSpacing'])

        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)


        # TODO: ???
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        # TODO: need to set a viewer?

        # root state when actor was created
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]

        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # lidar settings
        self.height_points = self.init_height_points()

        # TODO: ???
        self.reset_idx(torch.arange(self.num_envs, device=self.device))


        # self.sim = None
        # self.gym = gymapi.acquire_gym()
        #
        # self.device = 'cpu'
        # self.num_actions = 2
        # self.num_envs = self.cfg['env']['numEnvs']
        # self.envs_per_row = int(math.sqrt(self.num_envs))
        # self.env_spacing = self.cfg['env']['envSpacing']
        #
        # self.create_sim()
    def create_sim(self):
        # sim_params = self.set_sim_parameters()
        self.up_axis_idx = 2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # self._create_ground_plane()
        self._create_terrain()
        self._create_envs(self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_terrain(self):
        from tasks.terrains.create_terrain import OneTimeTerrainGenerator
        from tasks.terrains.perlin_terrain import TerrainPerlin

        # generator = TerrainGenerator(self.cfg["env"]["terrain"],self.num_envs,env_spacing=self.cfg["env"]['envSpacing'])
        # generator.randomized_terrain(self.gym,self.sim)
        # generator.add_boundary(self.gym,self.sim)
        # generator = MoJiaoTerrainGenerator(self.cfg["env"]["terrain"],self.num_envs,self.cfg["env"]['envSpacing'],self.gym,self.sim)

        # generator = OneTimeTerrainGenerator(self.cfg["env"]["terrain"],self.num_envs,self.cfg["env"]['envSpacing'],self.gym,self.sim)
        # self.height_samples = generator.get_height_samples()
        perlin_generator = TerrainPerlin(self.cfg["env"]["terrain"],self.num_envs,self.cfg["env"]['envSpacing'])
        perlin_generator.add_terrain_to_sim(self.gym,self.sim,)

    def _add_target(self, env, env_index):
        cube_size = 0.1
        asset_options = gymapi.AssetOptions()
        asset_options.density = 0.001
        cube_asset = self.gym.create_box(self.sim, cube_size, cube_size, cube_size, asset_options)

        # 定义方块的初始姿态
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, cube_size/2)  # 方块的初始位置 (x, y, z)
        pose.r = gymapi.Quat(0, 0, 0, 1)  # 方块的初始旋转
        cube_actor = self.gym.create_actor(env, cube_asset, pose, "cube", env_index, 1)
        return cube_actor

    def _create_envs(self,spacing, num_per_row):
        asset_root = self.cfg['env']['asset']['assetRoot']
        asset_file = self.cfg['env']['asset']['assetFileName']

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = False

        car_asset = self.gym.load_asset(self.sim, asset_root, asset_file,asset_options)
        self.num_dof = self.gym.get_asset_dof_count(car_asset)

        self.num_bodies = self.gym.get_asset_rigid_body_count(car_asset)

        body_names = self.gym.get_asset_rigid_body_names(car_asset)
        self.body_dict = {value: index
                         for index, value in enumerate(body_names)}
        self.dof_names = self.gym.get_asset_dof_names(car_asset)
        self.dof_dict = {value: index
                         for index, value in enumerate(self.dof_names)}


        env_lower = gymapi.Vec3(-spacing, -spacing, 0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # cache some common handles for later use
        self.env_handles = []
        self.actor_handles = []
        self.cube_handles = []
        # create and populate the environments
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.env_handles.append(env_handle)


            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.09)
            actor_handle = self.gym.create_actor(env_handle, car_asset, pose, "MyActor", i, 1)
            self._set_dof_properties(env_handle, actor_handle)
            self._set_rigid_friction(env_handle, actor_handle)
            self.actor_handles.append(actor_handle)


            # cube_handle = self._add_target(env_handle, i)
            # self.cube_handles .append(cube_handle)
            # self.gym.set_rigid_body_color(env_handle, cube_handle, 0,
            #                               gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1.0, 0.0, 0.0))

        # TODO:  why just use idx 0
        self.base_index = self.gym.find_actor_rigid_body_handle(self.env_handles[0], self.actor_handles[0], "base_link")
    def _set_rigid_friction(self,env_handle,actor_handle):

        rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)

        # 调整摩擦力
        rigid_shape_props[self.body_dict['base_link']].friction = -1
        rigid_shape_props[self.body_dict['right_wheel_link']].friction = 1
        rigid_shape_props[self.body_dict['left_wheel_link']].friction = 1
        self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)
    def _set_dof_properties(self, env, actor_handle):
        dof_props = self.gym.get_actor_dof_properties(env, actor_handle)

        left_wheel_joint_index = self.dof_dict["left_wheel_joint"]
        dof_props["driveMode"][left_wheel_joint_index] = gymapi.DOF_MODE_VEL
        dof_props["stiffness"][left_wheel_joint_index] = self.Kp
        dof_props["damping"][left_wheel_joint_index] = self.Kd

        right_wheel_joint_index = self.dof_dict["right_wheel_joint"]
        dof_props["driveMode"][right_wheel_joint_index] = gymapi.DOF_MODE_VEL
        dof_props["stiffness"][right_wheel_joint_index] = self.Kp
        dof_props["damping"][right_wheel_joint_index] = self.Kd
        self.gym.set_actor_dof_properties(env, actor_handle, dof_props)

        left_dof_handle = self.gym.find_actor_dof_handle(env, actor_handle, 'left_wheel_joint')
        self.gym.set_dof_target_velocity(env, left_dof_handle, 10)

        right_dof_handle = self.gym.find_actor_dof_handle(env, actor_handle, 'right_wheel_joint')
        self.gym.set_dof_target_velocity(env, right_dof_handle, 10)

    def pre_physics_step(self, actions: torch.Tensor):
        # actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        # # de-normalize
        # actions_tensor[self.dof_dict["left_wheel_joint"]::self.num_dof] \
        #     = actions.to(self.device).view(-1)[0::self.num_actions]
        # actions_tensor[self.dof_dict["right_wheel_joint"]::self.num_dof] \
        #     = actions.to(self.device).view(-1)[1::self.num_actions]
        # velocity = gymtorch.unwrap_tensor(actions_tensor)
        # self.gym.set_dof_velocity_target_tensor(self.sim, velocity)
        self.actions = actions.clone().to(self.device)
        # de-normalize
        targets = self.action_scale*self.actions
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))
    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
    def reset_idx(self,env_ids):
        # set random dof  vel
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # set the root the same as the root state when actor was created
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # self.dof_vel is in self.dof_state
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1



    def compute_reward(self, actions):
        self.rew_buf[:] , self.reset_buf[:] = compute_car_reward(
            self.root_states,
            self.commands,
            self.progress_buf,
            self.rew_scales,
            self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.measured_heights = None

        self.obs_buf[:] = compute_car_observations(
            self.root_states,
            self.commands,
            self.dof_vel,
            self.gravity_vec,
            self.actions,
            self.lin_vel_scale,
            self.ang_vel_scale,
            self.dof_vel_scale
        )

    def init_height_points(self):

        x = 0.1 * torch.tensor([-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False)
        y = 0.1*torch.tensor([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x,y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_heights(self):
        base_quat = self.root_states[:, 3:7]
        points = quat_apply_yaw(base_quat.repeat(1, self.num_height_points),
                                self.height_points) + (self.root_states[:, :3]).unsqueeze(1)


    def debug_functions(self):
        actor_vel = self.root_states[:,7:10]
        print(actor_vel)
    def modify_actor_vel(self):
        self.vel+=1
        target_vel = torch.ones((self.num_envs, self.num_actions),device=self.device)*self.vel
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(target_vel.view(-1)))
        pass

    def run(self):
        # self.draw_boundaries(viewer)
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_A, "modify_vel")
        self.vel = 10
        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            # actions = torch.rand((self.num_envs, self.num_actions)) * 10
            # self.pre_physics_step(actions)

            # need to be refreshed or the state will not be updated
            self.post_physics_step()

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)



            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "modify_vel" and evt.value > 0:
                    self.modify_actor_vel()
            # self.debug_functions()

@torch.jit.script
def compute_car_observations(
        root_states,
        commands,
        dof_vel,
        gravity_vec,
        actions,
        lin_vel_scale,
        ang_vel_scale,
        dof_vel_scale
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float ) -> Tensor

    # 3 floats for position, 4 floats for quaternion,
    # 3 floats for linear velocity, and 3 floats for angular velocity.
    base_quat = root_states[:,3:7]
    # transform to the orientation that the robot is in
    print(quat_rotate_inverse(base_quat, root_states[:, 10:13]) )
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    # make robot know the orientation of the gravity
    projected_gravity = quat_rotate(base_quat, gravity_vec)

    commands_vel = commands*torch.tensor([lin_vel_scale,lin_vel_scale,ang_vel_scale],
                                            requires_grad=False, device= commands.device)

    # print(base_lin_vel.shape,base_ang_vel.shape,projected_gravity.shape,commands_vel.shape,dof_vel.shape,actions.shape)
    # 3 + 3 + 3 + 3 + 2 + 2 = 16
    return torch.cat(
        (base_lin_vel,
         base_ang_vel,
         projected_gravity,
         commands_vel,
         dof_vel*dof_vel_scale,
         actions),
        dim=1
    )




@torch.jit.script
def compute_car_reward(
        root_states,
        commands,
        episode_lengths,
        rew_scales,
        max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, float], int) -> Tuple[Tensor, Tensor]

    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]

    total_reward = rew_lin_vel_xy + rew_ang_vel_z

    # reset agents
    # TODO: any additional operations in time out
    time_out = episode_lengths >= max_episode_length-1
    reset = time_out

    return total_reward.detach(), reset

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles


import hydra
from omegaconf import DictConfig
from typing import Dict
from isaacgymenvs.utils.reformat import omegaconf_to_dict


@hydra.main(version_base="1.1", config_name="config.yaml", config_path="/home/shuof/work_project/isaacgym_template/cfg")
def launch_rlg_hydra(cfg: DictConfig):
    task_cfg = cfg.task
    task_cfg = omegaconf_to_dict(task_cfg)
    env = FindTarget(
        task_cfg,
        rl_device='cuda:0',
        sim_device='cuda:0',
        graphics_device_id=0,
        headless=False,
        virtual_screen_capture=False,
        force_render=True
    )
    env.run()

if __name__ == "__main__":
    launch_rlg_hydra()
