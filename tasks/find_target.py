from isaacgym import gymutil, gymtorch, gymapi
import random
import torch
import math

class FindTarget:
    def __init__(self,cfg):
        self.cfg = cfg
        self.sim = None
        self.gym = gymapi.acquire_gym()

        self.device = 'cpu'
        self.num_actions = 2
        self.num_envs = self.cfg['env']['numEnvs']
        self.envs_per_row = int(math.sqrt(self.num_envs))
        self.env_spacing = self.cfg['env']['envSpacing']

        self.create_sim()
    def create_sim(self):
        sim_params = self.set_sim_parameters()
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        self._create_ground_plane()
        self._create_envs()

    def set_sim_parameters(self):
        sim_params = gymapi.SimParams()

        # set common parameters
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        # set PhysX-specific parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        # set Flex-specific parameters
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 20
        sim_params.flex.relaxation = 0.8
        sim_params.flex.warm_start = 0.5
        return sim_params

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)

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

    def _create_envs(self):
        asset_root = self.cfg['env']['asset']['assetRoot']
        asset_file = self.cfg['env']['asset']['assetFileName']
        asset = self.gym.load_asset(self.sim, asset_root, asset_file)
        self.dof_dict = {value: index
                         for index, value in enumerate(self.gym.get_asset_dof_names(asset))}
        self.num_dof = self.gym.get_asset_dof_count(asset)



        env_lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0)
        env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)

        # cache some common handles for later use
        self.env_handles = []
        self.actor_handles = []
        self.cube_handles = []
        # create and populate the environments
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, self.envs_per_row)
            self.env_handles.append(env_handle)

            height = random.uniform(1.0, 2.5)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, height)

            actor_handle = self.gym.create_actor(env_handle, asset, pose, "MyActor", i, 1)
            self.actor_handles.append(actor_handle)

            self._set_dof_properties(env_handle, actor_handle)

            cube_handle = self._add_target(env_handle, i)
            self.cube_handles .append(cube_handle)
            self.gym.set_rigid_body_color(env_handle, cube_handle, 0,
                                          gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1.0, 0.0, 0.0))

    def _set_dof_properties(self, env, actor_handle):
        props = self.gym.get_actor_dof_properties(env, actor_handle)

        left_wheel_joint_index = self.dof_dict["left_wheel_joint"]
        props["driveMode"][left_wheel_joint_index] = gymapi.DOF_MODE_VEL
        props["stiffness"][left_wheel_joint_index] = 0.0
        props["damping"][left_wheel_joint_index] = 200

        right_wheel_joint_index = self.dof_dict["right_wheel_joint"]
        props["driveMode"][right_wheel_joint_index] = gymapi.DOF_MODE_VEL
        props["stiffness"][right_wheel_joint_index] = 0.0
        props["damping"][right_wheel_joint_index] = 200
        self.gym.set_actor_dof_properties(env, actor_handle, props)

        left_dof_handle = self.gym.find_actor_dof_handle(env, actor_handle, 'left_wheel_joint')
        self.gym.set_dof_target_velocity(env, left_dof_handle, 0)

        right_dof_handle = self.gym.find_actor_dof_handle(env, actor_handle, 'right_wheel_joint')
        self.gym.set_dof_target_velocity(env, right_dof_handle, 0)

    def pre_physics_step(self, actions: torch.Tensor):
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        # de-normalize
        actions_tensor[self.dof_dict["left_wheel_joint"]::self.num_dof] \
            = actions.to(self.device).view(-1)[0::self.num_actions]
        actions_tensor[self.dof_dict["right_wheel_joint"]::self.num_dof] \
            = actions.to(self.device).view(-1)[1::self.num_actions]
        velocity = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_velocity_target_tensor(self.sim, velocity)
    def reset_idx(self,env_ids):
        pass
    def draw_boundaries(self,viewer):
        color = gymapi.Vec3(1.0, 0, 0)
        for i, env in enumerate(self.env_handles):
            x0 = self.env_spacing
            y0 = self.env_spacing
            x1 = -self.env_spacing
            y1 = self.env_spacing
            x2 = -self.env_spacing
            y2 = -self.env_spacing
            x3 = self.env_spacing
            y3 = -self.env_spacing
            verts = [
                [gymapi.Vec3(x0, y0, 0.0), gymapi.Vec3(x1, y1, 0.0)],
                [gymapi.Vec3(x1, y1, 0.0), gymapi.Vec3(x2, y2, 0.0)],
                [gymapi.Vec3(x2, y2, 0.0), gymapi.Vec3(x3, y3, 0.0)],
                [gymapi.Vec3(x3, y3, 0.0), gymapi.Vec3(x0, y0, 0.0)]
            ]
            for p1, p2 in verts:
                gymutil.draw_line(p1, p2, color, self.gym,viewer, env)

    def run(self):
        cam_props = gymapi.CameraProperties()
        viewer = self.gym.create_viewer(self.sim, cam_props)

        self.draw_boundaries(viewer)

        while not self.gym.query_viewer_has_closed(viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(viewer, self.sim, True)

            # actions = torch.rand((self.num_envs, self.num_actions)) * 10
            # self.pre_physics_step(actions)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

@hydra.main(version_base="1.1", config_name="test_config.yaml", config_path="/home/shuoshuof/RL-Projects/isaacgym_template/cfg")
def launch_rlg_hydra(cfg: DictConfig):
    task_cfg = cfg.task
    task_cfg = omegaconf_to_dict(task_cfg)
    env = FindTarget(task_cfg)
    env.run()

if __name__ == "__main__":
    launch_rlg_hydra()
