from isaacgym import gymutil, gymtorch, gymapi
import random
import torch
import math
from isaacgymenvs.tasks.base.vec_task import VecTask
import numpy as np
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse
from typing import Tuple, Dict
from isaacgym.terrain_utils import *

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
        self.draw_boundaries(self.cfg["env"]['envSpacing'])

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
        generator = TerrainGenerator(self.cfg["env"]["terrain"],self.num_envs,env_spacing=self.cfg["env"]['envSpacing'])
        generator.randomized_terrain(self.gym,self.sim)
        generator.add_boundary(self.gym,self.sim)
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

    def draw_boundaries(self,spacing):
        color = gymapi.Vec3(1.0, 0, 0)
        for i, env in enumerate(self.env_handles):
            x0 = spacing
            y0 = spacing
            x1 = -spacing
            y1 = spacing
            x2 = -spacing
            y2 = -spacing
            x3 = spacing
            y3 = -spacing
            verts = [
                [gymapi.Vec3(x0, y0, 0.0), gymapi.Vec3(x1, y1, 0.0)],
                [gymapi.Vec3(x1, y1, 0.0), gymapi.Vec3(x2, y2, 0.0)],
                [gymapi.Vec3(x2, y2, 0.0), gymapi.Vec3(x3, y3, 0.0)],
                [gymapi.Vec3(x3, y3, 0.0), gymapi.Vec3(x0, y0, 0.0)]
            ]
            assert self.viewer is not None, "did not set a viewer"
            for p1, p2 in verts:
                gymutil.draw_line(p1, p2, color, self.gym,self.viewer, env)

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

class TerrainGenerator:
    def __init__(self,cfg,num_envs, env_spacing, use_point_cloud=True):
        self.cfg = cfg
        self.num_envs = num_envs

        self.terrain_types = self.cfg['availableTerrainTypes']

        self.terrain_functions = {
            'randomUniformTerrain': random_uniform_terrain,
            'discreteObstaclesTerrain': discrete_obstacles_terrain,
            'waveTerrain': wave_terrain,
            'minStepTerrain': self.min_step_terrain
        }

        self.horizontal_scale = cfg.get('horizontalScale',1)
        self.vertical_scale = cfg.get('verticalScale',0.01)

        self.map_length = 2*env_spacing
        self.map_width = 2*env_spacing

        # # 非连续生成地形，应该生成多一行和列的点，不然地形不连续
        # self.width_per_env_pixels = int(self.map_width / self.horizontal_scale) + 1
        # self.length_per_env_pixels = int(self.map_length / self.horizontal_scale) + 1

        self.env_width_resolutions = int(self.map_width / self.horizontal_scale)
        self.env_length_resolutions = int(self.map_length / self.horizontal_scale)

        self.border_size = cfg["borderSize"]
        self.border_length_resolutions,self.border_width_resolutions = \
            int(self.border_size/self.horizontal_scale),int(self.border_size/self.horizontal_scale)

        self.num_env_per_row = int(np.sqrt(self.num_envs)) # x-axi
        self.num_env_rows = math.ceil(self.num_envs / self.num_env_per_row) # y-axi

        self.env_origins = np.zeros((self.num_env_rows, self.num_env_per_row, 3))

        # y-axi
        self.total_rows = int(self.num_env_rows * (self.env_length_resolutions)) \
                          + 2*self.border_length_resolutions
        # x-axi
        self.total_cols = int(self.num_env_per_row * (self.env_width_resolutions)) \
                          + 2*self.border_width_resolutions

        if use_point_cloud:
            self.height_samples = np.zeros((self.total_cols,self.total_rows))


    def randomized_terrain(self,gym,sim):
        for k in range(self.num_env_rows*self.num_env_per_row):

            i,j = np.unravel_index(k, (self.num_env_rows, self.num_env_per_row))

            env_origin_y = i * self.map_width - self.map_width / 2
            env_origin_x = j * self.map_length - self.map_length/2

            # 外围加多一圈为0的平滑连接处，需要-1
            terrain = SubTerrain("terrain",
                                 width=self.env_width_resolutions-1,
                                 length=self.env_length_resolutions-1,
                                 vertical_scale=self.vertical_scale,
                                 horizontal_scale=self.horizontal_scale)

            terrain_type = np.random.choice(self.terrain_types)

            self.terrain_functions[terrain_type](terrain,**self.cfg[terrain_type])

            # 非连续生成地形，应该生成多一行和列的点，不然地形不连续
            # 因此生成的点为分辨率加1，如果要保存高度图，需要去掉连接出的点
            heightfield = np.zeros((self.env_width_resolutions + 1, self.env_length_resolutions + 1))
            if terrain_type == 'minStepTerrain':
                heightfield = np.full((self.env_width_resolutions + 1, self.env_length_resolutions + 1),int(self.cfg[terrain_type]['height'] / self.vertical_scale))
            #     heightfield[1:-1, 1:-1] = terrain.height_field_raw
            # else:
            #     heightfield[2:-2, 2:-2] = terrain.height_field_raw[1:-1, 1:-1]
            heightfield[2:-2, 2:-2] = terrain.height_field_raw[1:-1, 1:-1]

            vertices, triangles\
                = convert_heightfield_to_trimesh(heightfield,
                                                 horizontal_scale=self.horizontal_scale,
                                                 vertical_scale=self.vertical_scale,
                                                 slope_threshold=0)

            tm_params = gymapi.TriangleMeshParams()
            tm_params.nb_vertices = vertices.shape[0]
            tm_params.nb_triangles = triangles.shape[0]
            tm_params.transform.p.x = env_origin_x
            tm_params.transform.p.y = env_origin_y
            tm_params.transform.p.z = -0.005
            if self.cfg['useRandomFriction']:
                tm_params.dynamic_friction = np.random.uniform(*self.cfg['dynamicFrictionRange'])
                tm_params.static_friction = np.random.uniform(*self.cfg['staticFrictionRange'])
            gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)




    def add_boundary(self,gym,sim):

        for i in range(-1, self.num_env_rows + 1):
            for j in range(-1, self.num_env_per_row + 1):
                if i == -1 or i == self.num_env_rows or j == -1 or j == self.num_env_per_row:

                    env_origin_y = i * self.map_width - self.map_width / 2
                    env_origin_x = j * self.map_length - self.map_width / 2

                    if j ==-1:
                        env_origin_x += -self.border_size + self.map_width
                    if i==-1:
                        env_origin_y += -self.border_size + self.map_length
                    if i == -1 or i == self.num_env_rows:
                        border_length_resolution = self.border_length_resolutions + 1
                    else:
                        border_length_resolution = self.env_length_resolutions + 1
                    if j == -1 or j == self.num_env_per_row:
                        border_width_resolution = self.border_width_resolutions + 1
                    else:
                        border_width_resolution = self.env_width_resolutions + 1

                    heightfield = np.zeros((border_width_resolution,border_length_resolution))
                    vertices, triangles \
                        = convert_heightfield_to_trimesh(heightfield,
                                                         horizontal_scale=self.horizontal_scale,
                                                         vertical_scale=self.vertical_scale,
                                                         slope_threshold=1.5)

                    tm_params = gymapi.TriangleMeshParams()
                    tm_params.nb_vertices = vertices.shape[0]
                    tm_params.nb_triangles = triangles.shape[0]
                    tm_params.transform.p.x = env_origin_x
                    tm_params.transform.p.y = env_origin_y
                    tm_params.transform.p.z = -0.005
                    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

    @staticmethod
    def min_step_terrain(terrain, height):

        height = int(height / terrain.vertical_scale)
        size = 2

        (rows, cols) = terrain.height_field_raw.shape

        for x in range(1, cols - size, 4):
            terrain.height_field_raw[x:x + size, 1:-1] = height
        for y in range(1, rows - size, 4):
            terrain.height_field_raw[1:-1,y:y + size] = height

        return terrain

import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

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
