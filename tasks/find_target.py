from isaacgym import gymutil, gymtorch, gymapi
import random
import torch


class Env:
    def __init__(self):
        self.sim = None
        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.num_envs = 64
        self.device = 'cpu'
        self.num_actions = 2
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

    def _create_envs(self):
        asset_root = "../assets"
        asset_file = "robot.urdf"
        asset = self.gym.load_asset(self.sim, asset_root, asset_file)

        self.dof_dict = {value: index
                        for index, value in enumerate(self.gym.get_asset_dof_names(asset))}
        self.num_dof = self.gym.get_asset_dof_count(asset)

        self.num_envs = 64
        envs_per_row = 8
        env_spacing = 2.0
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        # cache some common handles for later use
        envs = []
        actor_handles = []

        # create and populate the environments
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            envs.append(env)

            height = random.uniform(1.0, 2.5)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, height)

            actor_handle = self.gym.create_actor(env, asset, pose, "MyActor", i, 1)
            actor_handles.append(actor_handle)

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
            self.gym.set_dof_target_velocity(env, left_dof_handle, 5.0)

            right_dof_handle = self.gym.find_actor_dof_handle(env, actor_handle, 'right_wheel_joint')
            self.gym.set_dof_target_velocity(env, right_dof_handle, 10.0)

    def pre_physics_step(self, actions: torch.Tensor):
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)

        # de-normalize
        actions_tensor[self.dof_dict["left_wheel_joint"]::self.num_dof] \
            = actions.to(self.device).view(-1)[0::self.num_actions]
        actions_tensor[self.dof_dict["right_wheel_joint"]::self.num_dof] \
            = actions.to(self.device).view(-1)[1::self.num_actions]
        velocity = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_velocity_target_tensor(self.sim, velocity)

    def run(self):
        cam_props = gymapi.CameraProperties()
        viewer = self.gym.create_viewer(self.sim, cam_props)

        while not self.gym.query_viewer_has_closed(viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(viewer, self.sim, True)


            actions = torch.rand((self.num_envs,self.num_actions))*10
            self.pre_physics_step(actions)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)


if __name__ == "__main__":
    env = Env()
    env.run()
