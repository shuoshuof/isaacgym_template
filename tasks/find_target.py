from isaacgym import gymapi
import random


class Env:
    def __init__(self):
        self.sim = None
        self.gym = gymapi.acquire_gym()
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
    def _create_envs(self):
        asset_root = "../assets"
        asset_file = "robot.urdf"
        asset = self.gym.load_asset(self.sim, asset_root, asset_file)

        num_envs = 64
        envs_per_row = 8
        env_spacing = 2.0
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        # cache some common handles for later use
        envs = []
        actor_handles = []

        # create and populate the environments
        for i in range(num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            envs.append(env)

            height = random.uniform(1.0, 2.5)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, height, 0.0)

            actor_handle = self.gym.create_actor(env, asset, pose, "MyActor", i, 1)
            actor_handles.append(actor_handle)

            props = self.gym.get_actor_dof_properties(env, actor_handle)

            left_wheel_joint_index = self.get_joint_index(env,actor_handle,'left_wheel_joint')

            props["driveMode"][left_wheel_joint_index] = gymapi.DOF_MODE_VEL
            props["damping"][left_wheel_joint_index] = 0

            right_wheel_joint_index = self.get_joint_index(env,actor_handle,'right_wheel_joint')
            props["driveMode"][right_wheel_joint_index] = gymapi.DOF_MODE_VEL
            props["damping"][right_wheel_joint_index] = 0

            left_dof_handle = self.gym.find_actor_dof_handle(env,actor_handle,'left_wheel_joint')
            self.gym.set_dof_target_velocity(env, left_dof_handle, 10)

            right_dof_handle = self.gym.find_actor_dof_handle(env,actor_handle,'right_wheel_joint')
            self.gym.set_dof_target_velocity(env, right_dof_handle, 10)

    def get_joint_index(self,env_handle,actor_handle,joint_name:str):
        dof_names = self.gym.get_actor_dof_names(env_handle, actor_handle)
        joint_index = dof_names.index(joint_name)
        return joint_index

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

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)
# gym = gymapi.acquire_gym()
# # get default set of parameters
# sim_params = gymapi.SimParams()
#
# # set common parameters
# sim_params.dt = 1 / 60
# sim_params.substeps = 2
# sim_params.up_axis = gymapi.UP_AXIS_Z
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
#
# # set PhysX-specific parameters
# sim_params.physx.use_gpu = True
# sim_params.physx.solver_type = 1
# sim_params.physx.num_position_iterations = 6
# sim_params.physx.num_velocity_iterations = 1
# sim_params.physx.contact_offset = 0.01
# sim_params.physx.rest_offset = 0.0
#
# # set Flex-specific parameters
# sim_params.flex.solver_type = 5
# sim_params.flex.num_outer_iterations = 4
# sim_params.flex.num_inner_iterations = 20
# sim_params.flex.relaxation = 0.8
# sim_params.flex.warm_start = 0.5
#
#
# # create sim with these parameters
# sim = gym.create_sim(0,0, gymapi.SIM_PHYSX, sim_params)
#
#
#
# plane_params = gymapi.PlaneParams()
# plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
# plane_params.distance = 0
# plane_params.static_friction = 1
# plane_params.dynamic_friction = 1
# plane_params.restitution = 0
#
# # create the ground plane
# gym.add_ground(sim, plane_params)
#
# asset_root = "../assets"
# asset_file = "robot.urdf"
# asset = gym.load_asset(sim, asset_root, asset_file)
#
# # spacing = 2.0
# # lower = gymapi.Vec3(-spacing, 0.0, -spacing)
# # upper = gymapi.Vec3(spacing, spacing, spacing)
# #
# # env = gym.create_env(sim, lower, upper, 8)
# #
# # pose = gymapi.Transform()
# # pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
# # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
# #
# # actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)
#
# # set up the env grid
# num_envs = 64
# envs_per_row = 8
# env_spacing = 2.0
# env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
# env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
#
# # cache some common handles for later use
# envs = []
# actor_handles = []
#
# # create and populate the environments
# for i in range(num_envs):
#     env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
#     envs.append(env)
#
#     height = random.uniform(1.0, 2.5)
#
#     pose = gymapi.Transform()
#     pose.p = gymapi.Vec3(0.0, height, 0.0)
#
#     actor_handle = gym.create_actor(env, asset, pose, "MyActor", i, 1)
#     actor_handles.append(actor_handle)
#
# cam_props = gymapi.CameraProperties()
# viewer = gym.create_viewer(sim, cam_props)
#
#
# while not gym.query_viewer_has_closed(viewer):
#
#     # step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)
#
#     # update the viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)
#
#     # Wait for dt to elapse in real time.
#     # This synchronizes the physics simulation with the rendering rate.
#     gym.sync_frame_time(sim)

if __name__=="__main__":
    env = Env()
    env.run()
