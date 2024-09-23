import isaacgym
from isaacgym import gymapi

# 初始化 Gym 和仿真环境
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 加载 URDF 资产
asset_root = "."
asset_file = "assets/robot.urdf"
asset_options = gymapi.AssetOptions()
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 创建环境
env = gym.create_env(sim, gymapi.Vec3(-1.0, 0.0, -1.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)

# 创建actor实例
actor = gym.create_actor(env, asset, gymapi.Transform(), "robot_actor", 0, 1)

# 获取所有刚体名称
rigid_body_names = gym.get_actor_rigid_body_names(env, actor)

# 指定要调整摩擦力的 link 名称
target_link_name = "base_link"  # 将此名称替换为你想要调整的 link 名称

if target_link_name in rigid_body_names:
    # 获取该 link 对应的刚体索引
    rigid_body_index = rigid_body_names.index(target_link_name)
    print(rigid_body_index)
    print(rigid_body_names)
    # 获取刚体的形状属性
    rigid_shape_props = gym.get_actor_rigid_shape_properties(env, actor)
    print(rigid_shape_props)
    # 调整摩擦力
    rigid_shape_props[rigid_body_index].friction = -1  # 设置新的摩擦力值

    # 应用修改后的刚体形状属性
    gym.set_actor_rigid_shape_properties(env, actor, rigid_shape_props)

    print(f"Adjusted friction for {target_link_name} to {rigid_shape_props[rigid_body_index].friction}")
else:
    print(f"Link {target_link_name} not found in the actor's rigid bodies.")

# 启动仿真
gym.simulate(sim)
