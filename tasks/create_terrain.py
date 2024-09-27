
import math
from isaacgym.terrain_utils import *

class TerrainGenerator:
    def __init__(self,cfg,num_envs, env_spacing, use_point_cloud=True):
        self.cfg = cfg
        self.num_envs = num_envs

        self.terrain_types = self.cfg['availableTerrainTypes']

        self.terrain_functions = {
            'randomUniformTerrain': random_uniform_terrain,
            'discreteObstaclesTerrain': discrete_obstacles_terrain,
            'waveTerrain': wave_terrain,
            'minStepTerrain': min_step_terrain
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
            # 因此生成的点为分辨率加1，如果要保存高度图，需要去掉连接处的点
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



class OneTimeTerrainGenerator:
    def __init__(self, cfg, num_envs, env_spacing, gym, sim):
        self.cfg = cfg
        self.num_envs = num_envs

        self.terrain_types = self.cfg['availableTerrainTypes']

        self.terrain_functions = {
            'randomUniformTerrain': random_uniform_terrain,
            'discreteObstaclesTerrain': discrete_obstacles_terrain,
            'waveTerrain': wave_terrain,
            'minStepTerrain': min_step_terrain,
            'steppingStoneTerrain': stepping_stones_terrain,
        }

        self.horizontal_scale = cfg.get('horizontalScale', 1)
        self.vertical_scale = cfg.get('verticalScale', 0.01)

        self.map_length = 2 * env_spacing
        self.map_width = 2 * env_spacing


        self.env_width_resolutions = int(self.map_width / self.horizontal_scale)
        self.env_length_resolutions = int(self.map_length / self.horizontal_scale)

        self.border_size = cfg["borderSize"]
        self.border_length_resolutions, self.border_width_resolutions = \
            int(self.border_size / self.horizontal_scale), int(self.border_size / self.horizontal_scale)

        self.num_env_per_row = int(np.sqrt(self.num_envs))  # x-axi
        self.num_env_rows = math.ceil(self.num_envs / self.num_env_per_row)  # y-axi

        self.env_origins = np.zeros((self.num_env_rows, self.num_env_per_row, 3))

        # y-axi
        self.total_rows = int(self.num_env_rows * (self.env_length_resolutions)) \
                          + 2 * self.border_length_resolutions
        # x-axi
        self.total_cols = int(self.num_env_per_row * (self.env_width_resolutions)) \
                          + 2 * self.border_width_resolutions

        self.heightfield = np.zeros((self.total_cols, self.total_rows))

        self.randomized_terrain()

        vertices, triangles \
            = convert_heightfield_to_trimesh(self.heightfield,
                                             horizontal_scale=self.horizontal_scale,
                                             vertical_scale=self.vertical_scale,
                                             slope_threshold=0)

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -self.border_size - self.map_width/2
        tm_params.transform.p.y = -self.border_size - self.map_width/2
        tm_params.transform.p.z = -0.005
        if self.cfg['useRandomFriction']:
            tm_params.dynamic_friction = np.random.uniform(*self.cfg['dynamicFrictionRange'])
            tm_params.static_friction = np.random.uniform(*self.cfg['staticFrictionRange'])

        # 只在防止磨脚中使用
        z_negative_indices = np.argwhere(vertices[:,2]<0).reshape(-1)
        vertices[z_negative_indices,2] = -vertices[z_negative_indices,2]

        gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

    def randomized_terrain(self):
        for k in range(self.num_env_rows * self.num_env_per_row):

            i, j = np.unravel_index(k, (self.num_env_rows, self.num_env_per_row))

            env_origin_y = i * self.map_width - self.map_width / 2
            env_origin_x = j * self.map_length - self.map_length / 2

            terrain = SubTerrain("terrain",
                                 width=self.env_width_resolutions ,
                                 length=self.env_length_resolutions,
                                 vertical_scale=self.vertical_scale,
                                 horizontal_scale=self.horizontal_scale)

            terrain_type = np.random.choice(self.terrain_types)

            self.terrain_functions[terrain_type](terrain, **self.cfg[terrain_type])


            start_row =  i * self.env_length_resolutions + self.border_length_resolutions
            end_row = (i+1) * self.env_length_resolutions + self.border_length_resolutions
            start_col = j * self.env_width_resolutions + self.border_width_resolutions
            end_col = (j+1) * self.env_width_resolutions + self.border_width_resolutions
            self.heightfield[start_col:end_col, start_row:end_row] = terrain.height_field_raw

    def get_height_samples(self):
        return self.heightfield

class MoJiaoTerrainGenerator:
    def __init__(self, cfg, num_envs, env_spacing, gym, sim):
        self.cfg = cfg
        self.num_envs = num_envs

        self.terrain_types = ['minStepTerrain']

        self.terrain_functions = {
            'minStepTerrain': min_step_terrain,
        }

        self.horizontal_scale = cfg.get('horizontalScale', 1)
        self.vertical_scale = cfg.get('verticalScale', 0.01)

        self.map_length = 2 * env_spacing
        self.map_width = 2 * env_spacing


        self.env_width_resolutions = int(self.map_width / self.horizontal_scale)
        self.env_length_resolutions = int(self.map_length / self.horizontal_scale)

        self.border_size = cfg["borderSize"]
        self.border_length_resolutions, self.border_width_resolutions = \
            int(self.border_size / self.horizontal_scale), int(self.border_size / self.horizontal_scale)

        self.num_env_per_row = int(np.sqrt(self.num_envs))  # x-axi
        self.num_env_rows = math.ceil(self.num_envs / self.num_env_per_row)  # y-axi

        self.env_origins = np.zeros((self.num_env_rows, self.num_env_per_row, 3))

        # y-axi
        self.total_rows = int(self.num_env_rows * (self.env_length_resolutions)) \
                          + 2 * self.border_length_resolutions
        # x-axi
        self.total_cols = int(self.num_env_per_row * (self.env_width_resolutions)) \
                          + 2 * self.border_width_resolutions

        self.heightfield = np.zeros((self.total_cols, self.total_rows))

        self.randomized_terrain()

        vertices, triangles \
            = convert_heightfield_to_trimesh(self.heightfield,
                                             horizontal_scale=self.horizontal_scale,
                                             vertical_scale=self.vertical_scale,
                                             slope_threshold=0)

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -self.border_size - self.map_width/2
        tm_params.transform.p.y = -self.border_size - self.map_width/2
        tm_params.transform.p.z = -self.cfg['minStepTerrain']['height']
        if self.cfg['useRandomFriction']:
            tm_params.dynamic_friction = np.random.uniform(*self.cfg['dynamicFrictionRange'])
            tm_params.static_friction = np.random.uniform(*self.cfg['staticFrictionRange'])

        gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

    def randomized_terrain(self):
        for k in range(self.num_env_rows * self.num_env_per_row):

            i, j = np.unravel_index(k, (self.num_env_rows, self.num_env_per_row))

            env_origin_y = i * self.map_width - self.map_width / 2
            env_origin_x = j * self.map_length - self.map_length / 2

            terrain = SubTerrain("terrain",
                                 width=self.env_width_resolutions ,
                                 length=self.env_length_resolutions,
                                 vertical_scale=self.vertical_scale,
                                 horizontal_scale=self.horizontal_scale)

            terrain_type = np.random.choice(self.terrain_types)

            self.terrain_functions[terrain_type](terrain, **self.cfg[terrain_type])


            start_row =  i * self.env_length_resolutions + self.border_length_resolutions
            end_row = (i+1) * self.env_length_resolutions + self.border_length_resolutions
            start_col = j * self.env_width_resolutions + self.border_width_resolutions
            end_col = (j+1) * self.env_width_resolutions + self.border_width_resolutions
            self.heightfield[start_col:end_col, start_row:end_row] = terrain.height_field_raw

def min_step_terrain(terrain, height):

    height = int(height / terrain.vertical_scale)
    size = 2

    (rows, cols) = terrain.height_field_raw.shape

    # for x in range(1, cols - size, 4):
    #     terrain.height_field_raw[x:x + size, 1:-1] = height
    # for y in range(1, rows - size, 4):
    #     terrain.height_field_raw[1:-1, y:y + size] = height

    for x in range(0, cols , 4):
        terrain.height_field_raw[x:x + size, :] = height
    for y in range(0, rows , 4):
        terrain.height_field_raw[:, y:y + size] = height
    terrain.height_field_raw-=height
    return terrain