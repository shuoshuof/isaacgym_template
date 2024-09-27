from isaacgym import gymutil, gymtorch, gymapi

def visualize_env_boundaries(gym,viewer,env_handles ,spacing):
    color = gymapi.Vec3(1.0, 0, 0)
    for i, env in enumerate(env_handles):
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
        assert viewer is not None, "did not set a viewer"
        for p1, p2 in verts:
            gymutil.draw_line(p1, p2, color, gym, viewer, env)