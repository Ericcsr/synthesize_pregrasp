import numpy as np
from envs.small_block_contact_env import LaptopBulletEnv
from envs.bookshelf_env import BookShelfBulletEnv
import utils.rigidBodySento as rb

env = BookShelfBulletEnv()

# Center points
center_points = {
    "thumb": np.array([0.1, 0, 0.05]),
    "index": np.array([0.1, 0.2-0.4/6, -0.05]),
    "middle": np.array([0.1, 0, -0.05]),
    "ring": np.array([0.1, -0.2 + 0.4/6, -0.05])
}

# Draw a sphere on each region center
p_client = env.getBulletClient()
pos, quat = rb.get_link_com_xyz_orn(p_client, env.o_id, -1)
for finger in center_points.keys():
    v_id = p_client.createVisualShape(p_client.GEOM_SPHERE, radius=0.02, rgbaColor=[0.5, 0., 1.0, 1.0], specularColor=[1,1,1])
    world_pos, _ = p_client.multiplyTransforms(pos, quat, center_points[finger], [0, 0, 0, 1])
    b_id = p_client.createMultiBody(0.0, -1, v_id, world_pos, [0, 0, 0, 1])
while True:
    pass