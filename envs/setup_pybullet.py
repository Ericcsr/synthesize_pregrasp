import os
import numpy as np
import pytorch_kinematics as pk
import torch
import pybullet
import utils.rigidBodySento as rb
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Only load environment exclude hand
def create_bookshelf(p, alpha=0.8):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/plane.urdf"), useFixedBase=True)
    init_xyz = np.array([0.0, 0.0, 0.2])
    init_orn = torch.tensor([np.pi/2, 0, np.pi/2])
    init_orn_matrix = pk.euler_angles_to_matrix(init_orn, convention="XYZ")
    init_orn = pk.matrix_to_quaternion(init_orn_matrix).numpy()
    init_orn = np.array([init_orn[1], init_orn[2], init_orn[3], init_orn[0]])

    o_id = rb.create_primitive_shape(p, 1.0, pybullet.GEOM_BOX, (0.2, 0.2, 0.05),
                            color=(0.6, 0., 0., alpha), collidable=True,
                            init_xyz=init_xyz, init_quat=init_orn)
    w1_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.2, 0.2, 0.05),
                                      color=(0.0, 0.6, 0, 0.8), collidable=True,
                                      init_xyz=[0, 0.11, 0.2],
                                      init_quat=[0.7071068, 0, 0, 0.7071068])
    
    w2_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.2, 0.2, 0.05),
                                      color=(0.0, 0.6, 0, 0.8), collidable=True,
                                      init_xyz=[0, -0.11, 0.2],
                                      init_quat=[0.7071068, 0, 0, 0.7071068])
    w3_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.05, 0.2, 0.2),         # half-extend
                                      color=(0., 0.6, 0, 0.8), collidable=True,
                                      init_xyz=[-0.28, 0, 0.2],
                                      init_quat=[0.7071068, 0, 0, 0.7071068])
    return o_id

def create_laptop(p, alpha=0.8):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/plane.urdf"), useFixedBase=True)
    init_xyz = np.array([0.0, 0.0, 0.05])
    init_orn = np.array([0, 0, 0, 1])
    o_id = rb.create_primitive_shape(p, 1.0, pybullet.GEOM_BOX, (0.2, 0.2, 0.05),
                                     color=(0.6, 0, 0., alpha), collidable=True,
                                     init_xyz=init_xyz,
                                     init_quat=init_orn)
    return o_id

def create_tablebox(p, alpha=0.8):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/small_plane.urdf"), [0, 0, 0], useFixedBase=True)
    init_xyz = np.array([0.0, 0.0, 0.05])
    init_orn = np.array([0, 0, 0, 1])
    o_id = rb.create_primitive_shape(p, 1.0, pybullet.GEOM_BOX,(0.2, 0.2, 0.05),
                                     color=(0.6, 0, 0., alpha), collidable=True,
                                     init_xyz=init_xyz,
                                     init_quat=init_orn)
    return o_id
    
def create_wallbox(p, alpha=0.8):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/plane.urdf"), useFixedBase=True)
    init_xyz = np.array([0.0, 0.0, 0.05])
    init_orn = np.array([0, 0, 0, 1])
    o_id = rb.create_primitive_shape(p, 1.0, pybullet.GEOM_BOX, (0.2, 0.2, 0.05),
                                     color=(0.6, 0, 0., alpha), collidable=True,
                                     init_xyz=init_xyz,
                                     init_quat=init_orn)
    wall_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.1, 2.0, 0.5),
                                        color = (0.5, 0.5, 0.5, 0.8), collidable=True,
                                        init_xyz = np.array([-0.3, 0, 0.5]),
                                        init_quat = np.array([0, 0, 0, 1]))
    return o_id

pybullet_creator = {
    "laptop":create_laptop,
    "bookshelf":create_bookshelf,
    "tablebox":create_tablebox,
    "wallbox":create_wallbox
}