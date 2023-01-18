import os
import numpy as np
import pytorch_kinematics as pk
import torch
import pybullet
import utils.rigidBodySento as rb
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Only load environment exclude hand
def create_bookshelf(p, alpha=0.8,scale=[1,1,1]):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/plane.urdf"), useFixedBase=True)
    init_xyz = np.array([0.0, 0.0, 0.2*scale[1]])
    init_orn = torch.tensor([np.pi/2, 0, np.pi/2])
    init_orn_matrix = pk.euler_angles_to_matrix(init_orn, convention="XYZ")
    init_orn = pk.matrix_to_quaternion(init_orn_matrix).numpy()
    init_orn = np.array([init_orn[1], init_orn[2], init_orn[3], init_orn[0]])

    o_id = p.loadURDF(os.path.join(currentdir, "assets/book.urdf"), basePosition=init_xyz, baseOrientation=init_orn, useFixedBase=False)
    w1_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.2*scale[0], 0.2*scale[1], 0.05*scale[2]),
                                      color=(0.0, 0.6, 0, 0.8), collidable=True,
                                      init_xyz=np.array([0, 0.11*scale[2], 0.2*scale[1]]),
                                      init_quat=[0.7071068, 0, 0, 0.7071068])
    
    w2_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.2*scale[0], 0.2*scale[1], 0.05*scale[2]),
                                      color=(0.0, 0.6, 0, 0.8), collidable=True,
                                      init_xyz=np.array([0, -0.11*scale[2], 0.2*scale[1]]),
                                      init_quat=[0.7071068, 0, 0, 0.7071068]) # [x,y,z,w] 
    w3_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.05*scale[0], 0.2*scale[1], 0.2*scale[2]),         # half-extend
                                      color=(0., 0.6, 0, 0.8), collidable=True,
                                      init_xyz=np.array([-0.28*scale[0], 0, 0.2*scale[1]]),
                                      init_quat=[0.7071068, 0, 0, 0.7071068])
    return o_id, floor_id, w1_id, w2_id, w3_id

def create_foodbox(p, alpha=0.8, scale=[1,1,1]):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/plane.urdf"), useFixedBase=True)
    init_xyz = np.array([0.0, 0.0, 0.05*scale[2]])
    init_orn = np.array([0, 0, 0, 1])
    o_id = p.loadURDF(os.path.join(currentdir, "assets/foodbox.urdf"), 
                      basePosition=init_xyz,baseOrientation=init_orn, useFixedBase=False)
    return o_id,floor_id

    
def create_laptop(p, alpha=1.0, scale=[1,1,1]):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/plane.urdf"), useFixedBase=True)
    init_xyz = np.array([0.0, 0.0, 0.05*scale[2]])
    init_orn = np.array([0, 0, 0, 1])
    o_id = p.loadURDF(os.path.join(currentdir, "assets/laptop.urdf"), basePosition=init_xyz, baseOrientation=init_orn)

    wall_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.1, 2.0, 0.5),
                                        color = (0.5, 0.5, 0.5, 1.0), collidable=True,
                                        init_xyz = np.array([-0.11 - 0.2 * scale[0], 0, 0.5]),
                                        init_quat = np.array([0, 0, 0, 1]))
    return o_id, floor_id, wall_id

def create_keyboard(p,alpha=1.0,scale=[1,1,1]):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/small_plane.urdf"),basePosition=[-1,0,0])
    init_xyz = np.array([0.38, 0.0, 0.03])
    init_orn = np.array([0, 0, 0, 1])
    o_id = p.loadURDF(os.path.join(currentdir, "assets/keyboard.urdf"), basePosition=init_xyz, baseOrientation=init_orn)
    wall_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.1, 2.0, 0.5),
                                        color = (0.5, 0.5, 0.5, 1.0), collidable=True,
                                        init_xyz = np.array([-0.11 - 0.2 * scale[0], 0, 0.5]),
                                        init_quat = np.array([0, 0, 0, 1]))
    return o_id, floor_id, wall_id


def create_plate(p,alpha=0.8, scale=[1,1,1]):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/plane.urdf"), useFixedBase=True)
    init_xyz = np.array([0.0, 0.0, 0.0])
    init_orn = np.array([0, 0, 0, 1])
    o_id = p.loadURDF(os.path.join(currentdir, "assets/plate_cvx_simple.urdf"))
    p.resetBasePositionAndOrientation(o_id, init_xyz, init_orn)
    return o_id, floor_id

def create_handle(p, alpha=0.8):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/plane.urdf"), useFixedBase=True)
    init_xyz = np.array([0.145,0.0,0.0])
    shell_id = p.loadURDF(os.path.join(currentdir, "assets/handle_shell.urdf"),basePosition=init_xyz,useFixedBase=True)
    init_xyz = np.array([0.1,0.0,0.0])
    o_id = p.loadURDF(os.path.join(currentdir, "assets/handle.urdf"),basePosition=init_xyz,useFixedBase=True)
    return o_id, floor_id, shell_id

def create_waterbottle(p, alpha=0.8,scale=[1,1,1]):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/plane.urdf"), useFixedBase=True)
    o_id = p.loadURDF(os.path.join(currentdir, "assets/waterbottle.urdf"),basePosition=[0,0,0.115])
    w1_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.04, 0.025, 0.12),
                                      color=(0.6,0,0,1), collidable=True,
                                      init_xyz=[0.0,-0.075,0.12])
    w2_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.04, 0.025, 0.12),
                                      color=(0.6,0,0,1), collidable=True,
                                      init_xyz=[0.0, 0.075,0.12])
    w3_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.05, 5, 10),         # half-extend
                                      color=(0.6, 0.6, 0, 1), collidable=True,
                                      init_xyz= [-0.1, 0, 5],
                                      init_quat=[0.7071068, 0, 0, 0.7071068])
    return o_id, floor_id, w1_id, w2_id, w3_id

def create_waterbottle_compare(p,alpha=0.8, scale=[1,1,1]):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/plane.urdf"), useFixedBase=True)
    o_id = p.loadURDF(os.path.join(currentdir, "assets/waterbottle.urdf"),basePosition=[0,0,0.115])
    w1_id = p.loadURDF(os.path.join(currentdir, "assets/waterbottle.urdf"),basePosition=[0,-0.1,0.115])
    w2_id = p.loadURDF(os.path.join(currentdir, "assets/waterbottle.urdf"),basePosition=[0,0.1,0.115])
    w3_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.05, 5, 10),         # half-extend
                                      color=(0.6, 0.6, 0, 1), collidable=True,
                                      init_xyz=[-0.1, 0, -15], # [-0.1, 0, 5]
                                      init_quat=[0.7071068, 0, 0, 0.7071068])
    return o_id, floor_id, w1_id, w2_id, w3_id
    
def create_groovepen(p, alpha=0.8, scale=[1,1,1]):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/plane.urdf"), useFixedBase=True)
    o_id = p.loadURDF(os.path.join(currentdir, "assets/pen.urdf"), basePosition=[0.1,0,0.012])
    
    w1_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.2, 0.02, 0.015),
                                      color = (0.5, 0.5, 0.5, 1.0), collidable=True,
                                      init_xyz = np.array([0.2, -0.04, 0.015]),
                                      init_quat = np.array([0, 0, 0, 1]))

    w2_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.2, 0.02, 0.015),
                                        color = (0.5, 0.5, 0.5, 1.0), collidable=True,
                                        init_xyz = np.array([0.2, 0.04, 0.015]),
                                        init_quat = np.array([0, 0, 0, 1]))

    w3_id = rb.create_primitive_shape(p, 0, pybullet.GEOM_BOX, (0.02, 0.06, 0.03),
                                      color = (0.5, 0.5, 0.5, 1.0), collidable=True,
                                      init_xyz = np.array([-0.02, 0, 0.03]),
                                      init_quat = np.array([0, 0, 0, 1]))
    return o_id, floor_id, w1_id, w2_id, w3_id

def create_ruler(p, alpha=0.8, scale=[1,1,1]):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/small_plane.urdf"),useFixedBase=True,basePosition=[0, 1.48, 0])
    o_id = p.loadURDF(os.path.join(currentdir, "assets/ruler.urdf"),
                      basePosition=[0,0.16,0.001],baseOrientation=[0, 0, -0.7071068, 0.7071068])
    return o_id, floor_id

def create_cardboard(p, alpha=0.8, scale=[1,1,1]):
    floor_id = p.loadURDF(os.path.join(currentdir, "assets/small_plane.urdf"),useFixedBase=True, basePosition=[0, 1.48 ,0])
    o_id = p.loadURDF(os.path.join(currentdir, "assets/cardboard.urdf"), 
                      basePosition=[0,0.2 * scale[0],0.001], baseOrientation=[0, 0, -0.7071068, 0.7071068])
    return o_id, floor_id

pybullet_creator = {
    "foodbox":create_foodbox,
    "bookshelf":create_bookshelf,
    "laptop":create_laptop,
    "plate":create_plate,
    "waterbottle":create_waterbottle,
    "groovepen":create_groovepen,
    "ruler":create_ruler,
    "cardboard":create_cardboard,
    "keyboard":create_keyboard
}

if __name__ == "__main__":
    import pybullet as p
    import time
    p.connect(p.GUI)
    o_id = create_keyboard(p, scale=[1,1,1])
    while True:
        time.sleep(1/240.0)