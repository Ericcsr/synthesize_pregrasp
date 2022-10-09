import enum
import numpy as np
'''
Parameters related to the allegro hand
'''
floor_urdf_path = "envs/assets/plant.urdf"
allegro_hand_urdf_path     = 'model/resources/allegro_hand_description/urdf/allegro_hand_description_right.urdf'
allegro_arm_urdf_path = 'model/resources/allegro_hand_description/urdf/allegro_arm.urdf'

shadow_hand_urdf_path = "model/resources/shadow_hand_description/urdf/shadow_hand.urdf"
shadow_hand_urdf_collision_path = "model/resources/shadow_hand_description/urdf/shadow_hand_collision.urdf"

# Offset for counting joints for fingers
allegro_hand_offset = 0
allegro_arm_offset  = 4
shadow_hand_offset = 2
# allegro_base_inertia_offset = -0.0475
# allegro_base_collision_offset = 0.0475
gravity_vector = [0., 0., -9.8]
object_padding = 0.01  # Radius of the finger

force_closure_regularization_weight = 0.05
min_finger_normal = 0.1

SCALE = 0.6
POINT_NUM=20000
MAX_FORCE=80
CONTROL_SKIP=50
ALLOWANCE=0.05

HAS_FLOOR=False

min_normal_force = 1500. # 1500.
fingertip_radius = 0.012

tabletop_dim = (100.,100.,0.1)
tabletop_mu_static = 0.5
tabletop_mu_dynamic = 0.4

# Used for training and running GraspCVAE
initial_num_points_in_input_point_cloud = 2048
final_num_points_in_input_point_cloud = 1024
point_cloud_tabletop_occlusion_height=0.002

USE_SOFT_BOUNDING=False

# For checking collisions
# joint idx: local transform of collision checking point (one per link)
# TODO: Hard coded collision dict currently UNUSED
collision_points_dict = {-1:(2.04e-2, 0., -0.095),
                         0: (9.8e-3,0.,8.2e-3),
                         1: (9.8e-3,0.,2.7e-2),
                         2: (9.8e-3,0.,1.92e-2),
                         3: (9.8e-3,0.,1.335e-2),
                         5: (9.8e-3,0.,8.2e-3),
                         6: (9.8e-3,0.,1.92e-2),
                         7: (9.8e-3,0.,1.335e-2),
                         8: (9.8e-3,0.,8.2e-3),
                         15: (9.8e-3,0.,1.79e-2),
                         16: (9.8e-3,0.,8.85e-3),
                         17: (9.8e-3,0.,2.57e-2),
                         18:(9.8e-3,0.,2.115e-2)}

friction_coefficient = 0.3

drake_ycb_path = "drake/manipulation/models/ycb/sdf"
drake_ycb_objects = {
    "003_cracker_box": "base_link_cracker",
    "004_sugar_box": "base_link_sugar",
    "005_tomato_soup_can": "base_link_soup",
    "006_mustard_bottle": "base_link_mustard",
    "009_gelatin_box": "base_link_gelatin",
    "010_potted_meat_can": "base_link_meat",
    "naive_box":"manipulated_object"
}

drake_mesh_sdf_pose = {
    "003_cracker_box": (-0.014, 0.103, 0.013, 1.57, -1.57, 0),
    "004_sugar_box": (-0.018, 0.088, 0.0039, -0.77, -1.52, 2.36),
    "005_tomato_soup_can": (-0.0018,  0.051, -0.084, 1.57, 0.13, 0.0),
    "006_mustard_bottle": (0.0049, 0.092, 0.027, 1.57, -0.40, 0.0),
    "009_gelatin_box": (-0.0029, 0.024, -0.015, -0.0085, -0.002, 1.34),
    "010_potted_meat_can": (0.034, 0.039, 0.025, 1.57, 0.052, 0.0)
}


class AllegroHandFinger(enum.Enum):
    '''
    Stores the fingertip joint indices
    '''
    # THUMB = 19 # For pybullet
    # INDEX = 4
    # MIDDLE = 9
    # RING = 14
    THUMB = 15
    INDEX = 3
    MIDDLE = 7
    RING = 11

class AllegroArmFinger(enum.Enum):
    THUMB = 20
    INDEX = 8
    MIDDLE = 12
    RING = 16

class ShadowHandFinger(enum.Enum):
    THUMB = 6,
    INDEX = 10,
    MIDDLE = 14,
    RING = 18,
    LITTLE = 23

NameToFingerIndex = {
    "thumb":AllegroHandFinger.THUMB,
    "ifinger":AllegroHandFinger.INDEX,
    "rfinger":AllegroHandFinger.RING,
    "mfinger":AllegroHandFinger.MIDDLE}

IndexToFingerName = {
    AllegroHandFinger.THUMB:"thumb",
    AllegroHandFinger.INDEX:"ifinger",
    AllegroHandFinger.MIDDLE:"mfinger",
    AllegroHandFinger.RING:"rfinger"}

NameToArmFingerIndex = {
    "thumb":AllegroArmFinger.THUMB,
    "ifinger":AllegroArmFinger.INDEX,
    "rfinger":AllegroArmFinger.RING,
    "mfinger":AllegroArmFinger.MIDDLE}

NameToShadowFingerIndex = {
    "thumb":ShadowHandFinger.THUMB,
    "ifinger":ShadowHandFinger.INDEX,
    "mfinger":ShadowHandFinger.MIDDLE,
    "rfinger":ShadowHandFinger.RING,
    "lfinger":ShadowHandFinger.LITTLE
}


ActiveAllegroHandFingers =[AllegroHandFinger.THUMB,
                           AllegroHandFinger.INDEX,
                           AllegroHandFinger.MIDDLE,
                           AllegroHandFinger.RING]

ActiveShadowHandFingers = [ShadowHandFinger.THUMB,
                           ShadowHandFinger.INDEX,
                           ShadowHandFinger.MIDDLE,
                           ShadowHandFinger.RING,
                           ShadowHandFinger.LITTLE]

AllAllegroHandFingers =[AllegroHandFinger.THUMB,
                        AllegroHandFinger.INDEX,
                        AllegroHandFinger.MIDDLE,
                        AllegroHandFinger.RING]

AllAllegroArmFingers = [AllegroArmFinger.THUMB,
                        AllegroArmFinger.INDEX,
                        AllegroArmFinger.MIDDLE,
                        AllegroArmFinger.RING]

AllShadowHandFingers = {
    ShadowHandFinger.THUMB,
    ShadowHandFinger.INDEX,
    ShadowHandFinger.MIDDLE,
    ShadowHandFinger.RING,
    ShadowHandFinger.LITTLE
}

AllegroHandFingertipDrakeLink={
    AllegroHandFinger.THUMB: 15,
    AllegroHandFinger.INDEX: 3,
    AllegroHandFinger.MIDDLE: 7,
    AllegroHandFinger.RING: 11
}

AllegroArmFingertipDrakeLink={
    AllegroArmFinger.THUMB:15,
    AllegroArmFinger.INDEX:3,
    AllegroArmFinger.MIDDLE:7,
    AllegroArmFinger.RING:11 
}

ShadowHandFingertipDrakeLink={
    ShadowHandFinger.THUMB:6,
    ShadowHandFinger.INDEX:10,
    ShadowHandFinger.MIDDLE:14,
    ShadowHandFinger.RING:18,
    ShadowHandFinger.LITTLE:23
}

AllegroHandFingertipDrakeLinkOffset={
    AllegroHandFinger.THUMB: np.array([0.,0.,0.0423]),
    AllegroHandFinger.INDEX: np.array([0.,0.,0.0267]),
    AllegroHandFinger.MIDDLE: np.array([0.,0.,0.0267]),
    AllegroHandFinger.RING: np.array([0.,0.,0.0267])
}

AllegroArmFingertipDrakeLinkOffset={
    AllegroArmFinger.THUMB: np.array([0.,0.,0.0423]),
    AllegroArmFinger.INDEX: np.array([0.,0.,0.0267]),
    AllegroArmFinger.MIDDLE: np.array([0.,0.,0.0267]),
    AllegroArmFinger.RING: np.array([0.,0.,0.0267])
}

ShadowHandFingertipDrakeLinkOffset = {
    ShadowHandFinger.THUMB: np.array([0., 0., 0.02]),
    ShadowHandFinger.INDEX: np.array([0., 0., 0.016]),
    ShadowHandFinger.MIDDLE: np.array([0., 0., 0.016]),
    ShadowHandFinger.RING: np.array([0., 0., 0.016]),
    ShadowHandFinger.LITTLE: np.array([0., 0., 0.016]),
}

ShadowHandIndex2Link = {
    ShadowHandFinger.THUMB: "thumb_distal",
    ShadowHandFinger.INDEX: "index_finger_distal",
    ShadowHandFinger.MIDDLE: "middle_finger_distal",
    ShadowHandFinger.RING: "ring_finger_distal",
    ShadowHandFinger.LITTLE: "little_finger_distal"
}

ShadowHandIndex2Name = {
    ShadowHandFinger.THUMB: "thumb",
    ShadowHandFinger.INDEX: "index_finger",
    ShadowHandFinger.MIDDLE: "middle_finger",
    ShadowHandFinger.RING: "ring_finger",
    ShadowHandFinger.LITTLE: "little_finger"
}

# AllegroHandFingertipDrakeLinkOffset={
#     AllegroHandFinger.THUMB: np.array([0.,0.,0.0]),
#     AllegroHandFinger.INDEX: np.array([0.,0.,0.0]),
#     AllegroHandFinger.MIDDLE: np.array([0.,0.,0.0]),
#     AllegroHandFinger.RING: np.array([0.,0.,0.0])
# }

def finger_angles_dict_to_finger_q(finger_angles_dict):
    """
    TODO: add unit test
    :param finger_angles_dict:
    :return:
    """
    ans = np.zeros(16)
    for finger in AllAllegroHandFingers:
        idx_end = AllegroHandFingertipDrakeLink[finger] + 1
        idx_start = idx_end - 4
        ans[idx_start:idx_end] = finger_angles_dict[finger]
    return ans

def finger_q_to_finger_angles_dict(finger_q):
    """
    TODO: add unit test
    :param finger_angles_dict:
    :return:
    """
    finger_angles_dict = {}
    for finger in AllAllegroHandFingers:
        idx_end = AllegroHandFingertipDrakeLink[finger] + 1
        idx_start = idx_end - 4
        finger_angles_dict[finger] = finger_q[idx_start:idx_end]
    return finger_angles_dict
