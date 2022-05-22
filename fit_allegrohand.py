import pybullet as p
from allegro_hand import AllegroHand
import numpy as np
from model.param import SCALE
import torch
import time
import rigidBodySento as rb

TS = 1./250

# Parse the motion data from contact & force optimization
# Should output 5 distinct dictionary of finger tip's target
# position, as well as weight
def parse_finger_motion_data(data):
    tip_poses = []
    tip_weights = []
    for i in range(data.shape[0]):
        tip_pose = {"thumb":data[i,0] * SCALE,
                    "rfinger":data[i,1] * SCALE,
                    "mfinger":data[i,2] * SCALE,
                    "ifinger":data[i,3] * SCALE}
        tip_weight = {"thumb":1 if data[i,0].sum() < 100 else 0,
                      "rfinger":1 if data[i,1].sum() < 100 else 0,
                      "mfinger":1 if data[i,2].sum() < 100 else 0,
                      "ifinger":1 if data[i,3].sum() < 100 else 0}
        tip_poses.append(tip_pose)
        tip_weights.append(tip_weight)
    return tip_poses, tip_weights

def parse_object_motion_data(data):
    object_pos = []
    object_orn = []
    for i in range(data.shape[0]):
        object_pos.append((data[i,:3]*SCALE).tolist())
        object_orn.append(data[i,3:].tolist())
    return object_pos, object_orn

def animate_object(o_id, object_poses, object_orns):
    for i in range(len(object_poses)):
        p.resetBasePositionAndOrientation(o_id, object_poses[i],object_orns[i])
        time.sleep(20 * TS)

def setStates(r_id, states):
    for i in range(len(states)):
        p.resetJointState(r_id, i, states[i])

def animate_keyframe(o_id, hand_id, tips_id,object_poses, object_orns, joint_states, base_states, tips_poses):
    for i in range(len(joint_states)):
        p.resetBasePositionAndOrientation(o_id, object_poses[i*75],object_orns[i*75])
        setStates(hand_id, joint_states[i])
        p.resetBasePositionAndOrientation(hand_id, base_states[i][0], base_states[i][1])
        p.resetBasePositionAndOrientation(tips_id[0], tips_poses[i]["thumb"], base_states[i][1])
        p.resetBasePositionAndOrientation(tips_id[1], tips_poses[i]["ifinger"], base_states[i][1])
        p.resetBasePositionAndOrientation(tips_id[2], tips_poses[i]["mfinger"], base_states[i][1])
        p.resetBasePositionAndOrientation(tips_id[3], tips_poses[i]["rfinger"], base_states[i][1])
        input("Press Enter to continue")

def fit_finger_tips(targets, weights, hand):
    joint_states = []
    base_states = []
    for i in range(len(targets)):
        hand.reset()
        joint_state, base_state = hand.regressFingertipPos(targets[i], weights=weights[i])
        joint_state = AllegroHand.getBulletJointState(joint_state)
        base_state = AllegroHand.getBulletBaseState(base_state)
        joint_states.append(joint_state)
        base_states.append(base_state)
        print("Finished Fit:",i+1)
    return joint_states, base_states

def createTipsVisual(radius = [0.04,0.02,0.02,0.02]):
    tips_id = []
    colors = [(0.5, 0.5, 1.0, 1.0),
              (0.5, 0.0, 1.0, 1.0),
              (0.5, 0.0, 1.0, 1.0),
              (0.5, 0.0, 1.0, 1.0)]
    for i in range(4):
        id = rb.create_primitive_shape(p,1.0, p.GEOM_SPHERE,
                                       (radius[i],),
                                       color=colors[i],
                                       collidable=False,
                                       init_xyz=(100,100,100))
        tips_id.append(id)
    return tips_id

# steps should be a list, record gap between each joint key states
def interpolate_joints(joint_states, steps):
    pass

# steps should be a list, record gap between each base key states
def interpolate_base(base_states, steps):
    pass

if __name__ == "__main__":
    finger_tip_data = np.load("tip_poses.npy")
    object_data = np.load("object_poses.npy")

    tip_poses, tip_weights = parse_finger_motion_data(finger_tip_data)
    obj_poses, obj_orns = parse_object_motion_data(object_data)
    p.connect(p.GUI)
    obj_id = rb.create_primitive_shape(p, 1.0, p.GEOM_BOX, (0.2 * SCALE, 0.2 * SCALE, 0.05 * SCALE),         # half-extend
                                    color=(0.6, 0, 0, 0.8), collidable=True,
                                    init_xyz=(0, 0, 0.05 * SCALE),
                                    init_quat=(0, 0, 0, 1))
    tips_id = createTipsVisual()
    hand = AllegroHand(iters = 500,tuneBase=True)
    joint_states, base_states = fit_finger_tips(tip_poses, tip_weights, hand)
    hand_id = p.loadURDF("allegro_hand_description/urdf/allegro_hand_description_right.urdf",useFixedBase=1)
    animate_keyframe(obj_id,hand_id, tips_id, obj_poses, obj_orns, joint_states, base_states, tip_poses)