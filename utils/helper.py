import numpy as np
import pybullet as p
import torch
import pytorch_kinematics as pk
import time
import model.param as model_params
import imageio

TS=1/250.

finger_idxs = {
    "ifinger":[5,6,7,8],
    "mfinger":[10,11,12,13],
    "rfinger":[15,16,17,18],
    "thumb":[20,21,22,23]}

def parse_finger_motion_data(data):
    tip_poses = []
    tip_weights = []
    for i in range(data.shape[0]):
        tip_pose = {"thumb":data[i,0] * model_params.SCALE,
                    "ifinger":data[i,3] * model_params.SCALE,
                    "mfinger":data[i,2] * model_params.SCALE,
                    "rfinger":data[i,1] * model_params.SCALE}
        tip_weight = {"thumb":1 if data[i,0].sum() < 100 else 0,
                      "ifinger":1 if data[i,3].sum() < 100 else 0,
                      "mfinger":1 if data[i,2].sum() < 100 else 0,
                      "rfinger":1 if data[i,1].sum() < 100 else 0}
        tip_poses.append(tip_pose)
        tip_weights.append(tip_weight)
    return tip_poses, tip_weights

def parse_object_motion_data(data):
    object_pos = []
    object_orn = []
    for i in range(data.shape[0]):
        object_pos.append(data[i,:3]*model_params.SCALE)
        object_orn.append(data[i,3:])
    return object_pos, object_orn

def animate_object(o_id, object_poses, object_orns, renderer=None):
    for i in range(len(object_poses)):
        p.resetBasePositionAndOrientation(o_id, 
                                          object_poses[i], 
                                          object_orns[i])
        time.sleep(20 * TS)
        if not renderer == None:
            renderer.render()

def animate_object_states(o_id, object_states, renderer=None):
    for i in range(len(object_states)):
        p.resetBasePositionAndOrientation(o_id,
                                          object_states[i][:3],
                                          object_states[i][3:])
        time.sleep(20 * TS)
        if not renderer == None:
            renderer.render()

def setStates(hand_id, states):
    for i in range(len(states)):
        p.resetJointState(hand_id, i, states[i])

def animate_keyframe(o_id, 
                     hand_id,
                     object_poses_keyframe,  # Already extracted as key frame
                     object_orns_keyframe, 
                     joint_states, 
                     base_states=None,
                     hand_drake=None,
                     desired_positions=None,
                     hand_weights=None,
                     renderer=None,
                     point_cloud=None):
    for i in range(len(joint_states)):
        p.resetBasePositionAndOrientation(o_id, 
                                          object_poses_keyframe[i],
                                          object_orns_keyframe[i])
        setStates(hand_id, joint_states[i])
        if not base_states == None:
            p.resetBasePositionAndOrientation(hand_id,
                                              base_states[i][0],
                                              base_states[i][1])
        # Create Visual features # need to change this
        if not (point_cloud is None):
            print("Reach here")
            hand_drake.draw_point_cloud(point_cloud[i])
        if not desired_positions == None:
            hand_drake.draw_desired_positions(desired_positions[i],hand_weights[i])
        
        if not renderer == None:
            print("Press q to continue")
            renderer.render(blocking=True)
        else:
            input("Press Enter to continue")

def animate_keyframe_states(o_id,
                            hand_id,
                            object_states_keyframe,
                            joint_states,
                            base_states=None,
                            hand_drake=None,
                            desired_positions=None,
                            hand_weights=None,
                            renderer=None):
    for i in range(len(joint_states)):
        p.resetBasePositionAndOrientation(o_id, 
                                          object_states_keyframe[i][:3],
                                          object_states_keyframe[i][3:])
        setStates(hand_id, joint_states[i])
        if not base_states == None:
            p.resetBasePositionAndOrientation(hand_id,
                                              base_states[i][0],
                                              base_states[i][1])
        # Create Visual features # need to change this
        hand_drake.draw_desired_positions(desired_positions[i],hand_weights[i])
        if not renderer == None:
            print("Press q to continue")
            renderer.render(blocking=True)
        else:
            input("Press Enter to continue")
            

def animate(o_id, hand_id, object_poses, object_orns, joint_states, base_states=None, renderer=None, gif_name="default"):
    # Should be full joint states
    with imageio.get_writer(f"data/video/{gif_name}.gif", mode="I") as writer:
        for i in range(len(joint_states)):
            p.resetBasePositionAndOrientation(o_id, object_poses[i], object_orns[i])
            setStates(hand_id, joint_states[i])
            if not base_states==None:
                p.resetBasePositionAndOrientation(hand_id, 
                                                base_states[0][i], 
                                                base_states[1][i])
            time.sleep(10 * TS)
            if not renderer == None:
                image = renderer.render()
                writer.append_data(image)

def animate_states(o_id,hand_id, object_states, joint_states, base_states=None, renderer=None):
    for i in range(len(joint_states)):
        p.resetBasePositionAndOrientation(o_id, object_states[i][:3], object_states[i][3:])
        setStates(hand_id, joint_states[i])
        if not base_states == None:
            p.resetBasePositionAndOrientation(hand_id, 
                                              base_states[i][0], 
                                              base_states[i][1])
        time.sleep(20 * TS)
        if not renderer == None:
            renderer.render()


def convert_quat_for_bullet(quat):
    return np.array([quat[1],quat[2],quat[3],quat[0]])

def convert_quat_for_drake(quat):
    return np.array([quat[3], quat[0], quat[1], quat[2]])

def convert_joints_for_drake(joint_states, dofs=21, baseOffset=4):
    drake_states = np.zeros(dofs)
    drake_states[:baseOffset+1] = joint_states[:baseOffset+1]
    for i in range(4):
        drake_states[1+baseOffset+i*4:1+baseOffset+(i+1)*4] = joint_states[1+baseOffset+i*5:5+baseOffset+i*5]
    return drake_states

def convert_joints_for_bullet(joint_states, dofs=21, baseOffset=4):
    # Include all fixed joints
    bullet_states = np.zeros(dofs + 4)
    bullet_states[:baseOffset+1] = joint_states[:baseOffset+1]
    for i in range(4):
        bullet_states[1+baseOffset+i*5:5+baseOffset+i*5] = joint_states[1+baseOffset+i*4:1+baseOffset+(i+1)*4]
    return bullet_states

# Only reset part of joint state that related to a finger
def set_finger_joints(joint_states, finger_name, target_state):
    j_states = joint_states.copy()
    idxs = finger_idxs[finger_name]
    j_states[idxs] = target_state
    return j_states

def convert_q_bullet_to_matrix(q):
    q_drake = torch.from_numpy(convert_quat_for_drake(q))
    matrix = pk.quaternion_to_matrix(q_drake).numpy()
    return matrix

# Get Determinitics sampling sigma points of Standard Gaussian
def getSGSigmapoints(dimensions, sigma):
    pos_points = torch.eye(dimensions) * sigma
    neg_points = -torch.eye(dimensions) * sigma
    center = torch.zeros(1,dimensions)
    return torch.vstack([center, pos_points, neg_points])

    
