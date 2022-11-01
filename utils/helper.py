import numpy as np
import scipy.spatial.transform as tf
import pyquaternion as pyq
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
                     tip_ids=None,
                     desired_positions=None,
                     hand_weights=None,
                     renderer=None):
    for i in range(len(joint_states)):
        p.resetBasePositionAndOrientation(o_id, 
                                          object_poses_keyframe[i],
                                          object_orns_keyframe[i])
        setStates(hand_id, joint_states[i])
        if not (base_states is None):
            p.resetBasePositionAndOrientation(hand_id,
                                              base_states[i,:3],
                                              base_states[i,3:])
        # Create Visual features # need to change this
        if not (desired_positions is None):
            default_orn = [0, 0, 0, 1]
            obsolete_pos = [100, 100, 100]
            for j, desired_position in enumerate(desired_positions[i]):
                #print(desired_position)
                if desired_position[0] < 50:
                    p.resetBasePositionAndOrientation(tip_ids[j], desired_position[:3], default_orn)
                else:
                    p.resetBasePositionAndOrientation(tip_ids[j], obsolete_pos, default_orn)
        
        if not (renderer is None):
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
            
def animate(o_id, 
            hand_id, 
            object_poses, 
            object_orns, 
            joint_states, 
            base_states=None, 
            renderer=None, 
            gif_name="default",
            desired_positions=None,
            tip_ids=None):
    # Should be full joint states
    if renderer is None:
        for i in range(len(joint_states)+5):
            if i < len(joint_states):
                p.resetBasePositionAndOrientation(o_id, object_poses[i], object_orns[i])
                if hand_id != -1:
                    setStates(hand_id, joint_states[i])
            else:
                p.resetBasePositionAndOrientation(o_id, object_poses[-1], object_orns[-1])
                if hand_id != -1:
                    setStates(hand_id, joint_states[-1])
            if not (base_states is None) and hand_id != -1:
                if i<len(joint_states):
                    p.resetBasePositionAndOrientation(hand_id, 
                                                      base_states[i,:3],
                                                      base_states[i,3:])
                else:
                    p.resetBasePositionAndOrientation(hand_id, 
                                                      base_states[-1,:3],
                                                      base_states[-1,3:])
            if not (desired_positions is None):
                default_orn = [0, 0, 0, 1]
                obsolete_pos = [100, 100, 100]
                index = i if i<len(joint_states) else -1
                for j, desired_position in enumerate(desired_positions[index]):
                    if desired_position[0] < 50:
                        p.resetBasePositionAndOrientation(tip_ids[j], desired_position[:3], default_orn)
                    else:
                        p.resetBasePositionAndOrientation(tip_ids[j], obsolete_pos, default_orn)
            time.sleep(30 * TS)
    else:
        with imageio.get_writer(f"data/video/{gif_name}.gif", mode="I") as writer:
            for i in range(len(joint_states)+5):
                if i < len(joint_states):
                    p.resetBasePositionAndOrientation(o_id, object_poses[i], object_orns[i])
                    if hand_id != -1:
                        setStates(hand_id, joint_states[i])
                else:
                    p.resetBasePositionAndOrientation(o_id, object_poses[-1], object_orns[-1])
                    if hand_id != -1:
                        setStates(hand_id, joint_states[-1])
                if base_states is not None and hand_id != -1:
                    if i < len(joint_states):
                        p.resetBasePositionAndOrientation(hand_id, 
                                                        base_states[i,:3], 
                                                        base_states[i,3:])
                    else:
                        p.resetBasePositionAndOrientation(hand_id,
                                                        base_states[-1,:3],
                                                        base_states[-1,3:])
                if not (desired_positions is None):
                    default_orn = [0, 0, 0, 1]
                    obsolete_pos = [100, 100, 100]
                    index = i if i < len(joint_states) else -1
                    for j, desired_position in enumerate(desired_positions[index]):
                        if desired_position[0] < 50:
                            p.resetBasePositionAndOrientation(tip_ids[j], desired_position[:3], default_orn)
                        else:
                            p.resetBasePositionAndOrientation(tip_ids[j], obsolete_pos, default_orn)
                time.sleep(30 * TS)
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

def animate_base(o_id, hand_id, object_poses,object_orns, base_states,fixed_joint_states,gif_name="default", desired_positions=None,tip_ids=None,renderer=None):
    # Should be full joint states
    if hand_id != -1:
        setStates(hand_id, fixed_joint_states)
    if renderer is None:
        for i in range(len(object_poses)+50):
            if i < len(object_poses):
                p.resetBasePositionAndOrientation(o_id, object_poses[i], object_orns[i])
            else:
                p.resetBasePositionAndOrientation(o_id, object_poses[-1], object_orns[-1])
            if base_states is not None and hand_id != -1:
                if i<len(object_poses):
                    p.resetBasePositionAndOrientation(hand_id, 
                                                      base_states[i,:3],
                                                      base_states[i,3:])
                else:
                    p.resetBasePositionAndOrientation(hand_id, 
                                                      base_states[-1,:3],
                                                      base_states[-1,3:])
            if not (desired_positions is None):
                default_orn = [0, 0, 0, 1]
                obsolete_pos = [100, 100, 100]
                index = i if i<len(object_poses) else -1
                for j, desired_position in enumerate(desired_positions[index]):
                    if desired_position[0] < 50:
                        p.resetBasePositionAndOrientation(tip_ids[j], desired_position[:3], default_orn)
                    else:
                        p.resetBasePositionAndOrientation(tip_ids[j], obsolete_pos, default_orn)
            time.sleep(30 * TS)
    else:
        with imageio.get_writer(f"data/video/{gif_name}.gif", mode="I") as writer:
            for i in range(len(object_poses)+50):
                if i < len(object_poses):
                    p.resetBasePositionAndOrientation(o_id, object_poses[i], object_orns[i])
                else:
                    p.resetBasePositionAndOrientation(o_id, object_poses[-1], object_orns[-1])
                if base_states is not None and hand_id != -1:
                    if i < len(object_poses):
                        p.resetBasePositionAndOrientation(hand_id, 
                                                        base_states[i,:3], 
                                                        base_states[i,3:])
                    else:
                        p.resetBasePositionAndOrientation(hand_id,
                                                        base_states[-1,:3],
                                                        base_states[-1,3:])
                if desired_positions is not None:
                    default_orn = [0, 0, 0, 1]
                    obsolete_pos = [100, 100, 100]
                    index = i if i < len(object_poses) else -1
                    for j, desired_position in enumerate(desired_positions[index]):
                        if desired_position[0] < 50:
                            p.resetBasePositionAndOrientation(tip_ids[j], desired_position[:3], default_orn)
                        else:
                            p.resetBasePositionAndOrientation(tip_ids[j], obsolete_pos, default_orn)
                time.sleep(30 * TS)
                image = renderer.render()
                writer.append_data(image)

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

def apply_drake_q_rotation(q, vec, invert=False):
    vec = torch.from_numpy(vec)
    if invert:
        q = pk.quaternion_invert(torch.from_numpy(q))
    else:
        q = torch.from_numpy(q)
    rot_vec = pk.quaternion_apply(q, vec).numpy()
    return rot_vec

def express_tips_world_frame(tip_poses, tip_normals, obj_pose):
    new_tip_poses = np.zeros_like(tip_poses)
    new_tip_normals = np.zeros_like(tip_normals)
    obj_pos = obj_pose[:3]
    obj_orn = np.array([obj_pose[6], obj_pose[3], obj_pose[4], obj_pose[5]])
    for i, (normal, tip_pose) in enumerate(zip(tip_normals, tip_poses)):
        new_tip_normals[i] = apply_drake_q_rotation(obj_orn, normal)
        new_tip_poses[i] = apply_drake_q_rotation(obj_orn, tip_pose) + obj_pos
    return new_tip_poses, new_tip_normals

def interp_pybullet_quaternion(q1, q2, num_interp):
    """
    Using Scipy to interpolate quaternions, Scipy uses [x,y,z,w] convention, same as pybullet.
    Interpolating the quaternion using Slerp Method:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html
    """
    R = tf.Rotation.from_quat([q1,q2])
    slerp = tf.Slerp([0,1], R)
    times = np.linspace(0,1, num_interp)
    interp_rots = slerp(times).as_quat()
    return interp_rots

# Get Determinitics sampling sigma points of Standard Gaussian
def getSGSigmapoints(dimensions, sigma):
    pos_points = torch.eye(dimensions) * sigma
    neg_points = -torch.eye(dimensions) * sigma
    center = torch.zeros(1,dimensions)
    return torch.vstack([center, pos_points, neg_points])

def quaternion_distance(q1, q2):
    Q1 = pyq.Quaternion(q1)
    Q2 = pyq.Quaternion(q2)
    dist = pyq.Quaternion.absolute_distance(Q1, Q2)
    return dist
