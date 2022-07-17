import pybullet as p
from allegro_hand import AllegroHandDrake
import numpy as np
import torch
import time
import utils.rigidBodySento as rb
import model.param as model_params
import pytorch_kinematics as pk

TS = 1/250.

def parse_finger_motion_data(data):
    tip_poses = []
    tip_weights = []
    for i in range(data.shape[0]):
        tip_pose = {"thumb":data[i,0] * model_params.SCALE,
                    "ifinger":data[i,1] * model_params.SCALE,
                    "mfinger":data[i,2] * model_params.SCALE,
                    "rfinger":data[i,3] * model_params.SCALE}
        tip_weight = {"thumb":1 if data[i,0].sum() < 100 else 0,
                      "ifinger":1 if data[i,1].sum() < 100 else 0,
                      "mfinger":1 if data[i,2].sum() < 100 else 0,
                      "rfinger":1 if data[i,3].sum() < 100 else 0}
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

def animate_object(o_id, object_poses, object_orns):
    for i in range(len(object_poses)):
        p.resetBasePositionAndOrientation(o_id, 
                                          object_poses[i], 
                                          object_orns[i])
        time.sleep(20 * TS)

def setStates(hand_id, states):
    for i in range(len(states)):
        p.resetJointState(hand_id, i, states[i])

def animate_keyframe(o_id, 
                     hand_id,
                     object_poses,  # Already extracted as key frame
                     object_orns, 
                     joint_states, 
                     base_states,
                     hand_drake,
                     desired_positions,
                     hand_weights):
    for i in range(len(joint_states)):
        p.resetBasePositionAndOrientation(o_id, 
                                          object_poses[i],
                                          object_orns[i])
        setStates(hand_id, joint_states[i])
        p.resetBasePositionAndOrientation(hand_id,
                                          base_states[i][0],
                                          base_states[i][1])
        # Create Visual features # need to change this
        hand_drake.draw_desired_positions(desired_positions[i],hand_weights[i])
        input("Press enter to continue")

def animate(o_id, hand_id, object_poses, object_orns, joint_states, base_states):
    # Should be full joint states
    for i in range(len(joint_states)):
        p.resetBasePositionAndOrientation(o_id, object_poses[i], object_orns[i])
        setStates(hand_id, joint_states[i])
        p.resetBasePositionAndOrientation(hand_id, 
                                          base_states[0][i], 
                                          base_states[1][i])
        time.sleep(20 * TS)

def fit_finger_tips(targets, weights, object_poses, object_orns, hand_drake):
    joint_states = []
    base_states = []
    success_flags = []
    desired_positions = []
    for i in range(len(targets)):
        hand_drake.setObjectPose(object_poses[i],object_orns[i])
        joint_state, base_state, desired_position, success = hand_drake.regressFingerTipPos(
            targets[i],
            weights[i],
            has_normal=True)
        joint_states.append(joint_state)
        base_states.append(base_state)
        success_flags.append(success)
        desired_positions.append(desired_position)
        print("Finished Fit:",i+1, f"Result is: {success}")
    return joint_states, base_states, desired_positions, success

# Steps should be a list, record gap between each base key states
def interpolate_joints(joint_states, steps=[75,75,75,75]):
    result = np.zeros((sum(steps), len(joint_states[0])))
    cur = 0
    for i in range(len(steps)):
        result[cur:cur+steps[i]] = np.linspace(
                                      start=joint_states[i],
                                      stop=joint_states[i+1],
                                      num=steps[i])
        cur += steps[i]
    return result


def convert_q_for_bullet(q):
    return torch.tensor([q[1], q[2], q[3], q[0]])

def convert_q_for_drake(q):
    return torch.tensor([q[3], q[0], q[1], q[2]])

# Quaternion need to convert to euler angle for interpolation
# TODO: (Eric) maybe problematic
def interpolate_base(base_states, steps=[75,75,75,75]):
    result_pos = np.zeros((sum(steps), 3))
    result_orn = np.zeros((sum(steps), 4))
    cur = 0
    for i in range(len(steps)):
        result_pos[cur:cur+steps[i]] = np.linspace(
                                          start=base_states[i][0],
                                          stop=base_states[i+1][0],
                                          num=steps[i])
        start_orn = pk.matrix_to_euler_angles(
            pk.quaternion_to_matrix(convert_q_for_drake(base_states[i][1])),"XYZ").numpy()
        end_orn = pk.matrix_to_euler_angles(
            pk.quaternion_to_matrix(convert_q_for_drake(base_states[i+1][1])), "XYZ").numpy()
        euler_result_orn = np.linspace(
                              start=start_orn,
                              stop=end_orn,
                              num=steps[i])
        for j in range(len(euler_result_orn)):
            matrix = pk.euler_angles_to_matrix(torch.from_numpy(euler_result_orn[j]),"XYZ")
            q = pk.matrix_to_quaternion(matrix).numpy()
            result_orn[cur+j] = convert_q_for_bullet(q).numpy()
        cur += steps[i]
    return result_pos, result_orn

if __name__ == "__main__":
    # np.random.seed(3)
    finger_tip_data = np.load("new_tip_poses.npy")
    object_data = np.load("object_poses.npy")

    tip_poses, tip_weights = parse_finger_motion_data(finger_tip_data)
    obj_poses, obj_orns = parse_object_motion_data(object_data)
    
    hand_drake = AllegroHandDrake(object_collidable=True)
    
    keyframe_obj_poses = obj_poses[::75]
    keyframe_obj_orns = obj_orns[::75]
    joint_states, base_states, desired_positions, _ = fit_finger_tips(tip_poses, 
                                                tip_weights, 
                                                keyframe_obj_poses, 
                                                keyframe_obj_orns,
                                                hand_drake)
    
    p.connect(p.GUI)
    # Create Different pybullet objects
    obj_id = rb.create_primitive_shape(p, 1.0, p.GEOM_BOX, 
                                       (0.2 * model_params.SCALE,0.2 * model_params.SCALE, 0.05 * model_params.SCALE),
                                       color = (0.6, 0, 0, 0.8), collidable=True)
    hand_drake.createTipsVisual()
    hand_id = p.loadURDF("model/resources/allegro_hand_description/urdf/allegro_hand_description_right.urdf", useFixedBase=1)
    animate_keyframe(obj_id,
                     hand_id,
                     keyframe_obj_poses, 
                     keyframe_obj_orns, 
                     joint_states, 
                     base_states, 
                     hand_drake,
                     desired_positions,
                     tip_weights) # Good to go
    full_joint_states = interpolate_joints(joint_states)
    full_base_states = interpolate_base(base_states)
    np.savez("ik_result_2.npz",joint_states=full_joint_states,
                             base_pos=full_base_states[0], 
                             base_orn=full_base_states[1])
    animate(obj_id, hand_id, obj_poses, obj_orns, full_joint_states, full_base_states)

    

    

    