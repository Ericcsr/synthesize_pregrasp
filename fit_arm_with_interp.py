import pybullet as p
from allegro_hand import AllegroHandDrake
import numpy as np
import torch
import time
import utils.rigidBodySento as rb
import  model.param as model_params
import pytorch_kinematics as pk
import helper
import utils.render as render
from argparse import ArgumentParser

def fit_finger_tips(targets, weights, object_poses_keyframe, object_orns_keyframe, arm_drake):
    joint_states = []
    success_flags = []
    desired_positions = []
    for i in range(len(targets)):
        arm_drake.setObjectPose(object_poses_keyframe[i],object_orns_keyframe[i])
        # Need to enable sequence 
        joint_state, _, desired_position, success = arm_drake.regressFingerTipPosWithRandomSearch(
            targets[i],
            weights[i],
            has_normal=True)
        joint_states.append(joint_state)
        success_flags.append(success)
        desired_positions.append(desired_position)
        print("Finished Fit:",i+1, f"Result is: {success}, {joint_state.shape}")
    return joint_states, desired_positions, success

# Here
def fit_interp(targets, 
               weights, 
               q_sequence, 
               object_poses, 
               object_orns, 
               arm_drake,
               N=5):
    seg_len = len(object_poses) // len(q_sequence) # 75
    object_states = np.hstack([object_poses, object_orns])
    q_all = []
    flag = True
    q_sequence_drake = np.zeros((len(q_sequence), 21))
    for i in range(len(q_sequence)):
        q_sequence_drake[i] = helper.convert_joints_for_drake(q_sequence[i])

    for i in range(len(targets)-1):
        arm_drake.setObjectPose(object_poses[i*seg_len], object_orns[i*seg_len])
        q_sol, q_full, success = arm_drake.solveInterp(
            targets[i],
            weights[i],
            q_sequence_drake[i],
            q_sequence_drake[i+1],
            object_states[i*seg_len:(i+1)*seg_len],
            has_normal=True,
            N=N) # Definition may be wrong.. Or not if not using test mode?
        flag = flag and success
        q_all.append(q_full)
        print("Result of Interpolation:",success)
    q_all = np.vstack(q_all)
    return q_all
        
    
    

if __name__ == "__main__":
    # np.random.seed(3)
    parser = ArgumentParser()
    parser.add_argument("--exp_name",type=str, default="default")
    args = parser.parse_args()
    finger_tip_data = np.load("data/new_tip_poses.npy")
    object_data = np.load("data/object_poses.npy")

    tip_poses, tip_weights = helper.parse_finger_motion_data(finger_tip_data)
    obj_poses, obj_orns = helper.parse_object_motion_data(object_data)
    
    arm_drake = AllegroHandDrake(useFixedBase=True,
                                  robot_path=model_params.allegro_arm_urdf_path,
                                  baseOffset=model_params.allegro_arm_offset,
                                  all_fingers = model_params.AllAllegroArmFingers,
                                  object_collidable=True)
    
    keyframe_obj_poses = obj_poses[::75]
    keyframe_obj_orns = obj_orns[::75]
    joint_states, desired_positions, success = fit_finger_tips(tip_poses, 
                                                  tip_weights, 
                                                  keyframe_obj_poses, 
                                                  keyframe_obj_orns,
                                                  arm_drake)
    full_joint_states_drake = fit_interp(tip_poses,
                                         tip_weights,
                                         joint_states,
                                         obj_poses,
                                         obj_orns,
                                         arm_drake)

    full_joint_states = np.zeros((len(full_joint_states_drake), 25))
    for i in range(len(full_joint_states)):
        full_joint_states[i] = helper.convert_joints_for_bullet(full_joint_states_drake[i])
    np.save(f"key_joint_states_{args.exp_name}.npy", joint_states)
    p.connect(p.DIRECT)
    renderer = render.PyBulletRenderer()
    # Create Different pybullet objects
    obj_id = rb.create_primitive_shape(p, 1.0, p.GEOM_BOX, 
                                       (0.2 * model_params.SCALE,0.2 * model_params.SCALE, 0.05 * model_params.SCALE),
                                       color = (0.6, 0, 0, 0.8), collidable=True)
    arm_drake.createTipsVisual()
    hand_id = p.loadURDF("model/resources/allegro_hand_description/urdf/allegro_arm.urdf", useFixedBase=1)
    print(desired_positions)
    print(tip_weights)
    helper.animate_keyframe(obj_id,
                            hand_id,
                            keyframe_obj_poses, 
                            keyframe_obj_orns, 
                            joint_states, 
                            None, 
                            arm_drake,
                            desired_positions,
                            tip_weights,
                            renderer=renderer) # Good to go
    
    #np.savez("ik_result_2.npz",joint_states=full_joint_states,
    #                         base_pos=full_base_states[0], 
    #                         base_orn=full_base_states[1])
    print(len(full_joint_states))
    helper.animate(obj_id, hand_id, obj_poses[:len(full_joint_states)], obj_orns[:len(full_joint_states)], full_joint_states, renderer=renderer)

    

    

    