import pybullet as p
from allegro_hand import AllegroHandDrake
import numpy as np
import torch
import time
import rigidBodySento as rb
import  model.param as model_params
import pytorch_kinematics as pk
import helper
import render
from argparse import ArgumentParser

# Here
def fit_interp(targets, 
               weights, 
               q_sequence, 
               object_poses, 
               object_orns, 
               arm_drake,
               N=3):
    object_states = np.hstack([object_poses, object_orns])
    print(object_states.shape)
    seg_len = 75
    q_all = []
    q_sequence_drake = np.zeros((len(q_sequence), 21))
    for i in range(len(q_sequence)):
        q_sequence_drake[i] = helper.convert_joints_for_drake(q_sequence[i])

    for i in range(len(targets)-1):
        arm_drake.setObjectPose(object_poses[0], object_orns[0])
        q_sol, q_full, no_collision, match_finger_tips = arm_drake.solveInterpTest(
            targets[i],
            weights[i],
            q_sequence_drake[i],
            q_sequence_drake[i+1],
            object_states[i*seg_len:(i+1)*seg_len],
            has_normal=True,
            N=N)
        
        q_all.append(q_full)
        print(f"Collision: {no_collision}, Finger Tips: {match_finger_tips}")
    q_all = np.vstack(q_all)
    return q_all, q_sol

# Input key frame and key finger tip pose.
def fit_interp_ik(targets, 
                  weights, 
                  object_poses_keyframe, 
                  object_orns_keyframe, 
                  arm_drake):
    joint_states = []
    success_flags = []
    desired_positions = []
    for i in range(len(targets)):
        arm_drake.setObjectPose(object_poses_keyframe[i], object_orns_keyframe[i])
        if i==0:
            q_sol, joint_state, _, desired_position, success = arm_drake.regressFingerTipPosWithRandomSearch(
                targets[i],
                weights[i],
                has_normal=True)
        else:
            q_sol, joint_state, _, desired_position, success = arm_drake.regressFingerTipPosWithRandomSearch(
                targets[i],
                weights[i],
                has_normal=True,
                prev_q = q_sol)
        joint_states.append(joint_state)
        success_flags.append(success)
        desired_positions.append(desired_position)
        print("Finished Fit:",i+1, f"Result is: {success}")
    return joint_states, desired_positions, success_flags

if __name__ == "__main__":
    # np.random.seed(3)
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="default")
    args = parser.parse_args()
    finger_tip_data = np.load(f"data/tip_data/{args.exp_name}_tip_poses.npy")
    object_data = np.load(f"data/object_poses/{args.exp_name}_object_poses.npy")
    #key_joint_states = np.load("data/key_joint_states_default.npy")

    tip_poses, tip_weights = helper.parse_finger_motion_data(finger_tip_data)
    obj_poses, obj_orns = helper.parse_object_motion_data(object_data)
    
    p_client = p.connect(p.DIRECT)
    renderer = render.PyBulletRenderer()

    arm_drake = AllegroHandDrake(useFixedBase=True,
                                 robot_path=model_params.allegro_arm_urdf_path,
                                 baseOffset=model_params.allegro_arm_offset,
                                 all_fingers=model_params.AllAllegroArmFingers,
                                 object_collidable=True,
                                 sequenceSolve=True)
    
    keyframe_obj_poses = obj_poses[::10]
    keyframe_obj_orns = obj_orns[::10]
    tip_poses_keyframe = tip_poses[::10]
    tip_weights_keyframe = tip_weights[::10]
    joint_states, desired_positions, success_flags = fit_interp_ik(tip_poses_keyframe, 
                                                                   tip_weights_keyframe,
                                                                   keyframe_obj_poses, 
                                                                   keyframe_obj_orns, 
                                                                   arm_drake)
    arm_drake.createTipsVisual()

    helper.animate_keyframe(arm_drake.obj_id,
                            arm_drake.hand_id,
                            keyframe_obj_poses, 
                            keyframe_obj_orns, 
                            joint_states,
                            None, 
                            arm_drake,
                            renderer=renderer) # Good to go

    
    # full_joint_states_drake,joint_frames_drake = fit_interp(tip_poses[2:],
    #                                      tip_weights[2:],
    #                                      key_joint_states[2:],
    #                                      obj_poses[150:300],
    #                                      obj_orns[150:300],
    #                                      arm_drake)

    # full_joint_states = np.zeros((len(full_joint_states_drake), 25)) # Should be 75
    # solved_joint_frames = np.zeros((4, 25))
    # for i in range(len(full_joint_states)):
    #     full_joint_states[i] = helper.convert_joints_for_bullet(full_joint_states_drake[i])
    # for i in range(4):
    #     solved_joint_frames[i] = helper.convert_joints_for_bullet(joint_frames_drake[i])
    # p.connect(p.DIRECT)
    # renderer = render.PyBulletRenderer()
    # # Create Different pybullet objects
    # obj_id = rb.create_primitive_shape(p, 1.0, p.GEOM_BOX, 
    #                                    (0.2 * model_params.SCALE,0.2 * model_params.SCALE, 0.05 * model_params.SCALE),
    #                                    color = (0.6, 0, 0, 0.8), collidable=True)
    # arm_drake.createTipsVisual()
    # hand_id = p.loadURDF("model/resources/allegro_hand_description/urdf/allegro_arm.urdf", useFixedBase=1)
    # helper.animate_keyframe(obj_id,
    #                         hand_id,
    #                         keyframe_obj_poses[2:], 
    #                         keyframe_obj_orns[2:], 
    #                         key_joint_states[2:], 
    #                         None, 
    #                         arm_drake,
    #                         renderer=renderer) # Good to go
    # obj_poses_key = obj_poses[225:300:25]
    # obj_poses_key.append(obj_poses[-1])
    # obj_orns_key = obj_orns[225:300:25]
    # obj_orns_key.append(obj_orns[-1])
    # helper.animate_keyframe(obj_id,
    #                         hand_id,
    #                         obj_poses_key, 
    #                         obj_orns_key, 
    #                         solved_joint_frames, 
    #                         None, 
    #                         arm_drake,
    #                         renderer=renderer)
    
    # print(len(full_joint_states))
    # helper.animate(obj_id, hand_id, obj_poses[150:300], obj_orns[150:300], full_joint_states, renderer=renderer)

    

    

    