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

# Here randomly sample some point from object surface that need to be seperate enough, 
def fit_keypoints_ik(targets, 
                  weights, 
                  object_poses_keyframe, 
                  object_orns_keyframe, 
                  arm_drake):
    joint_states = []
    success_flags = []
    desired_positions = []
    pcs =  []
    for i in range(len(targets)):
        pcs.append(arm_drake.setObjectPose(object_poses_keyframe[i], object_orns_keyframe[i]))
        if i==0:
            q_sol, joint_state, _, desired_position, success = arm_drake.regressFingerTipPosWithRandomSearch(
                targets[i],
                weights[i],
                has_normal=True,
                n_process=64,
                interp_mode=False)
        else:
            q_sol, joint_state, _, desired_position, success = arm_drake.regressFingerTipPosWithRandomSearch(
                targets[i],
                weights[i],
                has_normal=True,
                prev_q = q_sol,
                n_process=64,
                interp_mode=False)
        joint_states.append(joint_state)
        success_flags.append(success)
        desired_positions.append(desired_position)
        print("Finished Fit:",i+1, f"Result is: {success}")
    return joint_states, desired_positions, success_flags, pcs

if __name__ == "__main__":
    # np.random.seed(3)
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--stage", type=str, default="keyframe")
    args = parser.parse_args()
    finger_tip_data = np.load(f"data/tip_data/{args.exp_name}_tip_poses.npy")
    object_data = np.load(f"data/object_poses/{args.exp_name}_object_poses.npy")
    try:
        key_joint_states = np.load(f"data/ik/{args.exp_name}_keyframes.npy")
    except:
        print("Key frame is not avaliable!")

    tip_poses, tip_weights = helper.parse_finger_motion_data(finger_tip_data)
    obj_poses, obj_orns = helper.parse_object_motion_data(object_data)
    
    p_client = p.connect(p.DIRECT)
    renderer = render.PyBulletRenderer()
    arm_drake = AllegroHandDrake(useFixedBase=True,
                                 robot_path=model_params.allegro_arm_urdf_path,
                                 baseOffset=model_params.allegro_arm_offset,
                                 all_fingers=model_params.AllAllegroArmFingers,
                                 object_collidable=True)


    if args.stage == "keyframe":
        kf_obj_poses = obj_poses[::50]
        kf_obj_orns = obj_orns[::50]
        tip_poses_kf = tip_poses[::50]
        tip_weights_kf = tip_weights[::50]

        joint_states, desired_positions, success_flags, pcs = fit_keypoints_ik(tip_poses_kf,
                                                                            tip_weights_kf,
                                                                            kf_obj_poses,
                                                                            kf_obj_orns,
                                                                            arm_drake)

        helper.animate_keyframe(arm_drake.obj_id,
                                arm_drake.hand_id,
                                kf_obj_poses,
                                kf_obj_orns,
                                joint_states,
                                None,arm_drake,renderer=renderer)
        np.save(f"data/ik/{args.exp_name}_keyframes.npy", joint_states)
    elif args.stage == "first":
        interp_obj_poses =   [obj_poses[10], obj_poses[20], obj_poses[30], obj_poses[40]]
        interp_obj_orns =    [obj_orns[10], obj_orns[20], obj_orns[30], obj_orns[40]]
        tip_poses_interp =   [tip_poses[10],tip_poses[20],tip_poses[30],tip_poses[40]]
        tip_weights_interp = [tip_weights[10],tip_weights[20],tip_weights[30],tip_weights[40]]
        joint_states, desired_positions, success_flags, pcs = fit_interp_ik(tip_poses_interp, 
                                                                            tip_weights_interp,
                                                                            interp_obj_poses, 
                                                                            interp_obj_orns, 
                                                                            arm_drake,
                                                                            key_joint_states[0],
                                                                            key_joint_states[1])

        helper.animate_keyframe(arm_drake.obj_id,
                                arm_drake.hand_id,
                                np.vstack([obj_poses[0],interp_obj_poses, obj_poses[50]]), 
                                np.vstack([obj_orns[0],interp_obj_orns, obj_orns[50]]), 
                                np.vstack([key_joint_states[0],joint_states,key_joint_states[1]]),
                                None, 
                                arm_drake,
                                renderer=renderer) # Good to go
        np.save(f"data/ik/{args.exp_name}_segment1.npy", joint_states)
    elif args.stage == "second":
        interp_obj_poses = [obj_poses[60],obj_poses[70],obj_poses[80],obj_poses[90]]
        interp_obj_orns =  [obj_orns[60],obj_orns[70],obj_orns[80],obj_orns[90]]
        tip_poses_interp = [tip_poses[60],tip_poses[70],tip_poses[80],tip_poses[90]]
        tip_weights_interp = [tip_weights[60],tip_weights[70],tip_weights[80],tip_weights[90]]
        joint_states, desired_positions, success_flags, pcs = fit_interp_ik(tip_poses_interp, 
                                                                            tip_weights_interp,
                                                                            interp_obj_poses, 
                                                                            interp_obj_orns, 
                                                                            arm_drake,
                                                                            key_joint_states[1],
                                                                            key_joint_states[2])

        helper.animate_keyframe(arm_drake.obj_id,
                                arm_drake.hand_id,
                                np.vstack([obj_poses[50],interp_obj_poses, obj_poses[100]]), 
                                np.vstack([obj_orns[50],interp_obj_orns, obj_orns[100]]), 
                                np.vstack([key_joint_states[1],joint_states,key_joint_states[2]]),
                                None, 
                                arm_drake,
                                renderer=renderer) # Good to go
        np.save(f"data/ik/{args.exp_name}_segment2.npy", joint_states)
    elif args.stage == "third":
        interp_obj_poses = [obj_poses[110],obj_poses[120],obj_poses[130],obj_poses[140], obj_poses[149]]
        interp_obj_orns =  [obj_orns[110],obj_orns[120],obj_orns[130],obj_orns[140],obj_orns[149]]
        tip_poses_interp = [tip_poses[110],tip_poses[120],tip_poses[130],tip_poses[140], tip_poses[149]]
        tip_weights_interp = [tip_weights[110],tip_weights[120],tip_weights[130],tip_weights[140], tip_weights[149]]
        joint_states, desired_positions, success_flags, pcs = fit_exterp_ik(tip_poses_interp, 
                                                                            tip_weights_interp,
                                                                            interp_obj_poses, 
                                                                            interp_obj_orns, 
                                                                            arm_drake,
                                                                            key_joint_states[2])

        helper.animate_keyframe(arm_drake.obj_id,
                                arm_drake.hand_id,
                                np.vstack([obj_poses[100],interp_obj_poses]), 
                                np.vstack([obj_orns[100],interp_obj_orns]), 
                                np.vstack([key_joint_states[2],joint_states]),
                                None, 
                                arm_drake,
                                renderer=renderer) # Good to go
        np.save(f"data/ik/{args.exp_name}_segment3.npy", joint_states)
    elif args.stage == "final":
        # Interpolate and animate all the frames..
        try:
            segment_1 = np.load(f"data/ik/{args.exp_name}_segment1.npy")
            segment_2 = np.load(f"data/ik/{args.exp_name}_segment2.npy")
            segment_3 = np.load(f"data/ik/{args.exp_name}_segment3.npy")
            segments = np.vstack([key_joint_states[0],segment_1,key_joint_states[1],segment_2, key_joint_states[2],segment_3])
        except:
            print("Segment data is not ready")
    

        full_joint_states = np.zeros((150, 25))
        for i in range(15):
            if i != 14:
                full_joint_states[i*10:i*10+10] = np.linspace(segments[i],segments[i+1],10)
            else:
                full_joint_states[i*10:i*10+9] = np.linspace(segments[i],segments[i+1],9)

        helper.animate(arm_drake.obj_id, arm_drake.hand_id, obj_poses, obj_orns, full_joint_states,renderer=renderer)
        np.save(f"data/ik/{args.exp_name}_full_states.npy", full_joint_states)
    else:
        print("Stage is invalid!")    

    

    