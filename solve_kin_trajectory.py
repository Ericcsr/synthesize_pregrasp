import os
import pybullet as p
import utils.kin_feasibility_check as kin
import numpy as np
import model.param as model_params

import torch
import pytorch_kinematics as pk

import utils.helper as helper
import utils.rigidBodySento as rb
from argparse import ArgumentParser

NUM_KEY_FRAMES=3
NUM_INTERP=4

# TODO: Load object poses as well as finger tip pose trajectory
def load_object_poses(exp_name, extrapolate_frames = 50):
    obj_poses = np.load(f"data/object_poses/{exp_name}_object_poses.npy")
    obj_pos = np.zeros((len(obj_poses)+extrapolate_frames+1, 3))
    obj_orn = np.zeros((len(obj_poses)+extrapolate_frames+1, 4))
    obj_pos[:len(obj_poses)] = obj_poses[:,:3]
    obj_orn[:len(obj_poses)] = obj_poses[:,3:]
    for i in range(len(obj_poses), len(obj_poses)+extrapolate_frames+1):
        obj_pos[i] = obj_poses[-1,:3]
        obj_orn[i] = obj_poses[-1,3:]
    return obj_pos, obj_orn, obj_poses[-1,:3], obj_poses[-1,3:]

def load_stage_one_finger_tip_trajectory(exp_name, active_fingers = [0,1]):
    tip_poses = np.load(f"data/tip_data/{exp_name}_tip_poses.npy")
    full_tip_poses = np.ones((len(tip_poses),4, 6)) * 100
    full_tip_poses[:,:,3:] = 0
    full_tip_poses[:,active_fingers] = tip_poses.copy()
    return full_tip_poses

def load_final_grasp(exp_name, last_obj_pos, last_obj_orn):
    local_tip_pose = np.load(f"data/predicted_grasps/{exp_name}.npy")
    world_tip_pose = np.zeros_like(local_tip_pose)
    drake_last_obj_orn = helper.convert_quat_for_drake(last_obj_orn)
    for i,point in enumerate(local_tip_pose):
        if point[0] < 50:
            world_tip_pose[i,:3] = helper.apply_drake_q_rotation(drake_last_obj_orn, point[:3]) + last_obj_pos
            world_tip_pose[i,3:] = helper.apply_drake_q_rotation(drake_last_obj_orn, point[3:])
        else:
            world_tip_pose[i] = np.array([100., 100., 100., 0, 0, 0])
    return world_tip_pose

def load_trajector_data(exp_name, extrapolate_frames=50, active_fingers=[0,1]):
    obj_poses, obj_orns, last_obj_pos, last_obj_orn = load_object_poses(exp_name, extrapolate_frames)
    stage_one_tip_poses = load_stage_one_finger_tip_trajectory(exp_name, active_fingers)
    print(stage_one_tip_poses[-1])
    stage_two_tip_poses = np.asarray([stage_one_tip_poses[-1]]*(extrapolate_frames+1))
    final_grasp = load_final_grasp(exp_name, last_obj_pos, last_obj_orn)
    stage_two_tip_poses[-2] = final_grasp.copy()
    stage_two_tip_poses[-1] = final_grasp.copy()
    merged_tip_poses = np.vstack([stage_one_tip_poses, stage_two_tip_poses])
    return obj_poses, obj_orns, merged_tip_poses

# TODO: Solve key frames
def fit_keypoints(num_key_points, obj_poses, obj_orns, tip_poses, hand_model="allegro"):
    """
    Assume obj_poses, obj_orns, tip_poses is full trajectory and have same length
    Assume obj_poses, obj_orns, tip_poses have last frame repeated
    Should include both very begining and very end
    """
    total_traj_len = len(obj_poses)
    segment_length = total_traj_len // num_key_points
    key_idx = list(range(total_traj_len))[::segment_length]
    
    key_obj_poses = obj_poses[key_idx]
    key_obj_orns = obj_orns[key_idx]
    key_tip_poses = tip_poses[key_idx]

    key_joint_states, key_q_sols, desired_positions = kin.solve_ik_keypoints(key_tip_poses, key_obj_poses, key_obj_orns, hand_model=hand_model)
    base_state = [np.hstack([joint_state[0], joint_state[1]]) for joint_state in key_joint_states]
    joints_state = [joint_state[2] for joint_state in key_joint_states]
    return (np.asarray(base_state), np.asarray(joints_state)), key_obj_poses, key_obj_orns, key_q_sols, key_idx, desired_positions

def fit_interp(num_interp, obj_poses, obj_orns, tip_poses, q_start, q_end, hand_model):
    """
    Assume obj_poses, obj_orns, tip_poses, are a segment of full poses (Both ends are keyframes)
    Assume num_interp doesn't include both end
    Assume q_start and q_end are two key point solution
    """
    total_segment_length = len(obj_poses)
    interp_length = total_segment_length // (num_interp + 1) # //5
    interp_idx = list(range(total_segment_length))[::interp_length]
    first_idx = interp_idx.pop(0)
    last_idx = interp_idx.pop(-1)

    interp_obj_poses = obj_poses[interp_idx]
    interp_obj_orns = obj_orns[interp_idx]
    interp_tip_poses = tip_poses[interp_idx]

    interp_joint_state = kin.solve_interpolation(interp_tip_poses, interp_obj_poses, interp_obj_orns, q_start, q_end, hand_model=hand_model)
    
    interp_full_joint_state = []
    interp_full_base_state = []
    interp_idx.append(last_idx)
    interp_idx.insert(0,first_idx)
    for i in range(len(interp_joint_state)-1):
        # Each time include the head instead of tail.
        num_frames = interp_idx[i+1]-interp_idx[i]+1
        interp_full_joint_state.append(np.linspace(interp_joint_state[i][2], interp_joint_state[i+1][2], num_frames)[:-1])
        interp_full_base_state.append(np.hstack([
            np.linspace(interp_joint_state[i][0],interp_joint_state[i+1][0], num_frames)[:-1],
            helper.interp_pybullet_quaternion(interp_joint_state[i][1],interp_joint_state[i+1][1], num_frames)[:-1]]))
    interp_full_joint_state = np.vstack(interp_full_joint_state)
    interp_full_base_state = np.vstack(interp_full_base_state)
    return (interp_full_base_state, interp_full_joint_state)
# TODO: Animate the whole trajectory

def main():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None, required=True)
    parser.add_argument("--mode", type=str, default="keypoints")
    parser.add_argument("--extrapolate_frames", type=int, default=50)
    parser.add_argument("--hand_model", type=str, default="allegro")
    args = parser.parse_args()

    # Create Pybullet Scenario
    p.connect(p.GUI)
    if args.hand_model == "allegro":
        hand_id = p.loadURDF("model/resources/allegro_hand_description/urdf/allegro_hand_description_right.urdf")
    elif args.hand_model == "shadow":
        hand_id = p.loadURDF("model/resources/shadow_hand_description/urdf/shadow_hand_collision.urdf")
    else:
        raise NotImplementedError
    o_id = rb.create_primitive_shape(p, 1.0, p.GEOM_BOX, (0.2, 0.2, 0.05),         # half-extend
                                     color=(0.6, 0, 0, 0.8), collidable=True,
                                     init_xyz=np.array([0, 0, 0.05]),
                                     init_quat=np.array([0, 0, 0, 1]))
    # Create Visualization for finger tip
    colors = [(0.9, 0.9, 0.9, 0.7),
              (0.9, 0.0, 0.0, 0.7),
              (0.0, 0.9, 0.0, 0.7),
              (0.0, 0.0, 0.9, 0.7),
              (0.0, 0.0, 0.0, 0.7)]
    tip_ids = []
    for i in range(4 if args.hand_model =="allegro" else 5):
        tip_ids.append(rb.create_primitive_shape(p, 0.1, p.GEOM_SPHERE,
                                                 (0.01,), color=colors[i], 
                                                 collidable=False, init_xyz = (100,100,100)))

    obj_poses, obj_orns, finger_tip_poses = load_trajector_data(args.exp_name, args.extrapolate_frames)

    if args.mode == "keypoints":
        key_states, key_obj_poses, key_obj_orns, key_q_sols, key_idx, des_poses = fit_keypoints(NUM_KEY_FRAMES, obj_poses, obj_orns, finger_tip_poses, hand_model=args.hand_model)
        helper.animate_keyframe(o_id, hand_id, key_obj_poses, key_obj_orns, key_states[1], key_states[0], tip_ids=tip_ids, desired_positions=des_poses)
        np.savez(f"data/ik/{args.exp_name}_keyframes.npz", q_sols=key_q_sols, key_idx=key_idx)

    elif args.mode.startswith("interp"):
        idx = int(args.mode[-1])
        key_frame_data = np.load(f"data/ik/{args.exp_name}_keyframes.npz")
        start_idx = key_frame_data["key_idx"][idx]
        end_idx = key_frame_data["key_idx"][idx+1]+1 # Should include next keyframe
        segment_obj_poses = obj_poses[start_idx:end_idx]
        segment_obj_orns = obj_orns[start_idx:end_idx]
        segment_tip_poses = finger_tip_poses[start_idx:end_idx]
        q_start = key_frame_data["q_sols"][idx]
        q_end = key_frame_data["q_sols"][idx+1]
        interp_state = fit_interp(NUM_INTERP, segment_obj_poses, segment_obj_orns, segment_tip_poses, q_start, q_end,hand_model=args.hand_model)
        helper.animate(o_id, hand_id, segment_obj_poses, segment_obj_orns, interp_state[1], base_states=interp_state[0])
        np.savez(f"data/ik/{args.exp_name}_segment{idx}.npz", joint_states=interp_state[1], base_states=interp_state[0])
    elif args.mode == "animate":
        file_lists = os.listdir("data/ik")
        joint_states = []
        base_states = []
        for k in range(NUM_KEY_FRAMES):
            assert(f"{args.exp_name}_segment{k}.npz" in file_lists)
            data = np.load(f"data/ik/{args.exp_name}_segment{k}.npz")
            joint_states.append(data["joint_states"].copy())
            base_states.append(data["base_states"].copy())
        joint_states = np.vstack(joint_states)
        base_states = np.vstack(base_states)
        input("Press Enter to animate")
        print(obj_poses.shape, obj_orns.shape, joint_states.shape)
        helper.animate(o_id, hand_id, obj_poses[20:-1], obj_orns[20:-1], joint_states[20:], base_states[20:])
        
if __name__ == "__main__":
    main()