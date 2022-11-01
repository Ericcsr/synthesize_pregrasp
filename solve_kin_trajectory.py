import os
import pybullet as p
from functools import partial
import utils.kin_feasibility_check as kin
import numpy as np
import envs.setup_pybullet as setup_pybullet
import model.manipulation_obj_creator as creator
import model.param as model_param
import model.allegro_hand_rigid_body_model as allegro_rbm
import model.shadow_hand_rigid_body_model as shadow_rbm
import utils.helper as helper
import utils.rigidBodySento as rb
from argparse import ArgumentParser

NUM_KEY_FRAMES=2
NUM_INTERP=4

ALLEGRO_FINGER_MAP = [model_param.AllegroHandFinger.THUMB, 
                  model_param.AllegroHandFinger.INDEX, 
                  model_param.AllegroHandFinger.MIDDLE, 
                  model_param.AllegroHandFinger.RING]

SHADOW_FINGER_MAP = [model_param.ShadowHandFinger.THUMB,
                     model_param.ShadowHandFinger.INDEX,
                     model_param.ShadowHandFinger.MIDDLE,
                     model_param.ShadowHandFinger.RING,
                     model_param.ShadowHandFinger.LITTLE]

# TODO: Load object poses as well as finger tip pose trajectory
def load_object_poses(exp_name, extrapolate_frames = 50, early_term=50*(NUM_KEY_FRAMES-1)):
    obj_poses = np.load(f"data/object_poses/{exp_name}_object_poses.npy")
    length = len(obj_poses) if early_term is None or len(obj_poses) < early_term else early_term
    obj_pos = np.zeros((length+extrapolate_frames+1, 3))
    obj_orn = np.zeros((length+extrapolate_frames+1, 4))
    obj_pos[:length] = obj_poses[:length,:3]
    obj_orn[:length] = obj_poses[:length,3:]
    for i in range(length, length+extrapolate_frames+1):
        obj_pos[i] = obj_poses[length-1,:3]
        obj_orn[i] = obj_poses[length-1,3:]
    return obj_pos, obj_orn, obj_poses[length-1,:3], obj_poses[length-1,3:]

def load_stage_one_finger_tip_trajectory(exp_name, active_fingers = [0,1], early_term=50*(NUM_KEY_FRAMES-1)):
    tip_poses = np.load(f"data/tip_data/{exp_name}_tip_poses.npy")
    length = len(tip_poses) if early_term is None or len(tip_poses)<early_term else early_term
    full_tip_poses = np.ones((length, 5, 6)) * 100
    full_tip_poses[:,:,3:] = 0
    full_tip_poses[:length,active_fingers] = tip_poses[:length].copy()
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
def fit_keypoints(num_key_points, obj_poses, obj_orns, tip_poses, hand_model="allegro", object_creator=None):
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

    key_joint_states, key_q_sols, desired_positions = kin.solve_ik_keypoints(key_tip_poses, 
                                                                             key_obj_poses, 
                                                                             key_obj_orns, 
                                                                             hand_model=hand_model,
                                                                             object_creator=object_creator)
    base_state = [np.hstack([joint_state[0], joint_state[1]]) for joint_state in key_joint_states]
    joints_state = [joint_state[2] for joint_state in key_joint_states]
    return (np.asarray(base_state), np.asarray(joints_state)), key_obj_poses, key_obj_orns, key_q_sols, key_idx, desired_positions

def fit_interp(num_interp, obj_poses, obj_orns, tip_poses, q_start, q_end, hand_model, object_creator=None):
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

    interp_joint_state = kin.solve_interpolation(interp_tip_poses, 
                                                 interp_obj_poses, 
                                                 interp_obj_orns, 
                                                 q_start, q_end, 
                                                 hand_model=hand_model,
                                                 object_creator=object_creator)
    
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
    parser.add_argument("--env", type=str, default="laptop")
    parser.add_argument("--extrapolate_frames", type=int, default=50)
    parser.add_argument("--hand_model", type=str, default="shadow")
    parser.add_argument("--has_floor", action="store_true", default=False)
    parser.add_argument("--target_offset",type=list, default=[0,0,0.3])
    parser.add_argument("--disable_hand",action="store_true", default=False)
    args = parser.parse_args()

    # Create Pybullet Scenario
    p.connect(p.GUI)
    if args.hand_model == "allegro"  and not args.disable_hand:
        hand_id = p.loadURDF("model/resources/allegro_hand_description/urdf/allegro_hand_description_right.urdf")
        alpha=0.8
    elif args.hand_model == "shadow" and not args.disable_hand:
        hand_id = p.loadURDF("model/resources/shadow_hand_description/urdf/shadow_hand_collision.urdf")
        alpha=0.8
    elif args.disable_hand:
        hand_id = -1
        alpha = 0.5
    o_id = setup_pybullet.pybullet_creator[args.env](p, alpha)
    object_creator = partial(creator.object_creators[args.env],has_floor=args.has_floor)
    # Create Visualization for finger tip
    colors = [(0.9, 0.9, 0.9, 0.7),
              (0.9, 0.0, 0.0, 0.7),
              (0.0, 0.9, 0.0, 0.7),
              (0.0, 0.0, 0.9, 0.7),
              (0.0, 0.0, 0.0, 0.7)]
    tip_ids = []
    for i in range(4 if args.hand_model =="allegro" else 5):
        tip_ids.append(rb.create_primitive_shape(p, 0.1, p.GEOM_SPHERE,
                                                 (0.02,), color=colors[i], 
                                                 collidable=False, init_xyz = (100,100,100)))

    obj_poses, obj_orns, finger_tip_poses = load_trajector_data(args.exp_name, args.extrapolate_frames)

    if args.mode == "keypoints":
        key_states, key_obj_poses, key_obj_orns, key_q_sols, key_idx, des_poses = fit_keypoints(NUM_KEY_FRAMES, 
                                                                                                obj_poses, 
                                                                                                obj_orns, 
                                                                                                finger_tip_poses, 
                                                                                                hand_model=args.hand_model,
                                                                                                object_creator=object_creator)
        helper.animate_keyframe(o_id, hand_id, key_obj_poses, key_obj_orns, key_states[1], key_states[0], tip_ids=tip_ids, desired_positions=des_poses)
        np.savez(f"data/ik/{args.exp_name}_keyframes.npz", q_sols=key_q_sols, key_idx=key_idx, key_obj_poses=key_obj_poses, key_obj_orns=key_obj_orns, 
                                                           key_base_states=key_states[0], key_joint_states=key_states[1], desired_positions=des_poses)
    elif args.mode == "animate_keyframes":
        data = np.load(f"data/ik/{args.exp_name}_keyframes.npz")
        helper.animate_keyframe(o_id, hand_id, data["key_obj_poses"], data["key_obj_orns"], data["key_joint_states"], data["key_base_states"], tip_ids=tip_ids, desired_positions=data["desired_positions"])
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
        interp_state = fit_interp(NUM_INTERP, 
                                  segment_obj_poses, 
                                  segment_obj_orns, 
                                  segment_tip_poses, 
                                  q_start, q_end,
                                  hand_model=args.hand_model,
                                  object_creator=object_creator)
        desired_positions = [shadow_rbm.get_desired_position(tip_poses, model_param.object_padding, True) for tip_poses in segment_tip_poses]
        desired_positions_np = np.ones((len(desired_positions),5,3)) * 100
        for i in range(len(desired_positions)):
            finger_map = ALLEGRO_FINGER_MAP if args.hand_model == "allegro" else SHADOW_FINGER_MAP
            for j,finger in enumerate(finger_map):
                desired_positions_np[i,j] = desired_positions[i][finger]
        helper.animate(o_id, hand_id, segment_obj_poses, segment_obj_orns, 
                       interp_state[1], base_states=interp_state[0],
                       desired_positions=desired_positions_np, tip_ids=tip_ids)
        np.savez(f"data/ik/{args.exp_name}_segment{idx}.npz", joint_states=interp_state[1], base_states=interp_state[0])
    elif args.mode == "final":
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
        final_base_state = base_states[-1] # 7 dofs
        final_obj_pose = obj_poses[-1]
        final_obj_quat = obj_orns[-1]
        final_tip_poses = finger_tip_poses[-1]
        fixed_joint_state = joint_states[-1]
        target_offset = np.array(args.target_offset)
        target_base_state = final_base_state.copy()
        target_base_state[:3] += target_offset
        target_obj_pose = final_obj_pose + target_offset[:3]
        target_tip_poses = final_tip_poses + np.hstack([target_offset[:3],np.zeros(3)])
        target_obj_quat = obj_orns[-1]
        base_pos_traj = np.linspace(final_base_state[:3], target_base_state[:3], args.extrapolate_frames)
        base_orn_traj = helper.interp_pybullet_quaternion(final_base_state[3:], target_base_state[3:], args.extrapolate_frames)
        base_traj = np.hstack([base_pos_traj,base_orn_traj])
        obj_pos_traj = np.linspace(final_obj_pose, target_obj_pose, args.extrapolate_frames)
        obj_orn_traj = helper.interp_pybullet_quaternion(final_obj_quat, target_obj_quat, args.extrapolate_frames)
        tip_poses = np.linspace(final_tip_poses, target_tip_poses, args.extrapolate_frames)
        desired_positions = [shadow_rbm.get_desired_position(tip_poses, model_param.object_padding, True) for tip_poses in tip_poses]
        desired_positions_np = np.ones((len(desired_positions),5,3)) * 100
        for i in range(len(desired_positions)):
            finger_map = ALLEGRO_FINGER_MAP if args.hand_model == "allegro" else SHADOW_FINGER_MAP
            for j,finger in enumerate(finger_map):
                desired_positions_np[i,j] = desired_positions[i][finger]
        helper.animate_base(o_id, hand_id, obj_pos_traj, obj_orn_traj, base_traj, fixed_joint_state, desired_positions=desired_positions_np,tip_ids=tip_ids)
        np.savez(f"data/ik/{args.exp_name}_final.npz", 
                 obj_pos_traj=obj_pos_traj, 
                 obj_orn_traj=obj_orn_traj, 
                 base_traj = base_traj,
                 fixed_joint_state=fixed_joint_state,
                 desired_positions=desired_positions_np)
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
        desired_positions = [shadow_rbm.get_desired_position(tip_poses, model_param.object_padding, True) for tip_poses in finger_tip_poses]
        desired_positions_np = np.ones((len(desired_positions),5,3)) * 100
        for i in range(len(desired_positions)):
            finger_map = ALLEGRO_FINGER_MAP if args.hand_model == "allegro" else SHADOW_FINGER_MAP[:-1]
            for j,finger in enumerate(finger_map):
                desired_positions_np[i,j] = desired_positions[i][finger]
        helper.animate(o_id, hand_id, obj_poses[20:50*NUM_KEY_FRAMES], obj_orns[20:50*NUM_KEY_FRAMES], joint_states[20:50*NUM_KEY_FRAMES], 
                       base_states[20:50*NUM_KEY_FRAMES], desired_positions=desired_positions_np[20:50*NUM_KEY_FRAMES],tip_ids=tip_ids)
        final = np.load(f"data/ik/{args.exp_name}_final.npz")
        helper.animate_base(o_id, hand_id, final["obj_pos_traj"],final["obj_orn_traj"], final["base_traj"],final["fixed_joint_state"], desired_positions=final["desired_positions"],tip_ids=tip_ids)
        
        
if __name__ == "__main__":
    main()