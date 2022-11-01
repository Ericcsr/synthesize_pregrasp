import os
from functools import partial
import itertools
import copy
from multiprocessing import Pool
import model.param as model_param
import model.allegro_hand_rigid_body_model as allegro_rbm
import model.shadow_hand_rigid_body_model as shadow_rbm
import model.manipulation_obj_creator as creator
import neurals.data_generation_config as dgc
import utils.helper as helper 
import numpy as np
from pydrake.solvers.snopt import SnoptSolver
from pydrake.math import RigidTransform
from pydrake.common import eigen_geometry
from scipy.spatial.transform import Rotation as scipy_rot
from itertools import permutations

################################ Initialize Global wise data ######################################
allegro_finger_permutations = list(permutations(allegro_rbm.ActiveAllegroHandFingers))
shadow_finger_permutations = list(permutations(shadow_rbm.ActiveShadowHandFingers))
random_state = np.random.RandomState(0)

ik_attempts = 10
num_process = 16
quat_inits = np.zeros((num_process, ik_attempts, 4))
pose_inits = np.zeros((num_process, ik_attempts, 2))
for j, yaw in enumerate(np.linspace(0, 2*np.pi, num_process, endpoint=False)):
    for i,rol in enumerate(np.linspace(0, 2*np.pi, ik_attempts, endpoint=False)):
        quat_init = scipy_rot.from_euler(
            "xyz", [rol, -np.pi/2, yaw]).as_quat()[[3, 0, 1, 2]]
        rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                            [np.sin(yaw),  np.cos(yaw)]])
        pose_inits[j, i] = rot_matrix @ np.array([-0.3, 0])
        quat_inits[j, i] = quat_init

ALLEGRO_FINGER_MAP = [model_param.AllegroHandFinger.THUMB, 
                  model_param.AllegroHandFinger.INDEX, 
                  model_param.AllegroHandFinger.MIDDLE, 
                  model_param.AllegroHandFinger.RING]

SHADOW_FINGER_MAP = [model_param.ShadowHandFinger.THUMB,
                     model_param.ShadowHandFinger.INDEX,
                     model_param.ShadowHandFinger.MIDDLE,
                     model_param.ShadowHandFinger.RING,
                     model_param.ShadowHandFinger.LITTLE]

###################################################################################################
# TODO: Need to visualize kin feasibility check, still not reliable..

def check_kin_feasible(contact_points, 
                       contact_normals, 
                       object_path=None, 
                       object_creator=None, 
                       base_link_name="manipulated_object", 
                       bounding_box=None,
                       hand_model = "allegro",
                       pid=0):
    """
    If object_path == None, then by default use naive_box
    Assume contact points and contact normals are squeezed np.ndarray
    Assume Object is always placed at canonical pose
    """
    object_world_pose = RigidTransform(p=np.array([0, 0, 0]))
    if hand_model == "allegro":
        hand_plant = allegro_rbm.AllegroHandPlantDrake(object_path=object_path, 
                                               object_base_link_name=base_link_name,
                                               object_world_pose=object_world_pose,
                                               object_creator=object_creator,
                                               threshold_distance=1.)
        active_fingers = allegro_rbm.ActiveAllegroHandFingers
    elif hand_model == "shadow":
        hand_plant = shadow_rbm.ShadowHandPlantDrake(object_path=object_path,
                                                     object_base_link_name = base_link_name,
                                                     object_world_pose = object_world_pose,
                                                     object_creator=object_creator,
                                                     threshold_distance=1.)
        active_fingers = shadow_rbm.ActiveShadowHandFingers
    else:
        raise NotImplementedError

    
    ik, constraints_on_finger, collision_constr, _ = hand_plant.construct_ik_given_fingertip_normals(
                        np.hstack([contact_points, contact_normals]),
                        padding=model_param.object_padding,
                        collision_distance=1e-4,
                        allowed_deviation=np.ones(3)*0.005)
    if not (bounding_box is None):
        base_constr = ik.AddPositionConstraint(hand_plant.hand_drake_frame,
                                               np.zeros(3),
                                               bounding_box["upper"],
                                               bounding_box["lower"])
    
    # Try multiple solve to the problem
    for i in range(ik_attempts):
        q_init = np.zeros_like(ik.q())
        # Set the hand to be hovering above the table with palm pointing down
        q_init[hand_plant.base_quaternion_idx_start:
                hand_plant.base_quaternion_idx_start+4] = quat_inits[pid,i]
        q_init[hand_plant.base_position_idx_start +
                2] = dgc.hand_ik_initial_guess_height
        q_init[hand_plant.base_position_idx_start:
                hand_plant.base_position_idx_start+2] = pose_inits[pid,i]
        q_init += (np.random.random(q_init.shape) - 0.5) / 0.5 * 0.03
        # Solve IK
        solver = SnoptSolver()
        try:
            result = solver.Solve(ik.prog(), q_init)
        except Exception as e:
            print(e)
            continue
        q_sol = result.GetSolution(ik.q())
        # Check collision
        match_fingertip = True
        # Check all fingertips are close to the target
        for finger in active_fingers:
            if isinstance(constraints_on_finger[finger], list):
                if not constraints_on_finger[finger][0].evaluator().CheckSatisfied(q_sol, tol=1e-2):
                    match_fingertip = False
                    break
        no_collision = collision_constr.evaluator().CheckSatisfied(q_sol, tol=1.5e-2)
        
        base_condition = True
        if not (bounding_box is None):
            base_condition = base_constr.evaluator().CheckSatisfied(q_sol, tol=1e-2)
        if no_collision and match_fingertip and base_condition:
            return True, hand_plant.get_bullet_hand_config_from_drake_q(q_sol)
    return False, hand_plant.get_bullet_hand_config_from_drake_q(q_sol)

def check_kin_feasible_parallel(contact_points, 
                                contact_normals, 
                                object_path=None, 
                                object_creator=None, 
                                base_link_name="manipulated_object", 
                                bounding_box=None, 
                                num_process=16,
                                hand_model = "allegro"):
    kwargs = {
        "contact_points":contact_points,
        "contact_normals":contact_normals,
        "object_path":object_path,
        "object_creator":object_creator,
        "base_link_name":base_link_name,
        "bounding_box":bounding_box,
        "hand_model":hand_model
    }
    args = []
    for i in range(num_process):
        arg = list(kwargs.values())
        arg.append(i)
        args.append(tuple(arg))
    with Pool(num_process) as proc:
        results = proc.starmap(check_kin_feasible, args)
    
    for result in results:
        if result[0]:
            return result[0], result[1]
    return result[0], result[1] # Last result, should be false at this time

# Some Repeatitive implement dedicated for solving IK

def solve_ik(contact_points, contact_normals, object_path=None, object_creator=None, 
             object_pose=None, bounding_box=None, q_init_guess=None, ref_q=None, hand_model="allegro",pid=0):
    """
    If object_path == None, then by default use naive_box
    Assume contact points and contact normals are squeezed np.ndarray
    Assume Object is always placed at canonical pose
    """
    if object_pose is None:
        object_world_pose = RigidTransform(p=np.array([0, 0, 0]))
    else:
        orn = np.array([object_pose[6],object_pose[3],object_pose[4],object_pose[5]])
        object_world_pose = RigidTransform(quaternion=eigen_geometry.Quaternion(orn), p=object_pose[:3])
    if hand_model == "allegro":
        hand_plant = allegro_rbm.AllegroHandPlantDrake(object_path=object_path, 
                                               object_base_link_name="manipulated_object",
                                               object_world_pose=object_world_pose,
                                               object_creator=object_creator,
                                               threshold_distance=1.)
        active_fingers = allegro_rbm.ActiveAllegroHandFingers
    elif hand_model == "shadow":
        hand_plant = shadow_rbm.ShadowHandPlantDrake(object_path=object_path, 
                                               object_base_link_name="manipulated_object",
                                               object_world_pose=object_world_pose,
                                               object_creator=object_creator,
                                               threshold_distance=1.)
        active_fingers = shadow_rbm.ActiveShadowHandFingers
    else:
        raise NotImplementedError
    

    tip_pose_normal = np.hstack([contact_points, contact_normals])
    ik, constraints_on_finger, collision_constr, desired_positions = hand_plant.construct_ik_given_fingertip_normals(
                    tip_pose_normal,
                    padding=model_param.object_padding,
                    collision_distance=1e-4,
                    allowed_deviation=np.ones(3)*0.005,
                    straight_unused_fingers=False if not (ref_q is None) else True)

    # Add Extra constraints related to IK Solving
    if not (bounding_box is None):
        base_constr = ik.AddPositionConstraint(hand_plant.hand_drake_frame,
                                               np.zeros(3),
                                               bounding_box["upper"],
                                               bounding_box["lower"])
    if not (ref_q is None):
        prog = ik.get_mutable_prog()
        lb, ub = hand_plant.get_joint_limits()
        lb[:7] = -model_param.ROOT_RANGE
        ub[:7] = model_param.ROOT_RANGE
        r = (ub-lb) * model_param.ALLOWANCE #TODO: Set bound for root joint!
        prog.AddBoundingBoxConstraint(ref_q -r , ref_q + r, ik.q())
    # Prepare reference data for each IK Solve

    
    # Try multiple solve to the problem
    q_init = copy.deepcopy(q_init_guess)
    flags = []
    joint_states = []
    q_sols = []
    desired_positions_list = []
    for i in range(ik_attempts):
        if q_init_guess is None:
            q_init = np.zeros_like(ik.q())
            # Set the hand to be hovering above the table with palm pointing down
            q_init[hand_plant.base_quaternion_idx_start:
                    hand_plant.base_quaternion_idx_start+4] = quat_inits[pid,i]
            q_init[hand_plant.base_position_idx_start +
                    2] = dgc.hand_ik_initial_guess_height
            q_init[hand_plant.base_position_idx_start:
                    hand_plant.base_position_idx_start+2] = pose_inits[pid,i]
            q_init += (np.random.random(q_init.shape) - 0.5) / 0.5 * 0.03
        # Solve IK
        solver = SnoptSolver()
        try:
            result = solver.Solve(ik.prog(), q_init)
        except Exception as e:
            print(e)
            continue
        q_sol = result.GetSolution(ik.q())
        # Check collision
        match_fingertip = True
        # Check all fingertips are close to the target
        for finger in active_fingers:
            if isinstance(constraints_on_finger[finger], list):
                if not constraints_on_finger[finger][0].evaluator().CheckSatisfied(q_sol, tol=1e-2):
                    match_fingertip = False
                    break
        no_collision = collision_constr.evaluator().CheckSatisfied(q_sol, tol=1.5e-2)
        
        base_condition = True
        if not (bounding_box is None):
            base_condition = base_constr.evaluator().CheckSatisfied(q_sol, tol=1e-2)
        if no_collision and match_fingertip and base_condition:
            flags.append((True, True)) 
            joint_states.append(hand_plant.get_bullet_hand_config_from_drake_q(q_sol)) 
            q_sols.append(q_sol)
            desired_positions_list.append(desired_positions)
        elif no_collision and not match_fingertip and q_init_guess is not None:
            q_init =  q_sol + (np.random.random(q_sol.shape) - 0.5) / 0.5 * 0.03
        elif q_init_guess is not None:
            q_init = q_sol + (np.random.random(q_sol.shape) - 0.5) / 0.5 * 0.2
    flags.append((no_collision, match_fingertip))
    joint_states.append(hand_plant.get_bullet_hand_config_from_drake_q(q_sol))
    q_sols.append(q_sol)
    desired_positions_list.append(desired_positions)
    return flags, joint_states, q_sols, desired_positions_list

def solve_ik_parallel(contact_points, contact_normals, object_path=None, object_creator=None, 
                      object_pose=None, bounding_box=None, q_init_guess=None, ref_q=None, num_process=16,
                      hand_model="allegro"):
    kwargs = {
        "contact_points":contact_points,
        "contact_normals":contact_normals,
        "object_path":object_path,
        "object_creator":object_creator,
        "object_pose":object_pose,
        "bounding_box":bounding_box,
        "q_init_guess":q_init_guess,
        "ref_q":ref_q,
        "hand_model":hand_model
    }
    args = []
    for i in range(num_process):
        arg = list(kwargs.values())
        arg.append(i)
        args.append(tuple(arg))
    with Pool(num_process) as proc:
        _results = proc.starmap(solve_ik, args)
    
    results = []
    for proc_results in _results:
        for result in zip(proc_results[0],proc_results[1],proc_results[2],proc_results[3]):
            results.append(result)
    return results

def solve_ik_keypoints(targets, obj_poses, obj_orns, object_path=None, object_creator=None, hand_model="allegro"):
    # Here we need to assume targets point are projected poses expressed in local frame.
    joint_states = []
    success_flags = []
    q_sols = []
    desired_positions = []
    for i in range(len(targets)-1,-1,-1):
        obj_pose = np.hstack([obj_poses[i],obj_orns[i]])
        #tip_pose, tip_normals = helper.express_tips_world_frame(targets[i,:,:3], targets[i,:,3:], obj_pose)
        tip_pose, tip_normals = targets[i,:,:3], targets[i,:,3:]
        if i == len(targets)-1:
            results = solve_ik_parallel(tip_pose, tip_normals, object_pose=obj_pose, 
                                        object_path=object_path, object_creator=object_creator, hand_model=hand_model)
            for result in results:
                print("Collision:",result[0][0],"Fingertip:",result[0][1])
                if result[0][0] and result[0][1]:
                    joint_states.append(result[1])
                    success_flags.append(True)
                    q_sols.append(result[2])
                    desired_positions.append(result[3])
                    break
            else:
                joint_states.append(results[-1][1])
                success_flags.append(False)
                q_sols.append(results[-1][2])
                desired_positions.append(results[-1][3])
        else:
            results = solve_ik_parallel(tip_pose, tip_normals, q_init_guess=None, object_pose=obj_pose,
                                        object_path=object_path, object_creator=object_creator, hand_model=hand_model)
            min_dist = np.inf
            min_dist_q = results[0][2]
            min_dist_joint = results[0][1]
            for result in results:
                print("Collision:",result[0][0],"Fingertip:",result[0][1])
                dist = helper.quaternion_distance(result[1][1], joint_states[-1][1]) + np.linalg.norm(result[1][0]-joint_states[-1][0])
                if result[0][0] and result[0][1] and dist < min_dist:
                    min_dist = dist
                    min_dist_q = result[2].copy()
                    min_dist_joint = copy.deepcopy(result[1])
            joint_states.append(min_dist_joint)
            q_sols.append(min_dist_q)
            if min_dist == np.inf:
                success_flags.append(False)
            else:
                success_flags.append(True)
            desired_positions.append(results[-1][3])
        print(f"Target: {i}, result: {success_flags[-1]}")
    desired_positions_np = np.ones((len(desired_positions),len(targets),3)) * 100
    for i in range(len(desired_positions)):
        finger_map = ALLEGRO_FINGER_MAP if hand_model == "allegro" else SHADOW_FINGER_MAP[:len(targets)]
        for j,finger in enumerate(finger_map):
            desired_positions_np[i,j] = desired_positions[i][finger]
    return list(reversed(joint_states)), list(reversed(q_sols)), desired_positions_np[::-1]

# Need each solve need to be regularized by the q_interp
def solve_interpolation(targets, obj_poses, obj_orns, q_start, q_end,
                        object_path=None, object_creator=None, hand_model="allegro"):
    joint_states = []
    success_flags = []
    # if quaternion is included in q_start and end interpolation may be problematic..
    if hand_model == "allegro":
        dummy_plant = allegro_rbm.AllegroHandPlantDrake(object_path=None, 
                                                     object_base_link_name="manipulated_object",
                                                     object_world_pose=RigidTransform(p=[0,0,0]),
                                                     threshold_distance=0.02)
        finger_map = ALLEGRO_FINGER_MAP
    elif hand_model == "shadow":
        dummy_plant = shadow_rbm.ShadowHandPlantDrake(object_path=None,
                                                      object_base_link_name="manipulated_object",
                                                      object_world_pose=RigidTransform(p=[0,0,0]),
                                                      threshold_distance=0.02)
        finger_map = SHADOW_FINGER_MAP
    else:
        raise NotImplementedError
    # ======== Interpolate q_start to q_end ========
    base_position_start, base_quaternion_start, finger_angles_start = dummy_plant.convert_q_to_hand_configuration(q_start)
    base_position_end, base_quaternion_end, finger_angles_end = dummy_plant.convert_q_to_hand_configuration(q_end)
    #finger_angles_start_np = np.zeros((4, len(finger_angles_start[finger_map[0]])))
    finger_angles_start_np = [np.zeros(len(finger_angles_start[finger_map[i]])) for i in range(len(finger_map))]
    #finger_angles_end_np = np.zeros((4, len(finger_angles_start[finger_map[0]])))
    finger_angles_end_np = [np.zeros(len(finger_angles_start[finger_map[i]])) for i in range(len(finger_map))]
    finger_angle_interp = []
    for i,finger in enumerate(finger_map):
        finger_angles_start_np[i] = finger_angles_start[finger]
        finger_angles_end_np[i] = finger_angles_end[finger]
        finger_angle_interp.append(np.linspace(finger_angles_start_np[i], finger_angles_end_np[i], len(targets)+2))
    base_position_interp = np.linspace(base_position_start, base_position_end, len(targets)+2)
    base_quaternion_interp = helper.interp_pybullet_quaternion(base_quaternion_start, base_quaternion_end, len(targets)+2)
    # Assemble the vector
    q_interp = np.zeros((len(targets)+2, len(q_start)))
    for i in range(len(targets)+2):
        finger_angle_intp = {}
        for j,finger in enumerate(finger_map):
            finger_angle_intp[finger] = finger_angle_interp[j][i]
        q_interp[i] = dummy_plant.convert_hand_configuration_to_q(base_position_interp[i], base_quaternion_interp[i], finger_angle_intp)
    q_interp = q_interp[1:-1]
    # ======== Interpolate q_start to q_end ========
    pb_states_start = dummy_plant.get_bullet_hand_config_from_drake_q(q_start)
    pb_states_end = dummy_plant.get_bullet_hand_config_from_drake_q(q_end)
    joint_states.append(pb_states_start)

    for i in range(len(targets)):
        obj_pose = np.hstack([obj_poses[i],obj_orns[i]])
        #tip_pose, tip_normals = helper.express_tips_world_frame(targets[i,:3], targets[i,3:], obj_pose)
        tip_pose, tip_normals = targets[i,:,:3], targets[i,:,3:]
        results = solve_ik_parallel(tip_pose, tip_normals, q_init_guess=q_interp[i], object_pose=obj_pose, ref_q=q_interp[i],
                                    object_path=object_path, object_creator=object_creator, hand_model=hand_model)
        min_dist = np.inf
        min_dist_joint = results[0][1]
        for result in results:
            if result[0][0] and result[0][1] and np.linalg.norm(result[2]-q_interp[i]) < min_dist:
                min_dist = np.linalg.norm(result[2]-q_interp[i])
                min_dist_joint = copy.deepcopy(result[1])
        joint_states.append(min_dist_joint)
        if min_dist == np.inf:
            success_flags.append(False)
        else:
            success_flags.append(True)
        print(f"Target: {i}, result: {success_flags[-1]}")
    joint_states.append(pb_states_end)
    return joint_states

def direct_interpolation(targets, obj_poses, obj_orns, q_start, q_end,
                        object_path=None, object_creator=None, hand_model="allegro"):
    joint_states = []
    success_flags = []
    # if quaternion is included in q_start and end interpolation may be problematic..
    if hand_model == "allegro":
        dummy_plant = allegro_rbm.AllegroHandPlantDrake(object_path=None, 
                                                     object_base_link_name="manipulated_object",
                                                     object_world_pose=RigidTransform(p=[0,0,0]),
                                                     threshold_distance=0.02)
        finger_map = ALLEGRO_FINGER_MAP
    elif hand_model == "shadow":
        dummy_plant = shadow_rbm.ShadowHandPlantDrake(object_path=None,
                                                      object_base_link_name="manipulated_object",
                                                      object_world_pose=RigidTransform(p=[0,0,0]),
                                                      threshold_distance=0.02)
        finger_map = SHADOW_FINGER_MAP
    else:
        raise NotImplementedError
    # ======== Interpolate q_start to q_end ========
    base_position_start, base_quaternion_start, finger_angles_start = dummy_plant.convert_q_to_hand_configuration(q_start)
    base_position_end, base_quaternion_end, finger_angles_end = dummy_plant.convert_q_to_hand_configuration(q_end)
    #finger_angles_start_np = np.zeros((4, len(finger_angles_start[finger_map[0]])))
    finger_angles_start_np = [np.zeros(len(finger_angles_start[finger_map[i]])) for i in range(len(finger_map))]
    #finger_angles_end_np = np.zeros((4, len(finger_angles_start[finger_map[0]])))
    finger_angles_end_np = [np.zeros(len(finger_angles_start[finger_map[i]])) for i in range(len(finger_map))]
    finger_angle_interp = []
    for i,finger in enumerate(finger_map):
        finger_angles_start_np[i] = finger_angles_start[finger]
        finger_angles_end_np[i] = finger_angles_end[finger]
        finger_angle_interp.append(np.linspace(finger_angles_start_np[i], finger_angles_end_np[i], len(targets)+2))
    base_position_interp = np.linspace(base_position_start, base_position_end, len(targets)+2)
    base_quaternion_interp = helper.interp_pybullet_quaternion(base_quaternion_start, base_quaternion_end, len(targets)+2)
    # Assemble the vector
    q_interp = np.zeros((len(targets)+2, len(q_start)))
    for i in range(len(targets)+2):
        finger_angle_intp = {}
        for j,finger in enumerate(finger_map):
            finger_angle_intp[finger] = finger_angle_interp[j][i]
        q_interp[i] = dummy_plant.convert_hand_configuration_to_q(base_position_interp[i], base_quaternion_interp[i], finger_angle_intp)
    q_interp = q_interp[1:-1]
    joint_states = []
    pb_states_start = dummy_plant.get_bullet_hand_config_from_drake_q(q_start)
    pb_states_end = dummy_plant.get_bullet_hand_config_from_drake_q(q_end)
    joint_states.append(pb_states_start)
    for i in range(len(targets)):
        joint_states.append(dummy_plant.get_bullet_hand_config_from_drake_q(q_interp[i]))
    joint_states.append(pb_states_end)
    return joint_states


