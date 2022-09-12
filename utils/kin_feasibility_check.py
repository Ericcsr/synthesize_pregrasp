import os
from functools import partial
import copy
from multiprocessing import Pool
from re import M
import model.param as model_param
import model.allegro_hand_rigid_body_model as rbm
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
finger_permutations = list(permutations(rbm.ActiveAllegroHandFingers))
random_state = np.random.RandomState(0)
quat_inits = []
pose_inits = np.zeros((dgc.ik_attempts, 2))
for i,yaw in enumerate(np.linspace(0, 2*np.pi, dgc.ik_attempts, endpoint=False)):
    quat_init = scipy_rot.from_euler(
        "xyz", [0., np.pi/2., yaw]).as_quat()[[3, 0, 1, 2]]
    rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                           [np.sin(yaw),  np.cos(yaw)]])
    pose_inits[i] = rot_matrix @ np.array([-1.0, 0])
    quat_inits.append(quat_init)

FINGER_MAP = [model_param.AllegroHandFinger.THUMB, 
                  model_param.AllegroHandFinger.INDEX, 
                  model_param.AllegroHandFinger.MIDDLE, 
                  model_param.AllegroHandFinger.RING]

###################################################################################################
# TODO: Need to visualize kin feasibility check, still not reliable..

def check_kin_feasible(contact_points, contact_normals, object_path=None, object_creator=None, base_link_name="manipulated_object", bounding_box=None):
    """
    If object_path == None, then by default use naive_box
    Assume contact points and contact normals are squeezed np.ndarray
    Assume Object is always placed at canonical pose
    """
    object_world_pose = RigidTransform(p=np.array([0, 0, 0]))
    hand_plant = rbm.AllegroHandPlantDrake(object_path=object_path, 
                                           object_base_link_name=base_link_name,
                                           object_world_pose=object_world_pose,
                                           object_creator=object_creator,
                                           threshold_distance=0.1)
    

    
    ik, constraints_on_finger, collision_constr, _ = hand_plant.construct_ik_given_fingertip_normals(
                        np.hstack([contact_points, contact_normals]),
                        padding=model_param.object_padding,
                        collision_distance=1e-5,
                        allowed_deviation=np.ones(3)*0.02)
    if not (bounding_box is None):
        base_constr = ik.AddPositionConstraint(hand_plant.hand_drake_frame,
                                               np.zeros(3),
                                               bounding_box["upper"],
                                               bounding_box["lower"])
    
    # Try multiple solve to the problem
    for i, quat_init in enumerate(quat_inits):
        q_init = np.zeros_like(ik.q())
        # Set the hand to be hovering above the table with palm pointing down
        q_init[hand_plant.base_quaternion_idx_start:
                hand_plant.base_quaternion_idx_start+4] = quat_init
        q_init[hand_plant.base_position_idx_start +
                2] = dgc.hand_ik_initial_guess_height
        # q_init[hand_plant.base_position_idx_start:
        #         hand_plant.base_position_idx_start+2] = pose_inits[i]
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
        for finger in rbm.ActiveAllegroHandFingers:
            if not constraints_on_finger[finger][0].evaluator().CheckSatisfied(q_sol, tol=1e-2):
                match_fingertip = False
                break
        no_collision = collision_constr.evaluator().CheckSatisfied(q_sol, tol=1e-2)
        
        base_condition = True
        if not (bounding_box is None):
            base_condition = base_constr.evaluator().CheckSatisfied(q_sol, tol=1e-2)
        if no_collision and match_fingertip and base_condition:
            return True, hand_plant.get_bullet_hand_config_from_drake_q(q_sol)
    return False, hand_plant.get_bullet_hand_config_from_drake_q(q_sol)

def check_kin_feasible_parallel(contact_points, contact_normals, object_path=None, object_creator=None, base_link_name="manipulated_object", bounding_box=None, num_process=16):
    kwargs = {
        "contact_points":contact_points,
        "contact_normals":contact_normals,
        "object_path":object_path,
        "object_creator":object_creator,
        "base_link_name":base_link_name,
        "bounding_box":bounding_box
    }
    with Pool(num_process) as proc:
        results = proc.starmap(partial(check_kin_feasible, **kwargs), [() for _ in range(num_process)])
    
    for result in results:
        if result[0]:
            return result[0], result[1]
    return result[0], result[1] # Last result, should be false at this time

# Some Repeatitive implement dedicated for solving IK

def solve_ik(contact_points, contact_normals, object_path=None, object_creator=None, 
             object_pose=None, bounding_box=None, q_init_guess=None, ref_q=None):
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
    hand_plant = rbm.AllegroHandPlantDrake(object_path=object_path, 
                                           object_base_link_name="manipulated_object",
                                           object_world_pose=object_world_pose,
                                           object_creator=object_creator,
                                           threshold_distance=1.)
    

    tip_pose_normal = np.hstack([contact_points, contact_normals])
    ik, constraints_on_finger, collision_constr, desired_positions = hand_plant.construct_ik_given_fingertip_normals(
                    tip_pose_normal,
                    padding=model_param.object_padding,
                    collision_distance=1e-5,
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
        r = (ub-lb) * model_param.ALLOWANCE
        prog.AddBoundingBoxConstraint(ref_q -r , ref_q + r, ik.q())
    # Prepare reference data for each IK Solve
    ik_attempts = 10
    quat_inits = []
    pose_inits = np.zeros((ik_attempts, 2))
    for i,yaw in enumerate(np.linspace(0, 2*np.pi, ik_attempts, endpoint=False)):
        quat_init = scipy_rot.from_euler(
            "xyz", [0., np.pi/2., yaw]).as_quat()[[3, 0, 1, 2]]
        rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                            [np.sin(yaw),  np.cos(yaw)]])
        pose_inits[i] = rot_matrix @ np.array([-1.0, 0])
        quat_inits.append(quat_init)
    
    # Try multiple solve to the problem
    q_init = copy.deepcopy(q_init_guess) 
    for i in range(ik_attempts):
        if not (q_init_guess is None):
            pass
        else:
            q_init = np.zeros_like(ik.q())
            # Set the hand to be hovering above the table with palm pointing down
            q_init[hand_plant.base_quaternion_idx_start:
                    hand_plant.base_quaternion_idx_start+4] = quat_inits[i]
            q_init[hand_plant.base_position_idx_start +
                    2] = dgc.hand_ik_initial_guess_height
            q_init[hand_plant.base_position_idx_start:
                    hand_plant.base_position_idx_start+2] = pose_inits[i]
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
        for finger in rbm.ActiveAllegroHandFingers:
            if isinstance(constraints_on_finger[finger], list):
                if not constraints_on_finger[finger][0].evaluator().CheckSatisfied(q_sol, tol=1e-2):
                    match_fingertip = False
                    break
        no_collision = collision_constr.evaluator().CheckSatisfied(q_sol, tol=1e-2)
        
        base_condition = True
        if not (bounding_box is None):
            base_condition = base_constr.evaluator().CheckSatisfied(q_sol, tol=1e-2)
        if no_collision and match_fingertip and base_condition:
            return (True, True), hand_plant.get_bullet_hand_config_from_drake_q(q_sol), q_sol, desired_positions
        elif no_collision and not match_fingertip and not(q_init_guess is None):
            q_init =  q_sol + (np.random.random(q_sol.shape) - 0.5) / 0.5 * 0.03
        elif not(q_init_guess is None):
            q_init = q_sol + (np.random.random(q_sol.shape) - 0.5) / 0.5 * 0.2
    return (no_collision, match_fingertip), hand_plant.get_bullet_hand_config_from_drake_q(q_sol), q_sol, desired_positions

def solve_ik_parallel(contact_points, contact_normals, object_path=None, object_creator=None, 
                      object_pose=None, bounding_box=None, q_init_guess=None, ref_q=None, num_process=16):
    kwargs = {
        "contact_points":contact_points,
        "contact_normals":contact_normals,
        "object_path":object_path,
        "object_creator":object_creator,
        "object_pose":object_pose,
        "bounding_box":bounding_box,
        "q_init_guess":q_init_guess,
        "ref_q":ref_q
    }
    arg_lists = [()] * num_process
    with Pool(num_process) as proc:
        results = proc.starmap(partial(solve_ik, **kwargs), arg_lists)
    return results

def solve_ik_keypoints(targets, obj_poses, obj_orns, object_path=None, object_creator=None):
    # Here we need to assume targets point are projected poses expressed in local frame.
    joint_states = []
    success_flags = []
    q_sols = []
    desired_positions = []
    for i in range(len(targets)):
        obj_pose = np.hstack([obj_poses[i],obj_orns[i]])
        #tip_pose, tip_normals = helper.express_tips_world_frame(targets[i,:,:3], targets[i,:,3:], obj_pose)
        tip_pose, tip_normals = targets[i,:,:3], targets[i,:,3:]
        if i == 0:
            results = solve_ik_parallel(tip_pose, tip_normals, object_pose=obj_pose, 
                                        object_path=object_path, object_creator=object_creator)
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
            results = solve_ik_parallel(tip_pose, tip_normals, q_init_guess=q_sols[-1], object_pose=obj_pose,
                                        object_path=object_path, object_creator=object_creator)
            min_dist = np.inf
            min_dist_q = results[0][2]
            min_dist_joint = results[0][1]
            for result in results:
                if result[0][0] and result[0][1] and np.linalg.norm(result[2]-q_sols[-1]) < min_dist:
                    min_dist = np.linalg.norm(result[2]-q_sols[-1])
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
    desired_positions_np = np.ones((len(desired_positions),4,3)) * 100
    for i in range(len(desired_positions)):
        for j,finger in enumerate(FINGER_MAP):
            desired_positions_np[i,j] = desired_positions[i][finger]
    return joint_states, q_sols, desired_positions_np

# Need each solve need to be regularized by the q_interp
def solve_interpolation(targets, obj_poses, obj_orns, q_start, q_end,
                        object_path=None, object_creator=None):
    joint_states = []
    success_flags = []
    # if quaternion is included in q_start and end interpolation may be problematic..
    dummy_plant = rbm.AllegroHandPlantDrake(object_path=None, 
                                                 object_base_link_name="manipulated_object",
                                                 object_world_pose=RigidTransform(p=[0,0,0]),
                                                 threshold_distance=0.02)
    # ======== Interpolate q_start to q_end ========
    base_position_start, base_quaternion_start, finger_angles_start = dummy_plant.convert_q_to_hand_configuration(q_start)
    base_position_end, base_quaternion_end, finger_angles_end = dummy_plant.convert_q_to_hand_configuration(q_end)
    finger_angles_start_np = np.zeros((4, len(finger_angles_start[FINGER_MAP[0]])))
    finger_angles_end_np = np.zeros((4, len(finger_angles_start[FINGER_MAP[0]])))
    for i,finger in enumerate(FINGER_MAP):
        finger_angles_start_np[i] = finger_angles_start[finger]
        finger_angles_end_np[i] = finger_angles_end[finger]
    base_position_interp = np.linspace(base_position_start, base_position_end, len(targets)+2)
    base_quaternion_interp = helper.interp_pybullet_quaternion(base_quaternion_start, base_quaternion_end, len(targets)+2)
    finger_angle_interp = np.linspace(finger_angles_start_np, finger_angles_end_np, len(targets)+2)
    q_interp = np.zeros((len(targets)+2, len(q_start)))
    for i in range(len(targets)+2):
        finger_angle_intp = {}
        for j,finger in enumerate(FINGER_MAP):
            finger_angle_intp[finger] = finger_angle_interp[i,j]
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
                                    object_path=object_path, object_creator=object_creator)
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




