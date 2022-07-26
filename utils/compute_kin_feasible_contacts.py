import pydrake.math

import model.param as model_param
import model.rigid_body_model_for_kin_feasible as rbm
import model.manipulation.scenario as scenario
import neurals.data_generation_config as dgc

import argparse
from functools import partial
import numpy as np
import multiprocessing
import os
import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.snopt import SnoptSolver
from pydrake.math import RigidTransform
from pydrake.common.eigen_geometry import Quaternion
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.geometry import Sphere
from scipy.spatial.transform import Rotation as scipy_rot
from itertools import permutations
import time

finger_permutations = list(permutations(rbm.ActiveAllegroHandFingers))
random_state = np.random.RandomState(0)

quat_inits = []
for yaw in np.linspace(0, 2*np.pi, dgc.ik_attempts, endpoint=False):
    quat_init = scipy_rot.from_euler(
        "xyz", [0., np.pi/2., yaw]).as_quat()[[3, 0, 1, 2]]
    quat_inits.append(quat_init)

def check_kin_feasible(contact_points, contact_normals, object_path=None, base_link_name="manipulated_object", has_tabletop=True):
    """
    If object_path == None, then by default use naive_box
    Assume contact points and contact normals are squeezed np.ndarray
    """
    object_world_pose = RigidTransform(p=np.array([0, 0, 0]))
    hand_plant = rbm.AllegroHandPlantDrake(object_path=object_path, object_base_link_name=base_link_name,
                                            object_world_pose=object_world_pose, meshcat_open_brower=False, meshcat=False,
                                            has_tabletop=has_tabletop)
    _, plant_context = hand_plant.create_context()

    ik, constraints_on_finger, collision_constr, desired_positions = hand_plant.construct_ik_given_fingertip_normals(
                    plant_context,
                    np.hstack([contact_points, contact_normals]),
                    padding=model_param.object_padding,
                    collision_distance=0.001,
                    allowed_deviation=np.ones(3)*0.010)
    # Try multiple solve to the problem
    is_good_grasp = False
    for quat_init in quat_inits:
        q_init = np.zeros_like(ik.q())
        # Set the hand to be hovering above the table with palm pointing down
        q_init[hand_plant.base_quaternion_idx_start:
                hand_plant.base_quaternion_idx_start+4] = quat_init
        q_init[hand_plant.base_position_idx_start +
                2] = dgc.hand_ik_initial_guess_height
        # Solve IK
        solver = SnoptSolver()
        try:
            result = solver.Solve(ik.prog(), q_init)
        except Exception as e:
            print(e)
            continue
        q_sol = result.GetSolution(ik.q())
        # Check if this is feasible
        # TODO(wualbert): better implementation
        # Check collision
        match_fingertip = True
        # Check all fingertips are close to the target
        for finger in rbm.ActiveAllegroHandFingers:
            if not constraints_on_finger[finger][0].evaluator().CheckSatisfied(q_sol, tol=1e-4):
                match_fingertip = False
                break
        no_collision = collision_constr.evaluator().CheckSatisfied(q_sol, tol=1e-2)
        # Debug
        if no_collision and match_fingertip:
            is_good_grasp = True
            break
    return is_good_grasp

if __name__ == "__main__":
    print("Reach Here!")
    parser = argparse.ArgumentParser(
        description="Find kinematically feasible grasps by solving IK")
    parser.add_argument('--drake_obj_name', type=str,
                        help='e.g. 006_mustard_bottle')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Index from which to compute. Only efffective for file start_pose_idx')
    parser.add_argument('--end_idx', type=float, default=np.inf,
                        help='Index from which to compute')
    parser.add_argument('--start_pose_idx', type=int, default=0,
                        help='Pose index from which to compute')
    parser.add_argument('--end_pose_idx', type=float, default=np.inf,
                        help='Pose index to which to compute')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the IK results')
    parser.add_argument('--override_existing', action='store_true',
                        help='override existing results', default=False)
    parser.add_argument('--compute_chunk', type=int, default=300,
                        help='how many configurations to store in each file')
    parser.add_argument('--thread_count', type=int, default=None,
                        help='how many threads to parallelize')
    args = parser.parse_args()
    if args.visualize:
        # Visualize segfaults with multiple thread
        assert args.thread_count == 1
    drake_obj_name = args.drake_obj_name  # "006_mustard_bottle" #"003_cracker_box"

    base_link_name = model_param.drake_ycb_objects[drake_obj_name]
    if drake_obj_name.startswith("0"):
        drake_path = os.path.join(
            model_param.drake_ycb_path, drake_obj_name+".sdf")
    else:
        drake_path = None
    start_idx = args.start_idx
    end_idx = args.end_idx
    # Load each of the files
    obj_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           dgc.dyn_feasible_points_cropped_path,
                           drake_obj_name)
    print("Try to load files from:",obj_dir)
    all_normals = []
    # TODO: Keep pose index only omit rigid body transform part
    all_pose_idxs = []
    for root, dirs, files in os.walk(obj_dir, topdown=False):
        for i,f in enumerate(files):
            if f.endswith(".npy"):
                file_normals = np.load(os.path.join(obj_dir, f))
                all_normals.append(file_normals)
                all_pose_idxs.append(i)
    print(f'Loaded {len(all_normals)} files')
    # Save the pose array
    output_path = '/'.join([os.path.dirname(os.path.abspath(__file__)), "..",
                            dgc.kin_feasible_configs_path])
    # Precompute the initial quaternions
    quat_inits = []
    for yaw in np.linspace(0, 2*np.pi, dgc.ik_attempts, endpoint=False):
        quat_init = scipy_rot.from_euler(
            "xyz", [0., np.pi/2., yaw]).as_quat()[[3, 0, 1, 2]]
        quat_inits.append(quat_init)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for idx, all_normals_in_file in enumerate(all_normals):
        pose_idx = all_pose_idxs[idx]
        if pose_idx < args.start_pose_idx or pose_idx >= args.end_pose_idx:
            continue
        elif pose_idx == args.start_pose_idx:
            start_idx = min(len(all_normals_in_file), start_idx)
            end_idx =   min(len(all_normals_in_file), end_idx)
        else:
            start_idx = 0
            end_idx = min(len(all_normals_in_file), end_idx)
        compute_chunk = args.compute_chunk
        object_world_pose = RigidTransform(p=np.array([0, 0, 0]))

        if args.thread_count is not None:
            thread_count = args.thread_count
        else:
            thread_count = multiprocessing.cpu_count()
        print(f'Solving with {thread_count} threads')
        hand_plants = [rbm.AllegroHandPlantDrake(object_path=drake_path, object_base_link_name=base_link_name,
                                                 object_world_pose=object_world_pose, meshcat_open_brower=False, meshcat=args.visualize,
                                                 has_tabletop=True)]
        hand_plants.extend([rbm.AllegroHandPlantDrake(object_path=drake_path, object_base_link_name=base_link_name,
                                                      object_world_pose=object_world_pose, meshcat_open_brower=False, meshcat=False,
                                                      has_tabletop=True)]*(thread_count-1))
        diagram_contexts = []
        plant_contexts = []
        for hp in hand_plants:
            diagram_context, plant_context = hp.create_context()
            diagram_contexts.append(diagram_context)
            plant_contexts.append(plant_context)

        def find_hand_contact_configs(contact_idx, all_normals_in_file,
                                      visualize=True):
            """
            :param contact:
            :param visualize:
            :param ik_attempts:
            :return:
            """
            contact = all_normals_in_file[contact_idx]
            thread_id = contact_idx % thread_count
            hand_plant = hand_plants[thread_id]
            diagram_context = diagram_contexts[thread_id]
            plant_context = plant_contexts[thread_id]
            # TODO: set object pose
            valid_configs = []
            total_evaluated = 0
            for finger_perm in finger_permutations:
                total_evaluated += 1
                fingertip_normals = {}
                for i in range(3):
                    assert len(contact[i, :]) == 6
                    fingertip_normals[finger_perm[i]] = contact[i, :]
                # Construct an IK problem
                ik, constraints_on_finger, collision_constr, desired_positions = hand_plant.construct_ik_given_fingertip_normals(
                    plant_context,
                    fingertip_normals,
                    padding=model_param.object_padding,
                    collision_distance=0.001,
                    allowed_deviation=np.ones(3)*0.010  # 0.012 is radius of the finger
                )
                # q_init = random_state.rand(
                #     len(ik.q())) - 0.5 # hand.get_drake_q_from_hand_config(np.array([0.,0.,-0.5]), np.array([0.,1.,0.,0.]), np.zeros(16), 'bullet')
                is_good_grasp = False
                for quat_init in quat_inits:
                    q_init = np.zeros_like(ik.q())
                    # Set the hand to be hovering above the table with palm pointing down
                    q_init[hand_plant.base_quaternion_idx_start:
                           hand_plant.base_quaternion_idx_start+4] = quat_init
                    q_init[hand_plant.base_position_idx_start +
                           2] = dgc.hand_ik_initial_guess_height
                    # Solve IK
                    solver = SnoptSolver()
                    try:
                        result = solver.Solve(ik.prog(), q_init)
                    except Exception as e:
                        print(e)
                        continue
                    q_sol = result.GetSolution(ik.q())
                    # Check if this is feasible
                    # TODO(wualbert): better implementation
                    # Check collision
                    match_fingertip = True
                    # Check all fingertips are close to the target
                    for finger in rbm.ActiveAllegroHandFingers:
                        if not constraints_on_finger[finger][0].evaluator().CheckSatisfied(q_sol, tol=1e-4):
                            match_fingertip = False
                            break
                    no_collision = collision_constr.evaluator().CheckSatisfied(q_sol, tol=1e-2)
                    # Debug
                    if no_collision and match_fingertip:
                        is_good_grasp = True
                        break
                    # elif no_collision:
                    #     q_init = q_sol + (random_state.random(q_sol.shape)-0.5)/0.5*0.03 # Perturb by
                    # else:
                    #     q_init = q_sol + (random_state.random(q_sol.shape)-0.5)/0.5*0.2 # Perturb by 0.1
                # FIXME(wualbert): collision is never satisfied. Why?
                hand_conf = hand_plant.convert_q_to_hand_configuration(q_sol)
                if visualize and thread_id == 0:
                    fingertip_positions = np.zeros([3, 3])
                    for fi, finger in enumerate(model_param.ActiveAllegroHandFingers):
                        fingertip_positions[fi,
                                            :] = fingertip_normals[finger][:3]
                    hand_plant.populate_q_with_viz_sphere_positions(fingertip_positions,
                                                                    q_sol)
                    hand_plant.plant.SetPositions(plant_context, q_sol)
                    hand_plant.diagram.Publish(diagram_context)
                    print(f"is_good_grasp: {is_good_grasp}, no_collision: {no_collision}, match_fingertip: {match_fingertip}")
                    if is_good_grasp:
                        time.sleep(10)
                    time.sleep(1)
                # Store the data
                if is_good_grasp:
                    valid_configs.append((fingertip_normals, hand_conf))
                print("Successful Solves")
            return valid_configs

        find_config_fn = partial(
            find_hand_contact_configs, visualize=args.visualize)

        for outer_idx in range(start_idx, end_idx, compute_chunk):
            upper_bound = min(len(all_normals_in_file),
                              outer_idx+compute_chunk)
            print(f'Solving idx {outer_idx}-{upper_bound}')
            save_path = os.path.join(
                output_path, drake_obj_name+f'_pose_{pose_idx}_{outer_idx}-{upper_bound}')
            if not args.override_existing and os.path.exists(save_path+'.npy'):
                print(save_path + " exists, skipping")
                continue
            contact_idxs = range(outer_idx, upper_bound)
            if thread_count > 1:
                with multiprocessing.Pool(thread_count) as p:
                    tmp_ans = p.map(
                        partial(find_config_fn, all_normals_in_file=all_normals_in_file), contact_idxs)
            else:
                tmp_ans = []
                for contact_idx in contact_idxs:
                    tmp_ans.append(find_config_fn(
                        contact_idx, all_normals_in_file))
            ans = []
            for a in tmp_ans:
                if len(a) > 0:
                    ans.extend(a)
            print(f'Found {len(ans)} valid configs')
            np.save(save_path, np.asanyarray(
                ans, dtype=object), allow_pickle=True)