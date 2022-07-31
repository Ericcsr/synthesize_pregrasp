import model.param as model_param
import model.rigid_body_model_for_kin_feasible as rbm
import model.manipulation.scenario as scenario
import neurals.data_generation_config as dgc
import numpy as np
from pydrake.solvers.snopt import SnoptSolver
from pydrake.math import RigidTransform
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

###################################################################################################

def check_kin_feasible(contact_points, contact_normals, object_path=None, base_link_name="manipulated_object", bounding_box=None):
    """
    If object_path == None, then by default use naive_box
    Assume contact points and contact normals are squeezed np.ndarray
    Assume Object is always placed at canonical pose
    """
    object_world_pose = RigidTransform(p=np.array([0, 0, 0]))
    hand_plant = rbm.AllegroHandPlantDrake(object_path=object_path, 
                                           object_base_link_name=base_link_name,
                                           object_world_pose=object_world_pose,
                                           threshold_distance=0.1)

    if bounding_box is None:
        ik, constraints_on_finger, collision_constr, desired_positions = hand_plant.construct_ik_given_fingertip_normals(
                        np.hstack([contact_points, contact_normals]),
                        padding=model_param.object_padding,
                        collision_distance=1e-4,
                        allowed_deviation=np.ones(3)*0.01,
                        bounding_box=bounding_box)
    else:
        ik, constraints_on_finger, collision_constr, base_constr, desired_positions = hand_plant.construct_ik_given_fingertip_normals(
                        np.hstack([contact_points, contact_normals]),
                        padding=model_param.object_padding,
                        collision_distance=1e-4,
                        allowed_deviation=np.ones(3)*0.01,
                        bounding_box=bounding_box)
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
            if not constraints_on_finger[finger][0].evaluator().CheckSatisfied(q_sol, tol=1e-4):
                match_fingertip = False
                break
        no_collision = collision_constr.evaluator().CheckSatisfied(q_sol, tol=1e-2)
        
        base_condition = True
        if not (bounding_box is None):
            base_condition = base_constr.evaluator().CheckSatisfied(q_sol, tol=1e-2)
        # Debug
        print(f"Collision: {no_collision} Target: {match_fingertip} Base: {base_condition}")
        if no_collision and match_fingertip and base_condition:
            return True, hand_plant.get_bullet_hand_config_from_drake_q(q_sol)
    return False, hand_plant.get_bullet_hand_config_from_drake_q(q_sol)
