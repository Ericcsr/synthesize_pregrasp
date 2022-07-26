import open3d as o3d

import pydrake.math

import model.param as model_param
import model.rigid_body_model as rbm
import model.manipulation.scenario as scenario
import data.data_generation_config as dgc
import utils.math_utils as math_utils

import argparse
from functools import partial
import numpy as np
import multiprocessing
import os
import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.snopt import SnoptSolver
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.geometry import Sphere
from itertools import permutations
import time
from scipy.spatial.transform import Rotation as scipy_rot

import matplotlib.pyplot as plt

# random_state = np.random.RandomState(0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find kinematically feasible grasps by solving IK")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--skip_pruning", action="store_true", default=False)
    # parser.add_argument('--drake_obj_name', type=str,
    #                     help='e.g. 006_mustard_bottle', default="010_potted_meat_can")
    args = parser.parse_args()
    drake_obj_names = list(model_param.drake_ycb_objects.keys()) #args.drake_obj_name#"006_mustard_bottle" #"003_cracker_box"

    fib_points = math_utils.fibonacci_sphere(samples=dgc.num_rest_pose_initial_orn)


    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(fib_points[:,0], fib_points[:,1], fib_points[:,2])

    # plt.show()

    for drake_obj_name in drake_obj_names:
        # Load the object
        base_link_name = model_param.drake_ycb_objects[drake_obj_name]
        drake_path = os.path.join(model_param.drake_ycb_path, drake_obj_name+".sdf")
        p_WB = np.array([0.,0.,10.])
        r_WB = np.zeros(3)
        object_world_pose = RigidTransform(RollPitchYaw(*r_WB), p_WB)
        object_plant = rbm.ObjectTabletopPlantDrake(object_path=drake_path, object_base_link_name=base_link_name, 
                        object_world_pose=object_world_pose, meshcat_open_brower=False, meshcat=args.visualize,
                        num_viz_spheres=0, viz_sphere_radius=1e-3)
        obj_dir = os.path.join(pydrake.getDrakePath(), "manipulation/models/ycb/meshes")
        obj_file = os.path.join(obj_dir, drake_obj_name+"_textured.obj")

        # Output path
        output_path = '/'.join([os.path.dirname(os.path.abspath(__file__)),'..',
                        dgc.rest_pose_path])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('Storing output to ', output_path)
        # Simulate
        # Note that this is stored in Drake quaternion convention!
        # i.e. [w, x, y, z]
        result = []
        assert (object_plant.plant.num_positions() == 7)
        for idx, orn_i in enumerate(fib_points):
            # Visualize the drake object
            q = np.zeros(7)
            q[6] = dgc.rest_pose_drop_height
            q[1:4] = orn_i
            q[0] = 0.
            q[:4] /= np.linalg.norm(q[:4])
            if args.visualize:
                q_ans = object_plant.simulate_forward(q, realtime_rate=0.3)
            else:
                q_ans = object_plant.simulate_forward(q, realtime_rate=0.)
            q_ans[:4] /= np.linalg.norm(q_ans[:4])
            # Note that q_ans is stored in Drake quaternion convention!
            # i.e. [q_w, q_x, q_y, q_z, p_x, p_y, p_z]
            is_yaw_transformed = False
            if not args.skip_pruning:
                for q_existing in result:
                    is_yaw_transformed = math_utils.is_yaw_transformed_quat(
                        q_ans[:4], q_existing[:4]
                    )
                    if is_yaw_transformed:
                        break
            if not is_yaw_transformed:
                result.append(q_ans)
        np.save(os.path.join(output_path, drake_obj_name), np.asanyarray(result), allow_pickle=True)
        print(f'Produced {len(result)} rest poses for ', drake_obj_name)
