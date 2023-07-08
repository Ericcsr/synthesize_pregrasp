import os
from argparse import ArgumentParser
from functools import partial
import open3d as o3d
import numpy as np
import utils.helper as helper
from neurals.NPGraspNet import NPGraspNet
from neurals.test_options import TestOptions
from utils.dyn_feasibility_check import simple_check_dyn_feasible
from utils.kin_feasibility_check import check_kin_feasible_parallel
import model.manipulation_obj_creator as creator
from envs.scales import SCALES
def load_traj_data(exp_name,idx=99):
    tip_poses = np.load(f"data/tip_data/{exp_name}_tip_poses.npy")
    obj_poses = np.load(f"data/object_poses/{exp_name}_object_poses.npy")
    return tip_poses[idx], obj_poses[idx,:3], obj_poses[idx, 3:]

def parse_raw_data(tip_pose, obj_pos, obj_orn):
    """
    tip_pose: finger tip pose in world frame
    obj_pos: object_pose in world frame
    obj_orn: object_orientation in pybullet's convention
    return finger tip pose in object frame projected
    """
    new_tip_pose = []
    finger_idx = []
    for i in range(tip_pose.shape[0]):
        print(tip_pose[i].sum(),i)
        if tip_pose[i].sum() < 10:
            candidate_pos = tip_pose[i,:3] - obj_pos
            orn = helper.convert_quat_for_drake(obj_orn)
            candidate_pos = helper.apply_drake_q_rotation(orn, candidate_pos, invert=True)
            candidate_orn = helper.apply_drake_q_rotation(orn, tip_pose[i,3:], invert=True)
            new_tip_pose.append(np.hstack([candidate_pos, candidate_orn]))
            finger_idx.append(i)
    return np.asarray(new_tip_pose), finger_idx

def process_pcd(point_cloud_o3d):
    points = np.asarray(point_cloud_o3d.points)
    normals = np.asarray(point_cloud_o3d.normals)
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    return points, normals, kd_tree

def project_grasp(grasp, kd_tree, points, normals=None):
    new_grasp = np.zeros((grasp.shape[0], 3 if normals is None else 6))
    for i,p in enumerate(grasp):
        idx = kd_tree.search_knn_vector_3d(p, 1)[1]
        new_grasp[i,:3] = points[idx]
        if not (normals is None):
            new_grasp[i,3:] = normals[idx]
    return new_grasp

def grasp_to_world(grasp,object_pos, object_orn):
    grasp = grasp.copy()
    for tip_pose in grasp:
        orn = helper.convert_quat_for_drake(object_orn)
        tip_pose[:3] = helper.apply_drake_q_rotation(orn, tip_pose[:3]) + object_pos
        tip_pose[3:] = helper.apply_drake_q_rotation(orn, tip_pose[3:])
    return grasp

def visualize_finger_tip(tip_poses, radius=0.01):
    """
    tip_poses is (N, 6), N is number of fingers involved
    """
    vis_spheres = []
    colors = [[0.6, 0, 0],[0.0, 0.6, 0],[0.0, 0.0, 0.6],[0.6,0.6,0.6],[0.6,0,0.6]]
    for i,tip_pose in enumerate(tip_poses):
        if tip_pose[0]+tip_pose[1]+tip_pose[2] < 20:
            sp = o3d.geometry.TriangleMesh.create_sphere(radius)
            sp.paint_uniform_color(colors[i])
            sp.translate(tip_pose)
            vis_spheres.append(sp)
    return vis_spheres

def validate_path(exp_name, env, visualize=False):
    tip_pose, obj_pos, obj_orn = load_traj_data(exp_name)
    # In relative coordinate
    new_tip_pose, finger_idx = parse_raw_data(tip_pose, obj_pos, obj_orn)
    print("Finger index:",finger_idx)
    if len(finger_idx) == 1:
        finger_idx[0] = 1
    #print(new_tip_pose[:], finger_idx)
    # Load pointcloud should be in canonical pose
    pcd_o3d = o3d.io.read_point_cloud(f"data/output_pcds/{exp_name}_pcd.ply")
    # if len(pcd_o3d.points) > 1024:
    #     pcd_o3d = pcd_o3d.farthest_point_down_sample(1024)
    if visualize:
        o3d.visualization.draw_geometries([pcd_o3d])
    assert(pcd_o3d.has_normals())
    points, normals, kd_tree = process_pcd(pcd_o3d)

    proj_tip_pose = new_tip_pose #project_grasp(new_tip_pose, kd_tree, points, normals)

    # Create and prepare neural network
    
    all_fingers = {0,1,2,3,4}
    pred_fingers = list(all_fingers.difference(finger_idx))
    if len(pred_fingers) == 3:
        name = "full_two"
    elif len(pred_fingers)==4:
        name = "five_finger_one"
    else:
        return 0
    parser = TestOptions()
    args = parser.parse()
    opt = args.__dict__
    opt["force_skip_load"] = False
    opt["name"] = name
    net = NPGraspNet(opt, mode="decoder", device="cuda:0",extra_cond_fingers=finger_idx, pred_fingers=pred_fingers)
    #print("Cond:",proj_tip_pose[:,:3])
    grasps = net.pred_grasp(points, proj_tip_pose[:,:3], 20)
    proj_grasps = []
    for grasp in grasps:
        proj_g = project_grasp(grasp, kd_tree, points, normals)
        proj_grasp = [np.array([100., 100., 100., 0., 0., 0.])] * len(all_fingers)
        for index in all_fingers:
            proj_grasp[index] = proj_g[index]
        proj_grasp = np.vstack(proj_grasp)
        proj_grasps.append(proj_grasp)

    # Visualize each grasps and check feasibility
    # Need to assume object is placed at canonical coordinate or it will crash
    score = 0
    # Need to customize each environments
    object_creator = partial(creator.object_creators[env], scale=SCALES[env], obstacle_collidable=True, has_floor=True)
    for i,grasp in enumerate(proj_grasps):
        
        dyn_feasibility = simple_check_dyn_feasible(grasp[:,:3], grasp[:,3:])
        if dyn_feasibility:
            grasp_world = grasp_to_world(grasp, obj_pos, obj_orn)
            print(grasp, grasp_world)
            kin_feasibility = check_kin_feasible_parallel(grasp_world[:,:3], grasp_world[:,3:],object_pose=np.hstack([obj_pos,obj_orn]), 
                                                          hand_model="shadow", object_creator=object_creator)
            if kin_feasibility[0]:
                score += 1.0
        if visualize:
            print("grasp ID:",i)
            vis_tip = visualize_finger_tip(grasp[:,:3])
            o3d.visualization.draw_geometries([pcd_o3d]+vis_tip)
    print("Success rate:", score / len(proj_grasps))
    grasp_id = int(input("Please input the grasp ID of the grasp you want"))


    if visualize: # Avoid corrupt current grasps
        np.save(f"data/predicted_grasps/{exp_name}.npy", proj_grasps[grasp_id])
    return score / len(proj_grasps)

def main():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None, required=True)
    parser.add_argument("--env",type=str, default=None, required=True)
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()
    r_suc = validate_path(args.exp_name, args.env, args.visualize)

if __name__ == "__main__":
    main()







    
        
    
