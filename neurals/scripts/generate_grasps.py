import os
import open3d as o3d
import numpy as np
import utils.helper as helper
from neurals.NPGraspNet import NPGraspNet
from neurals.test_options import TestOptions
from utils.dyn_feasibility_check import check_dyn_feasible_parallel
from utils.kin_feasibility_check import check_kin_feasible_parallel

def load_traj_data(exp_name):
    tip_poses = np.load(f"data/tip_data/{exp_name}_tip_poses.npy")
    obj_poses = np.load(f"data/object_poses/{exp_name}_object_poses.npy")
    return tip_poses[-1], obj_poses[-1,:3], obj_poses[-1, 3:]

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
        if tip_pose[i].sum() < 100:
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

def visualize_finger_tip(tip_poses, radius=0.01):
    """
    tip_poses is (N, 6), N is number of fingers involved
    """
    vis_spheres = []
    colors = [[0.6, 0, 0],[0.0, 0.6, 0],[0.0, 0.0, 0.6],[0.6,0.6,0.6],[0.6,0,0.6]]
    for i,tip_pose in enumerate(tip_poses):
        if tip_pose[0] < 50:
            sp = o3d.geometry.TriangleMesh.create_sphere(radius)
            sp.paint_uniform_color(colors[i])
            sp.translate(tip_pose)
            vis_spheres.append(sp)
    return vis_spheres

def main():
    parser = TestOptions()
    parser.parser.add_argument("--exp_name", type=str, default=None, required=True)
    args = parser.parse()
    tip_pose, obj_pos, obj_orn = load_traj_data(args.exp_name)
    new_tip_pose, finger_idx = parse_raw_data(tip_pose, obj_pos, obj_orn)
    print(new_tip_pose[:], finger_idx)
    # Load pointcloud should be in canonical pose
    pcd_o3d = o3d.io.read_point_cloud(f"data/output_pcds/{args.exp_name}_pcd.ply")
    o3d.visualization.draw_geometries([pcd_o3d])
    assert(pcd_o3d.has_normals())
    points, normals, kd_tree = process_pcd(pcd_o3d)

    proj_tip_pose = new_tip_pose #project_grasp(new_tip_pose, kd_tree, points, normals)

    # Create and prepare neural network
    opt = args.__dict__
    opt["force_skip_load"] = False
    all_fingers = {0,1,2,3,4}
    pred_fingers = list(all_fingers.difference(finger_idx))
    print("Pred fingers:",pred_fingers)
    net = NPGraspNet(opt, mode="decoder", device="cuda:0",extra_cond_fingers=finger_idx, pred_fingers=pred_fingers)
    print("Cond:",proj_tip_pose[:,:3])
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
    for i,grasp in enumerate(proj_grasps):
        vis_tip = visualize_finger_tip(grasp[:,:3])
        #kin_feasibility = check_kin_feasible_parallel(grasp[:,:3], grasp[:,3:])
        # dyn_feasibility = check_dyn_feasible_parallel(grasp[:,:3], grasp[:,3:])
        #print("Grasp:", i, "Kinematics:", kin_feasibility[0])
        # Should be in local frame
        print("PCD ID:",i)
        o3d.visualization.draw_geometries([pcd_o3d]+vis_tip)
    np.save(f"data/predicted_grasps/{args.exp_name}.npy", proj_grasps)

if __name__ == "__main__":
    main()









    
        
    
