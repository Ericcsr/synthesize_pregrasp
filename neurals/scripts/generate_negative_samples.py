import os
import time
import open3d as o3d
import neurals.dataset
import neurals.dex_grasp_net as dgn
from neurals.network import ScoreFunction 
from neurals.test_options import TestOptions
import torch.utils.data
import numpy as np
import torch
import torch.nn as nn

from utils.compute_dyn_feasible_contacts import check_dyn_feasible
from utils.compute_kin_feasible_contacts import check_kin_feasible

current_dir = "/home/sirius/sirui/contact_planning_dexterous_hand/neurals/pretrained_score_function"

# Here finger tips should include condition + result hence it is fine
def parse_input(data):
    return data['point_cloud'].cuda(), data["point_normals"], data['fingertip_pos'].cuda()

def parse_extra_cond(fingertip_poses, extra_cond_finger):
    """
    Assume fingertip pose is a [batch_size, 3 * num_fingers] vector
    extra_cond_finger is a list of finger id
    """
    batch_size = fingertip_poses.shape[0]
    fingertip_pose_folded = fingertip_poses.view(batch_size,-1, 3)
    return fingertip_pose_folded[:,extra_cond_finger,:].view(batch_size, -1)

def create_kd_tree(pcd_torch):
    pcd_np = pcd_torch[0].cpu().numpy()
    pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_np))
    kd_tree = o3d.geometry.KDTreeFlann(pcd_o3d)
    return kd_tree

def project_to_pcd(kd_tree, pcd_torch, normals_torch, grasp, extra_cond):
    """
    project one grasp onto pointcloud, one grasp only
    """
    pcd_np = pcd_torch[0].cpu().numpy()
    normals_np = normals_torch[0].cpu().numpy()
    grasp_points = grasp.flatten().cpu().numpy()
    extra_cond_points = extra_cond.flatten().cpu().numpy()
    points = np.hstack([grasp_points, extra_cond_points]).reshape(-1,3)
    projected_points = np.zeros_like(points)
    projected_normals = np.zeros_like(points)
    for i, point in enumerate(points):
        index = kd_tree.search_knn_vector_3d(point, 1)[1]
        projected_points[i] = pcd_np[index]
        projected_normals[i] = normals_np[index]
    return projected_points, projected_normals

def main():
    parser = TestOptions()
    parser.parser.add_argument("--num_samples", dtype=int, default=20)
    opt = parser.parse()
    if opt == None:
        return
    # TODO: Can we handle customized object now?
    objects = ("003_cracker_box",)
    print('objects', objects)

    model = dgn.DexGraspNetModel(opt, pred_base=False, num_in_feats=9)
    # Prepare dataset
    full_dataset = neurals.dataset.make_dataset_from_point_clouds_score_function(
        objects, copies=opt.grasp_perturb_copies, 
        pc_noise_variance=opt.pc_noise_variance)
    train_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=1)
    # writer = Writer(opt)
    print('Dataset loaded, beginning generating negative data')
    model.eval() # The encoder and decoder should not be trained
    # Prepare for data structures
    positive_sample =  []
    positive_point_cloud = []
    negative_sample = []
    negative_point_cloud = []
    randomized_z = torch.randn((opt.num_samples, opt.latent_size))
    for data in train_loader:
        pcd, normal, tip_pos = parse_input(data)
        kd_tree = create_kd_tree(pcd)
        pcds = pcd.expand(opt.num_samples, -1, 3) # [batch_size, num_points, 3]
        extra_conds = parse_extra_cond(tip_pos, [0,1]).expand(opt.num_samples,-1)
        grasps_raw = model.generate_grasps(pcds, randomized_z, extra_conds).cpu().numpy()
        
        # Project to the pointcloud
        projected_grasps = []
        for grasp in grasps_raw:
            projected_grasps.append(project_to_pcd(kd_tree, pcd, normal, grasp, extra_conds[0]))

        # check dyn_feasibility and kin_feasibility of each grasps (no need for permutation)
        for projected_grasp in projected_grasps:
            if check_dyn_feasible(projected_grasp[0], projected_grasp[1], tol=1e-4) and check_kin_feasible(projected_grasp[0], projected_grasp[1],has_tabletop=False):
                positive_sample.append(projected_grasp[0])
                positive_point_cloud.append(pcd[0].cpu().numpy())
            else:
                negative_sample.append(projected_grasp[0])
                negative_point_cloud.append(pcd[0].cpu().numpy())
    # Each grasp need to be paired with a pointcloud
    np.savez("data/score_function_data/positive.npz", contact_points=positive_sample, point_clouds=positive_point_cloud)
    np.savez("data/score_function_data/negative.npz", contact_points=negative_sample, point_clouds=negative_point_cloud)


if __name__ == '__main__':
    main()
