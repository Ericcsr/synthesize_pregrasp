import os
import time
from functools import partial
import open3d as o3d
import neurals.dataset
import neurals.dex_grasp_net as dgn
from neurals.test_options import TestOptions
import torch.utils.data
import numpy as np
import torch
import torch.nn as nn

from utils.dyn_feasibility_check import simple_check_dyn_feasible
from utils.kin_feasibility_check import check_kin_feasible_parallel
import model.manipulation_obj_creator as creator
NUM_NEIGHBORS = 1

# Here finger tips should include condition + result hence it is fine
def parse_input(data):
    return data['point_cloud'].cuda().float(), data["point_normals"].cuda().float(), data['fingertip_pos'].cuda().float(), float(data["intrinsic_score"]),  int(data["label"]), 

def parse_extra_cond(fingertip_poses, extra_cond_finger):
    """
    Assume fingertip pose is a [batch_size, 3 * num_fingers] vector
    extra_cond_finger is a list of finger id
    """
    batch_size = fingertip_poses.shape[0]
    fingertip_pose_folded = fingertip_poses.view(batch_size,-1, 3)
    return fingertip_pose_folded[:,extra_cond_finger,:].view(batch_size, -1).contiguous()

def create_kd_tree(pcd_torch):
    pcd_np = pcd_torch[0].cpu().numpy()
    pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_np))
    kd_tree = o3d.geometry.KDTreeFlann(pcd_o3d)
    return kd_tree

def project_to_pcd(kd_tree, pcd_torch, normals_torch, grasp, extra_cond, num_neighbors=NUM_NEIGHBORS):
    """
    project one grasp onto pointcloud, one grasp only
    """
    pcd_np = pcd_torch[0].cpu().numpy()
    normals_np = normals_torch[0].cpu().numpy()
    grasp_points = grasp.flatten() # [2]
    extra_cond_points = extra_cond.flatten().cpu().numpy() # [0, 1]
    points = np.hstack([extra_cond_points, grasp_points]).reshape(-1,3)
    projected_points = np.zeros((points.shape[0] * num_neighbors, 3))
    projected_normals = np.zeros((points.shape[0] * num_neighbors, 3))
    for i, point in enumerate(points):
        index = kd_tree.search_knn_vector_3d(point, num_neighbors)[1]
        projected_points[i*num_neighbors:(i+1)*num_neighbors] = pcd_np[index]
        projected_normals[i*num_neighbors:(i+1)*num_neighbors] = normals_np[index]
    return projected_points, projected_normals, projected_points[::num_neighbors], projected_normals[::num_neighbors]

def main():
    parser = TestOptions()
    parser.parser.add_argument("--num_samples", type=int, default=20)
    parser.parser.add_argument("--exp_name", type=str, default="")
    parser.parser.add_argument("--hand_model",type=str, default="shadow")
    parser.parser.add_argument("--env",type=str, default="laptop")
    parser.parser.add_argument("--start_idx", type=int, default=0)
    parser.parser.add_argument("--save_freq", type=int, default=50)

    opt = parser.parse()
    opt = opt.__dict__
    opt["force_skip_load"] = False
    if opt == None:
        return

    model = dgn.DexGraspNetModel(opt, pred_base=False, pred_fingers=[2,3,4], extra_cond_fingers=[0,1], gpu_id=0)
    # Prepare dataset, should consist of both negative dataset and positive ones
    full_dataset = neurals.dataset.SmallDataset(seed_folder="seeds_scale",
                                                point_clouds = ["pose_00_pcd","pose_01_pcd","pose_02_pcd","pose_03_pcd",
                                                                "pose_04_pcd","pose_05_pcd","pose_06_pcd","pose_07_pcd",
                                                                "pose_08_pcd","pose_09_pcd","pose_10_pcd","pose_11_pcd",
                                                                "pose_12_pcd","pose_13_pcd","pose_14_pcd","pose_15_pcd",
                                                                "pose_16_pcd","pose_17_pcd","pose_18_pcd","pose_19_pcd",
                                                                "pose_20_pcd","pose_21_pcd","pose_22_pcd","pose_23_pcd",
                                                                "pose_24_pcd","pose_25_pcd","pose_26_pcd","pose_27_pcd",
                                                                "pose_28_pcd","pose_29_pcd","pose_30_pcd","pose_31_pcd"],
                                                has_negative_grasp=True)
    train_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=1)
    # writer = Writer(opt)
    print('Dataset loaded, beginning generating negative data')
    model.eval() # The encoder and decoder should not be trained
    # Prepare for data structures
    scores = []
    point_clouds = []
    conditions = []
    point_cloud_labels = []
    # Each time draw a fixed batch of random normal sample as latent states
    torch.manual_seed(2020)
    
    exp_name = opt["exp_name"]
    if not os.path.isdir(f"data/score_function_data/{exp_name}/"):
        os.mkdir(f"data/score_function_data/{exp_name}/")

    for i,data in enumerate(train_loader):
        if i < opt['start_idx']:
            continue
        randomized_z = torch.randn((opt['num_samples'], opt['latent_size']))
        pcd, normal, tip_pos, intrinsic_score, label = parse_input(data)
        obj_creator = partial(creator.object_creators[opt["env"]], scale=data['scale'][0])
        kd_tree = create_kd_tree(pcd)
        pcds = pcd.expand(opt['num_samples'], -1, 3).contiguous() # [batch_size, num_points, 3]
        extra_cond = parse_extra_cond(tip_pos, [0,1])
        extra_conds = extra_cond.expand(opt['num_samples'],-1).contiguous()
        grasps_raw = model.generate_grasps(pcds, randomized_z, extra_conds)[0].cpu().numpy()
        
        # Project to the pointcloud Should only have one grasp
        projected_grasps = []
        for grasp in grasps_raw:
            # Each time handle a single grasp
            projected_grasps.append(project_to_pcd(kd_tree, pcd, normal, grasp, extra_conds[0]))

        # check dyn_feasibility and kin_feasibility of each grasps (no need for permutation)
        score = 0
        for projected_grasp in projected_grasps:
            if simple_check_dyn_feasible(projected_grasp[2], projected_grasp[3]) \
                and check_kin_feasible_parallel(projected_grasp[2],projected_grasp[3], hand_model=opt["hand_model"],object_creator=obj_creator)[0]:
                score += 1
        print(f"Current {i} score: ", score)
        scores.append(float(score)/opt['num_samples'])
        point_clouds.append(pcd.cpu().numpy())
        conditions.append(extra_cond.cpu().numpy())
        point_cloud_labels.append(int(label))
        # Each grasp need to be paired with a pointcloud
        if i % opt["save_freq"] == 0 and i != 0:
            np.savez(f"data/score_function_data/{exp_name}/score_data_{i}.npz",
                    scores=np.asarray(scores),
                    point_clouds=np.vstack(point_clouds),
                    conditions=np.vstack(conditions),
                    point_cloud_labels = point_cloud_labels)
            scores = []
            point_clouds = []
            conditions = []
            point_cloud_labels = []

if __name__ == '__main__':
    main()
