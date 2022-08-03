import os
from argparse import ArgumentParser
import numpy as np
import open3d as o3d

from utils.dyn_feasibility_check import check_dyn_feasible, check_dyn_feasible_parallel
from utils.kin_feasibility_check import check_kin_feasible, check_kin_feasible_parallel

DIFFUSION_RADIUS = 0.03
NUM_STEPS = 10
NUM_NEIGHBOR_SUPPORT=3

def load_seeds():
    """
    seeds: grasp seed have only finger tip pose, no normal vector
    """
    files = os.listdir(f"data/seeds/pointclouds")
    pcds = []
    for file in files:
        pcd = o3d.io.read_point_cloud(f"data/seeds/pointclouds/{file}")
        assert(pcd.has_normals())
        pcds.append(pcd)

    files = os.listdir("data/seeds/grasps")
    all_grasps = []
    for file in files:
        grasps = np.load(f"data/seeds/grasps/{file}")[:,:,:3]
        all_grasps.append(grasps)
    return pcds, all_grasps

def randomize_sample_trajectory(points, normals, kd_tree, seed, depth=8):
    """
    seed: only finger tip pose no normal vector
    """
    init_grasp = np.zeros((seed.shape[0], seed.shape[1] + 3))
    for i, point in enumerate(seed):
        init_idx = kd_tree.search_knn_vector_3d(point, 1)[1]
        init_grasp[i,:3] = points[init_idx]
        init_grasp[i,3:] = normals[init_idx]

    grasps_traj = [init_grasp]
    for i in range(depth):
        new_grasp = np.zeros_like(grasps_traj[-1])

        for i, point in enumerate(grasps_traj[-1]):
            indices = np.array(list(kd_tree.search_radius_vector_3d(point[:3], DIFFUSION_RADIUS)[1]))
            choice = np.random.choice(indices,1)
            new_grasp[i,:3] = points[choice]
            new_grasp[i,3:] = normals[choice]
        grasps_traj.append(new_grasp)
    return grasps_traj

def project_to_pcd(points, normals, grasp, kd_tree, num_neighbors):
    """
    grasp: finger tip pose and normal vector
    """
    new_poses=[]
    for i in range(len(grasp)):
        idx = kd_tree.search_knn_vector_3d(grasp[i,:3],num_neighbors)[1]
        new_poses.append(np.hstack([points[idx], normals[idx]]))
    new_poses.append(np.array([[100, 100, 100, 0, 0, 0]] * num_neighbors)) # Ring finger
    return np.vstack(new_poses)

def check_traj_feasibility(grasps_traj, points, normals, kd_tree, num_neighbors=3):
    """
    Assume grasps_traj have finger tip pose and normal vector
    """
    feasible_grasps = []
    infeasible_grasps = []
    for grasp in grasps_traj:
        tip_poses = project_to_pcd(points, normals, grasp, kd_tree, num_neighbors)
        flag = check_dyn_feasible_parallel(tip_poses[:,:3], tip_poses[:,3:])
        if flag is None:
            infeasible_grasps.append(grasp)
        elif not check_kin_feasible_parallel(np.vstack([grasp[:,:3], np.array([100,100,100])]),
                                             np.vstack([grasp[:,3:], np.array([0,0,0])]))[0]:
            infeasible_grasps.append(grasp)
        else:
            feasible_grasps.append(grasp)
    return feasible_grasps, infeasible_grasps

def main(args):
    all_feasible_grasps = []
    all_infeasible_grasps = []

    pcds, all_grasps = load_seeds()
    print(f"Seed Loaded: {len(pcds)} point clouds")
    for pcd, grasps in zip(pcds, all_grasps):
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        for seed_grasp in grasps:
            grasps_traj = randomize_sample_trajectory(points, normals, kd_tree, seed_grasp, depth=NUM_STEPS)
            feasible_grasps, infeasible_grasps = check_traj_feasibility(grasps_traj, points, normals, kd_tree, num_neighbors=NUM_NEIGHBOR_SUPPORT)
            all_feasible_grasps += feasible_grasps
            all_infeasible_grasps += infeasible_grasps
        print(f"Finish one pcd, total feasible grasps: {len(all_feasible_grasps)} infeasible grasps {len(all_infeasible_grasps)}")
    np.save(f"data/seeds/processed_data/{args.exp_name}_feasible_grasps.npy", np.asarray(all_feasible_grasps))
    np.save(f"data/seeds/processed_data/{args.exp_name}_infeasible_grasps.npy", np.asarray(all_infeasible_grasps))
    print("Processed data has been saved")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="default")
    args = parser.parse_args()
    main(args)
    
