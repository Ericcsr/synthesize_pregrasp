from functools import partial
import os
from argparse import ArgumentParser
import numpy as np
import open3d as o3d

from utils.dyn_feasibility_check import check_dyn_feasible, check_dyn_feasible_parallel, simple_check_dyn_feasible
from utils.kin_feasibility_check import check_kin_feasible, check_kin_feasible_parallel
import model.manipulation_obj_creator as creator

DIFFUSION_RADIUS = 0.03
NUM_STEPS = 10
NUM_NEIGHBOR_SUPPORT=1

def load_seeds(seed_folder, use_scale=False):
    """
    seeds: grasp seed have only finger tip pose, no normal vector
    """
    files = os.listdir(f"data/{seed_folder}/pointclouds")
    files.sort(key=lambda x: int(x[5:7]))
    pcds = []
    for file in files:
        pcd = o3d.io.read_point_cloud(f"data/{seed_folder}/pointclouds/{file}")
        assert(pcd.has_normals())
        pcds.append(pcd)

    files = os.listdir(f"data/{seed_folder}/grasps")
    files.sort(key=lambda x: int(x[5:7]))
    all_grasps = []
    for file in files:
        grasps = np.load(f"data/{seed_folder}/grasps/{file}")
        if len(grasps.shape) == 2:
            grasps = grasps.reshape(1,grasps.shape[0],grasps.shape[1])
        all_grasps.append(grasps[:,:,:3])
    if use_scale:
        files = os.listdir(f"data/{seed_folder}/scales")
        files.sort(key=lambda x: int(x[5:7]))
        all_scales = []
        for file in files:
            scale = np.load(f"data/{seed_folder}/scales/{file}")
            all_scales.append(scale)
        return pcds, all_grasps, all_scales
    else:
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
    if len(grasp) < 4:
        new_poses.append(np.array([[100, 100, 100, 0, 0, 0]] * num_neighbors)) # Ring finger
    return np.vstack(new_poses)

def check_traj_feasibility(grasps_traj, points, normals, kd_tree, num_neighbors=3, hand_model="allegro", env="laptop",scale=np.array([1,1,1])):
    """
    Assume grasps_traj have finger tip pose and normal vector
    For any small box object, use env == "laptop"
    """
    feasible_grasps = []
    infeasible_grasps = []
    for grasp in grasps_traj:
        tip_poses = project_to_pcd(points, normals, grasp, kd_tree, num_neighbors)
        flag = simple_check_dyn_feasible(tip_poses[:,:3], tip_poses[:,3:])
        print("Dyn:",flag)
        obj_creator = partial(creator.object_creators[env], scale=scale)
        if flag == False:
            infeasible_grasps.append(grasp)
        elif not check_kin_feasible_parallel(np.vstack([grasp[:,:3], np.array([100,100,100])]),
                                             np.vstack([grasp[:,3:], np.array([0,0,0])]), hand_model=hand_model,object_creator=obj_creator)[0]:
            infeasible_grasps.append(grasp)
        else:
            feasible_grasps.append(grasp)
    return feasible_grasps, infeasible_grasps

def main(args):
    if args.has_scale:
        pcds, all_grasps, all_scales = load_seeds(args.seed_folder, args.has_scale)
    else:
        pcds, all_grasps = load_seeds(args.seed_folder, args.has_scale)
    print(f"Seed Loaded: {len(pcds)} point clouds")
    for i, (pcd, grasps) in enumerate(zip(pcds, all_grasps)):
        if i < args.start_idx  or (args.use_idx is not None and str(i) not in args.use_idx):
            continue
        print(f"Seed {i} Loaded: {len(grasps)} grasps")
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        feasible_grasps_this_pose = []
        infeasible_grasps_this_pose = []
        for seed_grasp in grasps:
            grasps_traj = randomize_sample_trajectory(points, normals, kd_tree, seed_grasp, depth=NUM_STEPS)
            feasible_grasps, infeasible_grasps = check_traj_feasibility(grasps_traj, points, normals, kd_tree, num_neighbors=NUM_NEIGHBOR_SUPPORT, hand_model=args.hand_model, 
                                                                        env=args.env, scale=all_scales[i] if args.has_scale else np.array([1,1,1]))
            feasible_grasps_this_pose += feasible_grasps
            infeasible_grasps_this_pose += infeasible_grasps
        print(f"Finish {i} pcd, total feasible grasps: {len(feasible_grasps_this_pose)} infeasible grasps {len(infeasible_grasps_this_pose)}")
        if len(feasible_grasps_this_pose) != 0:
            np.save(f"data/{args.seed_folder}/processed_data/pose_{i:02}.npy", np.asarray(feasible_grasps_this_pose))
        np.save(f"data/{args.seed_folder}/processed_data/infeasible_pose_{i:02}.npy", np.asarray(infeasible_grasps_this_pose))
    print("Processed data has been saved")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--hand_model", type=str, default="shadow")
    parser.add_argument("--seed_folder", type=str, default="seeds")
    parser.add_argument("--env",type=str,default="laptop")
    parser.add_argument("--has_scale",action="store_true", default=False)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument('--use_idx','--list', action='append')
    args = parser.parse_args()
    main(args)
    
