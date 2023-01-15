import neurals.data_generation_config as dgc
import neurals.distancefield_utils as df
import model.param as model_param

import copy
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as scipy_rot
import torch.utils.data

# Each file correspond to a pointcloud
class SmallDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, seed_folder="seeds",
                       point_clouds=["pose_0_pcd", "pose_1_pcd", "pose_2_pcd", "pose_3_pcd", "pose_4_pcd"], 
                       has_negative_grasp=False,
                       use_scale = False):
        self.point_clouds = []
        self.point_normals = []
        pose_index = []
        for point_cloud in point_clouds:
            pose_index.append(int(point_cloud[5:7]))
            pcd = o3d.io.read_point_cloud(f"data/{seed_folder}/pointclouds/{point_cloud}.ply")
            pcd_np = np.asarray(pcd.points)
            normal_np = np.asarray(pcd.normals)
            if pcd_np.shape[0] < 1024:
                print(pcd_np.shape[0],point_cloud)
                assert(False)
            idx = np.random.choice(pcd_np.shape[0], 1024, replace=False)
            self.point_clouds.append(pcd_np[idx])
            self.point_normals.append(normal_np[idx])

        self.scales = []
        if os.path.isdir(f"data/{seed_folder}/scales") and use_scale:
            for idx in pose_index:
                scale = np.load(f"data/{seed_folder}/scales/pose_{idx:02}.npy")
                self.scales.append(scale)

        self.positive_grasps = []
        self.positive_pcd_mapping = []
        positive_grasp_files = os.listdir(f"data/{seed_folder}/grasps/")
        positive_grasp_files.sort(key=lambda x: int(x[5:7]))
        positive_files = []
        for positive_grasp_file in positive_grasp_files:
            if int(positive_grasp_file[5]) in pose_index:
                positive_files.append(positive_grasp_file)
        for i,positive_grasp_file in enumerate(positive_files):
            self.positive_grasps.append(np.load(f"data/{seed_folder}/grasps/{positive_grasp_file}")[:,:,:3])
            self.positive_pcd_mapping += [i] * len(self.positive_grasps[-1])

        if has_negative_grasp:
            self.negative_grasps = []
            self.negative_pcd_mapping = []
            negative_grasp_files = os.listdir(f"data/{seed_folder}/bad_grasps/")
            negative_grasp_files.sort(key=lambda x: int(x[5:7]))
            negative_files = []
            for negative_grasp_file in negative_grasp_files:
                if int(negative_grasp_file[5]) in pose_index:
                    negative_files.append(negative_grasp_file)
            for i, negative_grasp_file in enumerate(negative_files):
                self.negative_grasps.append(np.load(f"data/{seed_folder}/bad_grasps/{negative_grasp_file}")[:,:,:3])
                self.negative_pcd_mapping += [i] * len(self.negative_grasps[-1])
        
        
        if has_negative_grasp:
            self.grasps = np.concatenate(self.positive_grasps+self.negative_grasps, axis=0)
            self.pcd_mapping = np.array(self.positive_pcd_mapping+self.negative_pcd_mapping)
            self.labels = np.array([1]*len(self.positive_pcd_mapping)+[0]*len(self.negative_pcd_mapping))
        else:
            self.grasps = np.concatenate(self.positive_grasps, axis=0)
            self.pcd_mapping = np.array(self.positive_pcd_mapping)
            self.labels = np.array([1]*len(self.grasps))
        self.grasps = self.grasps.reshape(len(self.grasps),-1)
        print(self.grasps.shape)
        self.point_clouds = np.asarray(self.point_clouds)
    
    def __len__(self):
        return len(self.pcd_mapping)

    def __getitem__(self, idx):
        ans = {}
        ans["point_cloud"] = self.point_clouds[self.pcd_mapping[idx]]
        ans["point_normals"] = self.point_normals[self.pcd_mapping[idx]]
        ans["fingertip_pos"] = self.grasps[idx]
        ans["intrinsic_score"] = self.labels[idx] # Good or bad pointclouds
        ans["label"] = self.pcd_mapping[idx]
        if self.scales != []:
            ans["scale"] = self.scales[self.pcd_mapping[idx]]
        return ans

# Dataset for score function, add noise tensor in order to prevent overfitting
class ScoreDataset(torch.utils.data.dataset.Dataset):
    """
    Remark: To avoid overfitting between each round, we need to recreate dataset after each
    optimization iteration.
    """
    def __init__(self, score_file="score_data", noise_scale = 0.0, has_distance_field=False):
        self.noise_scale = noise_scale
        self.has_distance_field = has_distance_field
        data = np.load(f"data/score_function_data/{score_file}.npz")
        self.scores = data["scores"]
        self.point_clouds = data["point_clouds"] + np.random.normal(size=data["point_clouds"].shape, 
                                                                       scale=self.noise_scale)
        self.point_clouds = data["point_clouds"]
        self.conditions = data["conditions"]
        if has_distance_field:
            # Need to create different environments for computing distance field
            self.point_cloud_labels = data["point_cloud_labels"]
            self.envs = [env() for env in df.env_lists]
            self.load_env_configs()
            self.point_cloud_dfs = []
            for i, pcd in enumerate(self.point_clouds):
                self.point_cloud_dfs.append(self.compute_distance_field(pcd, int(self.point_cloud_labels[i])))
            self.point_cloud_dfs = np.asarray(self.point_cloud_dfs)
    
    def load_env_configs(self):
        self.pcd_to_env = []
        self.pcd_pos = []
        self.pcd_rotation = []
        filelist = os.listdir("data/seeds_scale/obj_pose")
        filelist.sort(key = lambda x: int(x[5]))
        for file in filelist:
            z = np.load(f"data/seeds_scale/obj_pose/{file}")
            self.pcd_to_env.append(int(z['env_id']))
            self.pcd_pos.append(z['trans'])
            self.pcd_rotation.append(z['rot'])

    def __len__(self):
        return len(self.scores)

    def compute_distance_field(self, pcd, pcd_id):
        """
        Assume pcd is a numpy array
        """
        pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
        env_id = self.pcd_to_env[pcd_id]
        dist_env = self.envs[env_id]
        pcd_o3d.rotate(self.pcd_rotation[pcd_id])
        pcd_o3d.translate(self.pcd_pos[pcd_id])
        points = np.asarray(pcd_o3d.points)
        dist_field = dist_env.get_points_distance(points)
        return dist_field.numpy()

    def __getitem__(self, idx):
        ans = {}
        ans["point_cloud"] = self.point_clouds[idx] # np.random.normal(size=self.point_clouds[idx].shape, scale=self.noise_scale)
        ans["condition"] = self.conditions[idx]
        ans["score"] = self.scores[idx]
        if self.has_distance_field:
            ans["point_cloud_df"] = self.point_cloud_dfs[idx]
        return ans
