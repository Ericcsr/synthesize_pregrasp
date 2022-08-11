import neurals.data_generation_config as dgc
import model.param as model_param

import copy
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as scipy_rot
import torch.utils.data

# Each file correspond to a pointcloud
class SmallDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, positive_grasp_folder="good_grasps", 
                       point_clouds=["pose_0_pcd", "pose_1_pcd", "pose_2_pcd", "pose_3_pcd", "pose_4_pcd"], 
                       negative_grasp_folder=None):
        self.positive_grasps = []
        self.positive_pcd_mapping = []
        positive_grasp_files = os.listdir(f"data/{positive_grasp_folder}/")
        for i,positive_grasp_file in enumerate(positive_grasp_files):
            self.positive_grasps.append(np.load(f"data/{positive_grasp_folder}/{positive_grasp_file}")[:,:,:3])
            self.positive_pcd_mapping += [i] * len(self.positive_grasps[-1])
        
        if not(negative_grasp_folder is None):
            self.negative_grasps = []
            self.negative_pcd_mapping = []
            negative_grasp_files = os.listdir(f"data/{negative_grasp_folder}")
            for i, negative_grasp_file in enumerate(negative_grasp_files):
                self.negative_grasps.append(np.load(f"data/{negative_grasp_folder}/{negative_grasp_file}")[:,:,:3])
                self.negative_pcd_mapping += [i] * len(self.negative_grasps[-1])
        
        self.point_clouds = []
        self.point_normals = []
        for point_cloud in point_clouds:
            pcd = o3d.io.read_point_cloud(f"data/hard_code_point_cloud/{point_cloud}.ply")
            pcd_np = np.asarray(pcd.points)
            normal_np = np.asarray(pcd.normals)
            assert(pcd_np.shape[0] > 1024)
            idx = np.random.choice(pcd_np.shape[0], 1024, replace=False)
            self.point_clouds.append(pcd_np[idx])
            self.point_normals.append(normal_np[idx])
        if not(negative_grasp_folder is None):
            self.grasps = np.concatenate(self.positive_grasps+self.negative_grasps, axis=0)
            self.pcd_mapping = np.array(self.positive_pcd_mapping+self.negative_pcd_mapping)
            self.labels = np.array([1]*len(self.positive_pcd_mapping)+[0]*len(self.negative_pcd_mapping))
        else:
            self.grasps = np.concatenate(self.positive_grasps, axis=0)
            self.pcd_mapping = np.array(self.positive_pcd_mapping)
            self.labels = np.array([1]*len(self.grasps))
        self.grasps = self.grasps.reshape(len(self.grasps),-1)
        self.point_clouds = np.asarray(self.point_clouds)
    
    def __len__(self):
        return len(self.pcd_mapping)

    def __getitem__(self, idx):
        ans = {}
        ans["point_cloud"] = self.point_clouds[self.pcd_mapping[idx]]
        ans["point_normals"] = self.point_normals[self.pcd_mapping[idx]]
        ans["fingertip_pos"] = self.grasps[idx]
        ans["label"] = self.labels[idx]
        return ans

# Dataset for score function, add noise tensor in order to prevent overfitting
class ScoreDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, score_file="score_data", noise_scale = 0.0):
        self.noise_scale = noise_scale
        data = np.load(f"data/score_function_data/{score_file}.npz")
        self.scores = data["scores"]
        self.point_clouds = data["point_clouds"]
        self.conditions = data["conditions"]
    
    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        ans = {}
        ans["point_cloud"] = self.point_clouds[idx] + np.random.normal(size=self.point_clouds[idx].shape, 
                                                                       scale=self.noise_scale)
        ans["condition"] = self.conditions[idx]
        ans["score"] = self.scores[idx]
        return ans
