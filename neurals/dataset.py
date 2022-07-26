import neurals.data_generation_config as dgc
import model.param as model_param

import copy
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as scipy_rot
import torch.utils.data

class DexGrasp:
    def __init__(self, point_cloud, fingertip_normals, base_position, base_quaternion, finger_q):
        """

        :param point_cloud:
        :param fingertip_normals: kx6 where k=len(ActiveAllegroHandFingers)
        :param base_position:
        :param base_quaternion:
        :param finger_q:
        """
        self.point_cloud = point_cloud
        self.fingertip_normals = fingertip_normals
        self.base_position = base_position
        self.base_quaternion = np.squeeze(base_quaternion)
        # Normalize quaternion
        self.base_quaternion /= np.linalg.norm(self.base_quaternion)
        base_quaternion_scipy = base_quaternion[[1, 2, 3, 0]]
        self.base_rotation_matrix_flattened = scipy_rot.from_quat(
            base_quaternion_scipy).as_matrix().flatten()  # 9D
        self.finger_q = np.squeeze(finger_q)
        # A concatenated version of base_position, base_quaternion, finger_q
        self.grasp = np.hstack(
            [self.base_position, self.base_rotation_matrix_flattened])

class DexGraspFingerOnly:
    def __init__(self, point_cloud, point_cloud_normals, fingertip_pos, label):
        self.point_cloud = torch.from_numpy(point_cloud).float()
        self.point_normals = torch.from_numpy(point_cloud_normals).float()
        self.fingertip_pos = fingertip_pos.flatten()
        self.label = label

class DexGraspFingerOnlyDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, grasps):
        torch.utils.data.dataset.Dataset.__init__(self)
        assert(isinstance(grasps, list))
        assert(isinstance(grasps[0], DexGraspFingerOnly))
        self.grasps = grasps

    def __len__(self):
        return len(self.grasps)

    def __getitem__(self, idx):
        ans = {}
        ans["point_cloud"] = self.grasps[idx].point_cloud
        ans["point_normals"] = self.grasps[idx].point_normals
        ans["fingertip_pos"] = self.grasps[idx].fingertip_pos
        ans["label"] = self.grasps[idx].label
        return ans

class DexGraspDatasetNew(torch.utils.data.dataset.Dataset):
    def __init__(self, grasps):
        torch.utils.data.dataset.Dataset.__init__(self)
        assert(isinstance(grasps, list))
        assert(isinstance(grasps[0], DexGrasp))
        self.grasps = grasps

    def __len__(self):
        return len(self.grasps)

    def __getitem__(self, idx):
        """
        If the user doesn't provide a transform function in the class
        constructor, then we will convert nan in the contact configuration to
        the number of faces in the manipuland (because the negative
        log-likelihood loss function only accept non-negative value, so we
        choose a positive value number_of_manipuland_faces to represent not
        in contact).
        @return (state[n], (q[n+1], contact_configuration[n+1])).
        If the user doesn't provide a transform function, then by default we
        change all np.nan in contact_configuration to num_manipuland_faces.
        """
        ans = {}
        ans['point_cloud'] = self.grasps[idx].point_cloud
        ans['grasp'] = self.grasps[idx].grasp
        ans['fingertip_normals'] = self.grasps[idx].fingertip_normals
        # The following are not used by the network
        ans['finger_q'] = self.grasps[idx].finger_q
        ans['base_quaternion'] = self.grasps[idx].base_quaternion
        return ans

class ScoreFunctionDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, positive_filepaths, negative_filepaths):
        positive_samples = []
        positive_point_cloud = []
        for positive_filepath in positive_filepaths:
            data = np.load(f"data/score_function_data/{positive_filepath}.npz")
            positive_samples.append(data['contact_points'])
            positive_point_cloud.append(data['point_clouds'])
        positive_samples = np.concatenate(positive_samples, axis=0)
        positive_point_cloud = np.concatenate(positive_point_cloud, axis=0)

        negative_samples = []
        negative_point_cloud = []
        for negative_filepath in negative_filepaths:
            data = np.load(f"data/score_function_data/{negative_filepath}.npz")
            negative_samples.append(data['contact_points'])
            negative_point_cloud.append(data['point_clouds'])
        negative_samples = np.concatenate(negative_samples, axis=0)
        negative_point_cloud = np.concatenate(negative_point_cloud, axis=0)

        self.labels = np.hstack((np.ones(len(positive_samples)), np.zeros(len(negative_samples))))
        self.samples = np.concatenate([positive_samples, negative_samples], axis=0)
        self.point_clouds = np.concatenate([positive_point_cloud, negative_point_cloud], axis=0)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ans = {}
        ans["point_cloud"] = self.point_clouds[idx]
        ans["fingertip_pos"] = self.samples[idx]
        ans["label"] = self.labels[idx]
        return ans

# Each file correspond to a pointcloud
class SmallDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, positive_grasp_files=["pose_0","pose_1","pose_2","pose_3","pose_4"], 
                       point_clouds=["pose_0_pcd", "pose_1_pcd", "pose_2_pcd", "pose_3_pcd", "pose_4_pcd"], 
                       negative_grasp_files=None):
        self.positive_grasps = []
        self.positive_pcd_mapping = []
        for i,positive_grasp_file in enumerate(positive_grasp_files):
            self.positive_grasps.append(np.load(f"data/good_grasps/{positive_grasp_file}.npy"))
            self.positive_pcd_mapping += [i] * len(self.positive_grasps[-1])
        
        if not(negative_grasp_files is None):
            self.negative_grasps = []
            self.negative_pcd_mapping = []
            for i, negative_grasp_file in enumerate(negative_grasp_files):
                self.negative_grasps.append(np.load(f"data/bad_grasps/{negative_grasp_file}.npy"))
                self.negative_pcd_mapping += [i] * len(self.negative_grasps[-1])
        
        self.point_clouds = []
        for point_cloud in point_clouds:
            pcd = o3d.io.read_point_cloud(f"data/hard_code_point_cloud/{point_cloud}.ply")
            pcd_np = np.asarray(pcd.points)
            assert(pcd_np.shape[0] > 1024)
            idx = np.random.choice(pcd_np.shape[0], 1024, replace=False)
            self.point_clouds.append(pcd_np[idx])
        if not(negative_grasp_files is None):
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
        ans["fingertip_pos"] = self.grasps[idx]
        ans["label"] = self.labels[idx]
        return ans   

# Inner dataset creation function shouldn't be called outside.
def make_dataset_from_same_point_cloud(obj_name, make_datset=True, random_xy=True,
                                       random_yaw=True, random_state=None,
                                       copies=3,
                                       data_idx_start=0, data_idx_end=np.inf,
                                       file_name_num_grasps=None, return_object_transforms=False,
                                       num_points_in_input_point_cloud=model_param.initial_num_points_in_input_point_cloud,
                                       num_points_in_ouput_point_cloud=model_param.final_num_points_in_input_point_cloud,
                                       pc_noise_variance=0.):
    if random_state is None:
        random_state = np.random.RandomState()
    kin_feasible_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '../data',
                                          dgc.kin_feasible_configs_path)
    obj_poses_path = os.path.join(
        kin_feasible_file_path, obj_name+f'_poses.npy')
    obj_poses = np.load(obj_poses_path)
    all_grasps_list = []
    object_base_positions = []
    object_base_quaternions = []
    pc_load = o3d.io.read_point_cloud(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../data',
        dgc.point_cloud_path, obj_name+f'_textured_{num_points_in_input_point_cloud}.ply'))

    for pose_idx, obj_pose in enumerate(obj_poses):
        if file_name_num_grasps is not None:
            grasp_data_path = os.path.join(
                kin_feasible_file_path, obj_name+f'_pose_{pose_idx}_{file_name_num_grasps}.npy')
        else:
            grasp_data_path = os.path.join(
                kin_feasible_file_path, obj_name+f'_pose_{pose_idx}.npy')
        p_WB = obj_pose[4:]
        q_WB = obj_pose[:4]
        T_WB = np.eye(4)
        rot_WB_scipy = scipy_rot.from_quat(q_WB[[1, 2, 3, 0]])
        T_WB[:3, :3] = rot_WB_scipy.as_matrix()
        T_WB[:3, -1] = p_WB
        # Currently we don't use normals
        data = np.load(grasp_data_path, allow_pickle=True)
        grasps_list = []
        data_idx_end = int(min(len(data), data_idx_end))
        for data_i in data[data_idx_start:data_idx_end]:
            # Note that Drake stores quaternion in wxyz format
            for _ in range(copies):
                fingertip_normals, hand_conf = copy.deepcopy(data_i)
                base_position, base_quaternion, finger_angles = copy.deepcopy(
                    hand_conf)
                fingertip_normals_array = np.copy(np.vstack(
                    [fingertip_normals[finger] for finger in model_param.ActiveAllegroHandFingers]))
                finger_q = np.copy(model_param.finger_angles_dict_to_finger_q(
                    finger_angles))
                # Handle the random transform with scipy
                random_T = np.eye(4)
                if random_yaw:
                    # Random angle in 0~2pi
                    yaw_val = np.random.rand(1)*2*np.pi
                    rnd_yaw_scipy = scipy_rot.from_euler(
                        "xyz", [0., 0., yaw_val])
                    rnd_yaw_matrix = rnd_yaw_scipy.as_matrix()
                    random_T[:3,:3] = rnd_yaw_matrix
                    # Compute for hand
                    # base_quaternion, base_position, fingertip_normals_array are computed w.r.t. q_WB,
                    # so they are only changed by rnd_yaw_scipy and rnd_trans_scipy
                    base_quaternion_scipy = scipy_rot.from_quat(
                        base_quaternion[[1, 2, 3, 0]])
                    base_quaternion = (
                        rnd_yaw_scipy*base_quaternion_scipy).as_quat()[[3, 0, 1, 2]]
                    # compute for the normals
                    # For the normals, just rotate
                    fingertip_normals_array[:, 3:] = (
                        rnd_yaw_matrix @ (fingertip_normals_array[:, 3:].T)).T
                    # For the positions, rotate and translate
                    fingertip_normals_array[:, :3] = (
                        rnd_yaw_matrix @ (fingertip_normals_array[:, :3].T)).T
                    base_position = rnd_yaw_matrix @ base_position
                    
                if random_xy:
                    rnd_trans = random_state.uniform(
                        -dgc.max_random_translation, dgc.max_random_translation, 3)
                    # No z change
                    rnd_trans[2] = 0.
                    random_T[:3,-1] = rnd_trans
                    # Note that this is w.r.t. object center frame
                    # Thus it contains the rest pose and random rotation
                    # base_quaternion, base_position, fingertip_normals_array are computed w.r.t. q_WB,
                    # so they are only changed by rnd_yaw_scipy and rnd_trans_scipy
                    fingertip_normals_array[:, :3] += rnd_trans
                    base_position += rnd_trans
                
                # Random transform is applied after world transform
                final_obj_T = random_T @ T_WB
                final_obj_trans = final_obj_T[:3,-1]
                final_obj_rot_scipy = scipy_rot.from_matrix(final_obj_T[:3,:3])

                # Do all the point cloud transformation here
                pc = copy.deepcopy(pc_load).transform(T_WB)
                # Crop and downsample the point cloud
                aabb = pc.get_axis_aligned_bounding_box()
                aabb = aabb.translate(
                    np.array([0., 0., model_param.point_cloud_tabletop_occlusion_height]))
                # Crop the point cloud
                pc_cropped = copy.deepcopy(pc).crop(aabb)
                # print(f"Cropped cloud of ", obj_name,
                #     f" pose {pose_idx} has {len(pc_cropped.points)} points")
                pc_cropped_random_T = copy.deepcopy(pc_cropped).transform(random_T)
                pc_points_random_T_np = np.copy(np.array(pc_cropped_random_T.points))

                # Downsample to num_points_in_ouput_point_cloud
                point_idxs = random_state.choice(np.arange(
                    pc_points_random_T_np.shape[0]), num_points_in_ouput_point_cloud, replace=False)
                pc_points_downsampled = pc_points_random_T_np[point_idxs, :]
                # Add Gaussian noise to the point cloud
                if pc_noise_variance > 0.:
                    pc_points_downsampled += random_state.normal(
                        scale=pc_noise_variance, size=pc_points_downsampled.shape)

                # transform the point cloud
                grasps_list.append(DexGrasp(pc_points_downsampled,
                                            fingertip_normals_array,
                                            base_position,
                                            base_quaternion,
                                            finger_q))
                object_base_positions.append(final_obj_trans)
                # Convert to Drake quaternion convention
                object_base_quaternions.append(
                    final_obj_rot_scipy.as_quat()[[3, 0, 1, 2]])
        all_grasps_list.extend(grasps_list)
    if make_datset:
        all_grasps_list = DexGraspDatasetNew(all_grasps_list)
    if return_object_transforms:
        return all_grasps_list, object_base_positions, object_base_quaternions, [obj_name]*len(all_grasps_list)
    return all_grasps_list

def make_dataset_from_point_clouds(obj_name_list, shuffle=True, shuffle_seed=0,
                                   copies=3, num_grasps=None,
                                   random_xy=True,
                                   random_yaw=True,
                                   return_object_transforms=False,
                                   num_points_in_input_point_cloud=model_param.initial_num_points_in_input_point_cloud,
                                   num_points_in_ouput_point_cloud=model_param.final_num_points_in_input_point_cloud,
                                   pc_noise_variance=0.):
    grasps_list = []
    object_base_positions = []
    object_base_quaternions = []
    object_names_list = []
    for i in range(len(obj_name_list)):
        print(f'Making dataset for object '+obj_name_list[i])
        if return_object_transforms:
            g, pos, quat, names = make_dataset_from_same_point_cloud(obj_name_list[i],
                                                                     make_datset=False,
                                                                     copies=copies,
                                                                     random_xy=random_xy,
                                                                     random_yaw=random_yaw,
                                                                     file_name_num_grasps=num_grasps,
                                                                     return_object_transforms=return_object_transforms,
                                                                     num_points_in_input_point_cloud=num_points_in_input_point_cloud,
                                                                     num_points_in_ouput_point_cloud=num_points_in_ouput_point_cloud,
                                                                     pc_noise_variance=pc_noise_variance)
            grasps_list.extend(g)
            object_base_positions.extend(pos)
            object_base_quaternions.extend(quat)
            object_names_list.extend(names)
        else:
            grasps_list.extend(make_dataset_from_same_point_cloud(obj_name_list[i],
                                                                  make_datset=False,
                                                                  copies=copies,
                                                                  random_xy=random_xy,
                                                                  random_yaw=random_yaw,
                                                                  file_name_num_grasps=num_grasps,
                                                                  return_object_transforms=return_object_transforms,
                                                                  num_points_in_input_point_cloud=num_points_in_input_point_cloud,
                                                                  num_points_in_ouput_point_cloud=num_points_in_ouput_point_cloud,
                                                                  pc_noise_variance=pc_noise_variance))
    if shuffle:
        rs = np.random.RandomState(shuffle_seed)
        if not return_object_transforms:
            rs.shuffle(grasps_list)
        else:
            raise NotImplementedError
    if return_object_transforms:
        return DexGraspDatasetNew(grasps_list), object_base_positions, object_base_quaternions, object_names_list
    return DexGraspDatasetNew(grasps_list)

# TODO: Make dataset for classification problem
def make_dataset_from_same_point_cloud_score_function(obj_name, make_datset=True, random_xy=True,
                                                      random_yaw=True, random_state=None,
                                                      copies=3,
                                                      data_idx_start=0, data_idx_end=np.inf,
                                                      file_name_num_grasps=None, return_object_transforms=False,
                                                      num_points_in_input_point_cloud=model_param.initial_num_points_in_input_point_cloud,
                                                      num_points_in_ouput_point_cloud=model_param.final_num_points_in_input_point_cloud,
                                                      pc_noise_variance=0.,
                                                      include_negative=True):
    if random_state is None:
        random_state = np.random.RandomState()
    kin_feasible_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '../data',
                                          dgc.kin_feasible_configs_path)
    kin_infeasible_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "../data",
                                       dgc.kin_infeasible_configs_path)

    obj_poses_path = os.path.join(
        kin_feasible_file_path, obj_name+f'_poses.npy')
    obj_poses = np.load(obj_poses_path)

    all_grasps_list = []
    pc_load = o3d.io.read_point_cloud(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../data',
        dgc.point_cloud_path, obj_name+f'_textured_{num_points_in_input_point_cloud}.ply'))


    # Positive grasps
    for pose_idx, obj_pose in enumerate(obj_poses):
        if file_name_num_grasps is not None:
            grasp_data_path = os.path.join(
                kin_feasible_file_path+'/finger_tips_pose/', obj_name+f'_pose_{pose_idx}_{file_name_num_grasps}.npy')
        else:
            grasp_data_path = os.path.join(
                kin_feasible_file_path+'/finger_tips_pose/', obj_name+f'_pose_{pose_idx}.npy')
        p_WB = obj_pose[4:]
        q_WB = obj_pose[:4]
        T_WB = np.eye(4)
        rot_WB_scipy = scipy_rot.from_quat(q_WB[[1, 2, 3, 0]])
        T_WB[:3, :3] = rot_WB_scipy.as_matrix()
        T_WB[:3, -1] = p_WB
        # Currently we don't use normals
        data = np.load(grasp_data_path) # [N, 3, 3]
        grasps_list = []
        object_base_positions = []
        object_base_quaternions = []
        data_idx_end = int(min(len(data), data_idx_end))
        for data_i in data[data_idx_start:data_idx_end]:
            # Note that Drake stores quaternion in wxyz format
            for _ in range(copies):
                fingertip_pos = copy.deepcopy(data_i)
                
                # Handle the random transform with scipy
                random_T = np.eye(4)
                if random_yaw:
                    # Random angle in 0~2pi
                    yaw_val = np.random.rand(1)*2*np.pi
                    rnd_yaw_scipy = scipy_rot.from_euler(
                        "xyz", [0., 0., yaw_val])
                    rnd_yaw_matrix = rnd_yaw_scipy.as_matrix()
                    random_T[:3,:3] = rnd_yaw_matrix
                    # Compute for hand
                    # base_quaternion, base_position, fingertip_normals_array are computed w.r.t. q_WB,
                    # so they are only changed by rnd_yaw_scipy and rnd_trans_scipy
                    # compute for the normals
                    # For the normals, just rotate
                    fingertip_pos = (
                        rnd_yaw_matrix @ (fingertip_pos.T)).T
                    # For the positions, rotate and translate
                    fingertip_pos = (
                        rnd_yaw_matrix @ (fingertip_pos.T)).T
                    
                if random_xy:
                    rnd_trans = random_state.uniform(
                        -dgc.max_random_translation, dgc.max_random_translation, 3)
                    # No z change
                    rnd_trans[2] = 0.
                    random_T[:3,-1] = rnd_trans
                    # Note that this is w.r.t. object center frame
                    # Thus it contains the rest pose and random rotation
                    # base_quaternion, base_position, fingertip_normals_array are computed w.r.t. q_WB,
                    # so they are only changed by rnd_yaw_scipy and rnd_trans_scipy
                    fingertip_pos += rnd_trans
                
                # Random transform is applied after world transform
                final_obj_T = random_T @ T_WB
                final_obj_trans = final_obj_T[:3,-1]
                final_obj_rot_scipy = scipy_rot.from_matrix(final_obj_T[:3,:3])

                # Do all the point cloud transformation here
                pc = copy.deepcopy(pc_load).transform(T_WB)
                # Crop and downsample the point cloud
                aabb = pc.get_axis_aligned_bounding_box()
                aabb = aabb.translate(
                    np.array([0., 0., model_param.point_cloud_tabletop_occlusion_height]))
                # Crop the point cloud
                pc_cropped = copy.deepcopy(pc).crop(aabb)
                # print(f"Cropped cloud of ", obj_name,
                #     f" pose {pose_idx} has {len(pc_cropped.points)} points")
                pc_cropped_random_T = copy.deepcopy(pc_cropped).transform(random_T)
                pc_points_random_T_np = np.copy(np.array(pc_cropped_random_T.points))
                assert(pc_cropped_random_T.has_normals())
                pc_normals_random_T_np = np.copy(np.array(pc_cropped_random_T.normals))

                # Downsample to num_points_in_ouput_point_cloud
                point_idxs = random_state.choice(np.arange(
                    pc_points_random_T_np.shape[0]), num_points_in_ouput_point_cloud, replace=False)
                pc_points_downsampled = pc_points_random_T_np[point_idxs, :]
                pc_normals_downsampled = pc_normals_random_T_np[point_idxs, :]
                # Add Gaussian noise to the point cloud
                if pc_noise_variance > 0.:
                    pc_points_downsampled += random_state.normal(
                        scale=pc_noise_variance, size=pc_points_downsampled.shape)

                # transform the point cloud
                grasps_list.append(DexGraspFingerOnly(pc_points_downsampled, pc_normals_downsampled, fingertip_pos, 1.0))
                object_base_positions.append(final_obj_trans)
                # Convert to Drake quaternion convention
                object_base_quaternions.append(
                    final_obj_rot_scipy.as_quat()[[3, 0, 1, 2]])
        all_grasps_list.extend(grasps_list)

    if not include_negative:
        if make_datset:
            all_grasps_list = DexGraspDatasetNew(all_grasps_list)
        if return_object_transforms:
            return all_grasps_list, object_base_positions, object_base_quaternions, [obj_name]*len(all_grasps_list)
        return all_grasps_list

    # Prepare negative examples
    for pose_idx, obj_pose in enumerate(obj_poses):
        if file_name_num_grasps is not None:
            grasp_data_path = os.path.join(
                kin_infeasible_file_path, obj_name+f'_pose_{pose_idx}_{file_name_num_grasps}.npy')
        else:
            grasp_data_path = os.path.join(
                kin_infeasible_file_path, obj_name+f'_pose_{pose_idx}.npy')
        p_WB = obj_pose[4:]
        q_WB = obj_pose[:4]
        T_WB = np.eye(4)
        rot_WB_scipy = scipy_rot.from_quat(q_WB[[1, 2, 3, 0]])
        T_WB[:3, :3] = rot_WB_scipy.as_matrix()
        T_WB[:3, -1] = p_WB
        # Currently we don't use normals
        data = np.load(grasp_data_path) # [N, 3, 3]
        grasps_list = []
        data_idx_end = int(min(len(data), data_idx_end))
        for data_i in data[data_idx_start:data_idx_end]:
            # Note that Drake stores quaternion in wxyz format
            for _ in range(copies):
                fingertip_pos = copy.deepcopy(data_i)
                
                # Handle the random transform with scipy
                random_T = np.eye(4)
                if random_yaw:
                    # Random angle in 0~2pi
                    yaw_val = np.random.rand(1)*2*np.pi
                    rnd_yaw_scipy = scipy_rot.from_euler(
                        "xyz", [0., 0., yaw_val])
                    rnd_yaw_matrix = rnd_yaw_scipy.as_matrix()
                    random_T[:3,:3] = rnd_yaw_matrix
                    # Compute for hand
                    # base_quaternion, base_position, fingertip_normals_array are computed w.r.t. q_WB,
                    # so they are only changed by rnd_yaw_scipy and rnd_trans_scipy
                    # compute for the normals
                    # For the normals, just rotate
                    fingertip_pos = (
                        rnd_yaw_matrix @ (fingertip_pos.T)).T
                    # For the positions, rotate and translate
                    fingertip_pos = (
                        rnd_yaw_matrix @ (fingertip_pos.T)).T
                    
                if random_xy:
                    rnd_trans = random_state.uniform(
                        -dgc.max_random_translation, dgc.max_random_translation, 3)
                    # No z change
                    rnd_trans[2] = 0.
                    random_T[:3,-1] = rnd_trans
                    # Note that this is w.r.t. object center frame
                    # Thus it contains the rest pose and random rotation
                    # base_quaternion, base_position, fingertip_normals_array are computed w.r.t. q_WB,
                    # so they are only changed by rnd_yaw_scipy and rnd_trans_scipy
                    fingertip_pos += rnd_trans
                
                # Random transform is applied after world transform
                final_obj_T = random_T @ T_WB
                final_obj_trans = final_obj_T[:3,-1]
                final_obj_rot_scipy = scipy_rot.from_matrix(final_obj_T[:3,:3])

                # Do all the point cloud transformation here
                pc = copy.deepcopy(pc_load).transform(T_WB)
                # Crop and downsample the point cloud
                aabb = pc.get_axis_aligned_bounding_box()
                aabb = aabb.translate(
                    np.array([0., 0., model_param.point_cloud_tabletop_occlusion_height]))
                # Crop the point cloud
                pc_cropped = copy.deepcopy(pc).crop(aabb)
                # print(f"Cropped cloud of ", obj_name,
                #     f" pose {pose_idx} has {len(pc_cropped.points)} points")
                pc_cropped_random_T = copy.deepcopy(pc_cropped).transform(random_T)
                pc_points_random_T_np = np.copy(np.array(pc_cropped_random_T.points))

                # Downsample to num_points_in_ouput_point_cloud
                point_idxs = random_state.choice(np.arange(
                    pc_points_random_T_np.shape[0]), num_points_in_ouput_point_cloud, replace=False)
                pc_points_downsampled = pc_points_random_T_np[point_idxs, :]
                # Add Gaussian noise to the point cloud
                if pc_noise_variance > 0.:
                    pc_points_downsampled += random_state.normal(
                        scale=pc_noise_variance, size=pc_points_downsampled.shape)

                # transform the point cloud
                grasps_list.append(DexGraspFingerOnly(pc_points_downsampled,fingertip_pos, 0.0))
                object_base_positions.append(final_obj_trans)
                # Convert to Drake quaternion convention
                object_base_quaternions.append(
                    final_obj_rot_scipy.as_quat()[[3, 0, 1, 2]])
        all_grasps_list.extend(grasps_list)
    
    if make_datset:
        all_grasps_list = DexGraspDatasetNew(all_grasps_list)
    if return_object_transforms:
        return all_grasps_list, object_base_positions, object_base_quaternions, [obj_name]*len(all_grasps_list)
    return all_grasps_list

def make_dataset_from_point_clouds_score_function(obj_name_list, shuffle=True, shuffle_seed=0,
                                   copies=3, num_grasps=None,
                                   random_xy=True,
                                   random_yaw=True,
                                   return_object_transforms=False,
                                   num_points_in_input_point_cloud=model_param.initial_num_points_in_input_point_cloud,
                                   num_points_in_ouput_point_cloud=model_param.final_num_points_in_input_point_cloud,
                                   pc_noise_variance=0.,
                                   include_negative=True):
    grasps_list = []
    object_base_positions = []
    object_base_quaternions = []
    object_names_list = []
    for i in range(len(obj_name_list)):
        print(f'Making dataset for object '+obj_name_list[i])
        if return_object_transforms:
            g, pos, quat, names = make_dataset_from_same_point_cloud_score_function(obj_name_list[i],
                                                                     make_datset=False,
                                                                     copies=copies,
                                                                     random_xy=random_xy,
                                                                     random_yaw=random_yaw,
                                                                     file_name_num_grasps=num_grasps,
                                                                     return_object_transforms=return_object_transforms,
                                                                     num_points_in_input_point_cloud=num_points_in_input_point_cloud,
                                                                     num_points_in_ouput_point_cloud=num_points_in_ouput_point_cloud,
                                                                     pc_noise_variance=pc_noise_variance,
                                                                     include_negative=include_negative)
            grasps_list.extend(g)
            object_base_positions.extend(pos)
            object_base_quaternions.extend(quat)
            object_names_list.extend(names)
        else:
            grasps_list.extend(make_dataset_from_same_point_cloud_score_function(obj_name_list[i],
                                                                  make_datset=False,
                                                                  copies=copies,
                                                                  random_xy=random_xy,
                                                                  random_yaw=random_yaw,
                                                                  file_name_num_grasps=num_grasps,
                                                                  return_object_transforms=return_object_transforms,
                                                                  num_points_in_input_point_cloud=num_points_in_input_point_cloud,
                                                                  num_points_in_ouput_point_cloud=num_points_in_ouput_point_cloud,
                                                                  pc_noise_variance=pc_noise_variance,
                                                                  include_negative=include_negative))
    if shuffle:
        rs = np.random.RandomState(shuffle_seed)
        if not return_object_transforms:
            rs.shuffle(grasps_list)
        else:
            raise NotImplementedError
    if return_object_transforms:
        return DexGraspFingerOnlyDataset(grasps_list), object_base_positions, object_base_quaternions, object_names_list
    return DexGraspFingerOnlyDataset(grasps_list)

if __name__ == "__main__":
    objects = ("003_cracker_box",)
    dataset = make_dataset_from_point_clouds_score_function(objects, copies=3, pc_noise_variance=0.0)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=4)
    for data in dataloader:
        print(data)
        exit(0)