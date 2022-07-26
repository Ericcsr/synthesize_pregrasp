import neurals.dex_grasp_net as dgn
import model.param as model_param
from neurals.test_options import TestOptions
import copy
import numpy as np
import open3d as o3d
import torch.utils.data

def main():
    opt = TestOptions()
    opt.parser.add_argument(
        '--point_cloud_file',
        help="point cloud path",
        default="data/point_clouds/003_cracker_box_textured_2048.ply"
    )
    opt.parser.add_argument(
        '--num_latent',
        default=5,
        type=int
    )
    opt = opt.parse()
    if opt is None:
        return
    if opt.point_cloud_file is None:
        return
    while 1:
        seed = int(input("Enter random seed: "))
        if seed >= 0:
            print(f"Random seed {seed}")
            break
    rs = np.random.RandomState(seed)
    # First load the point cloud
    print("Loading point cloud from ", opt.point_cloud_file)
    # The original point cloud is given in B frame
    pc_o3d_B = o3d.io.read_point_cloud(opt.point_cloud_file)
    print('Initial point cloud number of points: ', len(pc_o3d_B.points))

    # Transform the pointcloud into canonical form
    pc_o3d_B_center = pc_o3d_B.get_center()
    print(f"point cloud center: {pc_o3d_B_center}")
    pc_o3d_P = copy.deepcopy(pc_o3d_B).translate(-pc_o3d_B_center)
    pc_np = np.copy(np.array(pc_o3d_P.points))

    # Load the model
    model = dgn.DexGraspNetModel(opt)

    # Downsample the point cloud to 1024
    if pc_np.shape[0] > model_param.final_num_points_in_input_point_cloud:
        point_idxs = np.random.choice(np.arange(
            pc_np.shape[0]), model_param.final_num_points_in_input_point_cloud, replace=False)
        pc_np = pc_np[point_idxs, :]
    # pc_torch should have shape (1,n,3)
    pc_torch = torch.Tensor(pc_np)[None,:]

    # Search for a reasonable answer among all results
    rs = np.random.RandomState(seed)
    for z_idx in range(opt.num_latent):
        # Sample a latent variable
        z = torch.Tensor(rs.normal(size=3)[None,:])
        print('\n\n################################')
        print('Beginning inference')
        predicted_grasp, _, z = model.generate_grasps(pc_torch, z)
        print('Sampled latent', z)
        predicted_grasp = torch.squeeze(predicted_grasp).cpu().detach().numpy()
        fingertip_positions_np_original = predicted_grasp[12:21].reshape((3,3))
        print('Predicted fingertip positions', fingertip_positions_np_original)
        nearby_radius = 4e-2#2e-2 # 4cm
        # First compute the surface manifolds
        nearest_points = {}
        nn_idxs = []
        fingertip_positions = {}
        all_nearby_points = []

        for fi, finger in enumerate(model_param.ActiveAllegroHandFingers):
            fingertip_positions[finger] = fingertip_positions_np_original[fi,:]
            # Naive nearest neighbors
            nn_idx = np.argmin(np.linalg.norm(pc_np-np.squeeze(fingertip_positions[finger]), axis=1))
            nearest_points[finger] = pc_np[nn_idx,:]
            nn_idxs.append(nn_idx)
            # Find the points in the vicinity of the projection
            near_idx = np.argwhere(np.linalg.norm(pc_np - nearest_points[finger], axis=1) < nearby_radius)
            all_nearby_points.append(pc_np[np.ndarray.flatten(near_idx),:])
        print(f"All Nearest Points in attemps {z_idx}:", nearest_points)


if __name__ == '__main__':
    main()
