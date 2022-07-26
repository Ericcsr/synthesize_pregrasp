import model.param as model_param
import data.data_generation_config as dgc
import pydrake
import numpy as np
import os
import open3d as o3d
import argparse

from scipy.spatial.transform import Rotation as scipy_rot


def compute_drake_ycb_point_clouds_from_mesh(output_path, num_points = dgc.num_points_in_point_cloud):
    # Walk the data path
    obj_dir = os.path.join(pydrake.getDrakePath(), "manipulation/models/ycb/meshes")
    for root, dirs, files in os.walk(obj_dir,
                                     topdown=False):
        for f in files:
            if f.endswith(".obj"):
                # Object file
                obj_file = os.path.join(obj_dir, f)
                print(obj_file)
                mesh = o3d.io.read_triangle_mesh(obj_file)
                pc = mesh.sample_points_poisson_disk(num_points)
                # eg. 003_cracker_box_textured -> 003_cracker_box
                drake_obj_name = "_".join(f.split("_")[:-1])
                # Transform the point cloud against the pose in SDF so it aligns
                # FIXME: somehow this doesn't work directly with open3d. We now convert to
                # numpy and back, then store in open3d.
                # Maybe just eliminate open3d altogether in the future?
                mesh_sdf_xyz_rpy = np.asarray(model_param.drake_mesh_sdf_pose[drake_obj_name])
                # SDF uses xyz convention
                r_WP = scipy_rot.from_euler('xyz', mesh_sdf_xyz_rpy[3:]).as_matrix()
                t_WP = mesh_sdf_xyz_rpy[:3]

                pc_np = np.array(pc.points)
                pc_np = (r_WP @ pc_np.T).T+t_WP
                pc_normal_np = np.array(pc.normals)
                pc_normal_np = (r_WP @ pc_normal_np.T).T

                pc_transformed = o3d.geometry.PointCloud()
                pc_transformed.points = o3d.utility.Vector3dVector(pc_np)
                pc_transformed.normals = o3d.utility.Vector3dVector(pc_normal_np)

                name = f.split('.')[0]
                o3d.io.write_point_cloud('/'.join([output_path,
                                                   name+f'_{num_points}.ply']),
                                         pc_transformed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute object point cloud")
    parser.add_argument('--num_points', type=int, default=dgc.num_points_in_point_cloud,
                        help='Index from which to compute')
    args = parser.parse_args()
    output_path = '/'.join([os.path.dirname(os.path.abspath(__file__)),"..",
                            dgc.point_cloud_path])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    compute_drake_ycb_point_clouds_from_mesh(output_path, num_points=args.num_points)