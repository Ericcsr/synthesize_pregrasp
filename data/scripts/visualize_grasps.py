import open3d as o3d
import numpy as np
from argparse import ArgumentParser

COLORS = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1]]

def project_to_pcd(points, normals, grasp, kd_tree):
    """
    grasp: finger tip pose and normal vector
    """
    new_poses=[]
    for i in range(len(grasp)):
        idx = kd_tree.search_knn_vector_3d(grasp[i,:3],1)[1]
        new_poses.append(np.hstack([points[idx], normals[idx]]))
    return np.vstack(new_poses)

def visualize_tip(grasp):
    """
    grasp: valid finger tip poses
    """
    vis_sps = []
    for i,tip in enumerate(grasp):
        vis_sp = o3d.geometry.TriangleMesh.create_sphere(0.005)
        vis_sp.translate(tip[:3])
        vis_sp.paint_uniform_color(COLORS[i])
        vis_sps.append(vis_sp)
    return vis_sps

def load_data(seed_folder, posename):
    pcd = o3d.io.read_point_cloud(f"../{seed_folder}/pointclouds/{posename}_pcd.ply")
    raw_grasps = np.load(f"../{seed_folder}/grasps/{posename}.npy")
    return pcd, raw_grasps[:,:,:3]

def main(args):
    pcd, grasps = load_data(args.seed_folder,args.posename)
    if len(grasps.shape) == 2:
        grasps = grasps.reshape(1,-1,3)
    for grasp in grasps:
        print("Current grasp:",grasp)
        vis_tips = visualize_tip(grasp)
        o3d.visualization.draw_geometries([pcd]+vis_tips)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--posename", type=str, default=None, required=True)
    parser.add_argument("--seed_folder",type=str, default="seeds_full")
    args = parser.parse_args()
    main(args)