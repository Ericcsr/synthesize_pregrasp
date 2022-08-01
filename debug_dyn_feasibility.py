import numpy as np
import open3d as o3d
from utils.dyn_feasibility_check import check_dyn_feasible, check_dyn_feasible_parallel

def create_o3d_box():
    mesh_box = o3d.geometry.TriangleMesh.create_box(0.4, 0.4, 0.1)
    mesh_box.translate([-0.2, -0.2, -0.05])
    mesh_box.compute_vertex_normals()
    mesh_box.compute_triangle_normals()
    point_box = mesh_box.sample_points_poisson_disk(2048, use_triangle_normal=True)
    kd_tree = o3d.geometry.KDTreeFlann(point_box)
    return point_box, kd_tree

def project_to_pcd(points, normals, tip_poses, kd_tree, num_neighbors):
    new_poses=[]
    for i in range(len(tip_poses)):
        idx = kd_tree.search_knn_vector_3d(tip_poses[i],num_neighbors)[1]
        new_poses.append(np.hstack([points[idx], normals[idx]]))
    new_poses.append(np.array([[100, 100, 100, 0, 0, 0]] * num_neighbors)) # Ring finger
    return np.vstack(new_poses)


def gen_grasp(tip_pose, pcd, kd_tree, num_fingers=4,num_neighbors=3):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    projected_tip_pose = project_to_pcd(points, normals, tip_pose, kd_tree, num_neighbors=num_neighbors)
    #print(projected_tip_pose.shape)
    vis_spheres = []
    for i in range(num_fingers-1):
        sp = o3d.geometry.TriangleMesh.create_sphere(0.01)
        sp.translate(projected_tip_pose[i*num_neighbors,:3])
        vis_spheres.append(sp)
    return projected_tip_pose, vis_spheres


box, kd_tree = create_o3d_box()
grasp = np.array([[0.15,  0.,    0.05],
                  [0.15,  0.05,  0.05],
                  [0.15, -0.05,  0.05]])
projected_grasp, vis = gen_grasp(grasp, box, kd_tree)
#print(projected_grasp)
#o3d.visualization.draw_geometries([box]+vis)
print("Start Checking Kin feasibility")
success_cnt = 0
for i in range(10):
    flag = check_dyn_feasible(projected_grasp[:,:3], projected_grasp[:,3:])
    success_cnt += 1 if not (flag is None) else 0
print(f"Start Checking done {success_cnt}")