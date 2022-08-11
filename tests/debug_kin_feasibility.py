from utils.kin_feasibility_check import check_kin_feasible, check_kin_feasible_parallel
from utils import rigidBodySento as rb 
import open3d as o3d
import numpy as np
import pybullet as p
import time

def create_o3d_box():
    mesh_box = o3d.geometry.TriangleMesh.create_box(0.4, 0.4, 0.1)
    mesh_box.translate([-0.2, -0.2, -0.05])
    mesh_box.compute_vertex_normals()
    mesh_box.compute_triangle_normals()
    point_box = mesh_box.sample_points_poisson_disk(2048, use_triangle_normal=True)
    kd_tree = o3d.geometry.KDTreeFlann(point_box)
    return point_box, kd_tree

def project_to_pcd(points, normals, tip_poses, kd_tree):
    new_poses=[]
    for i in range(len(tip_poses)):
        idx = kd_tree.search_knn_vector_3d(tip_poses[i],1)[1]
        new_poses.append(np.hstack([points[idx], normals[idx]]).flatten())
    new_poses.append(np.array([100, 100, 100, 0, 0, 0])) # Ring finger
    return np.asarray(new_poses)

def create_pybullet_box():
    box = rb.create_primitive_shape(p, 1.0, p.GEOM_BOX, (0.2, 0.2, 0.05),         # half-extend
                                    color=(0.6, 0, 0, 0.8), collidable=True,
                                    init_xyz=[0, 0, 0],
                                    init_quat=[0, 0, 0, 1])
    return box

def load_hand():
    hand = p.loadURDF("model/resources/allegro_hand_description/urdf/allegro_hand_description_right.urdf", useFixedBase=True)
    return hand

def set_pybullet_hand_state(state, hand):
    base_position, base_quaternion, hand_q = state
    p.resetBasePositionAndOrientation(hand, base_position, base_quaternion)
    for i in range(len(hand_q)):
        p.resetJointState(hand, i, targetValue=hand_q[i])


def gen_grasp(tip_pose, pcd, kd_tree):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    projected_tip_pose = project_to_pcd(points, normals, tip_pose, kd_tree)
    vis_spheres = []
    for i in range(len(projected_tip_pose)-1):
        sp = o3d.geometry.TriangleMesh.create_sphere(0.01)
        sp.translate(projected_tip_pose[i,:3])
        vis_spheres.append(sp)
    return projected_tip_pose, vis_spheres

if __name__ == "__main__":
    box, kd_tree = create_o3d_box()
    grasp = np.array([[0.15, 0., 0.05],
                      [0.2, 0.05, -0.04],
                      [0.15, -0.05, -0.05]])
    projected_grasp, vis = gen_grasp(grasp, box, kd_tree)
    #o3d.visualization.draw_geometries([box]+vis)
    print("Start Checking Kin feasibility")
    flag, state = check_kin_feasible_parallel(projected_grasp[:,:3], projected_grasp[:,3:])
    print(f"Start Checking done {flag}")
    p.connect(p.GUI)
    obj = create_pybullet_box()
    hand = load_hand()
    set_pybullet_hand_state(state, hand)
    time.sleep(5)
    