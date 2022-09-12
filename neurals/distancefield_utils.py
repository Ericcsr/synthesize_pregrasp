import open3d as o3d
import numpy as np
import torch

class DistFieldEnv:
    def __init__(self, meshes):
        scene = o3d.t.geometry.RaycastingScene()
        for mesh in meshes:
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene.add_triangles(mesh_t)
        self.scene = scene

    def get_points_distance(self, points):
        """
        Assume points are expressed in world coordinate with occluded part removed.
        output should be concatenated to the pointcloud as new extra features.
        """
        if isinstance(points, torch.Tensor):
            points = points.numpy()
        return torch.from_numpy(self.scene.compute_distance(points.astype(np.float32)).numpy())

def create_laptop_df():
    floor = o3d.geometry.TriangleMesh.create_box(2,2,0.02)
    floor.translate([-1, -1, -0.02])
    return DistFieldEnv([floor])

def create_laptop_with_x_wall_df():
    floor = o3d.geometry.TriangleMesh.create_box(2,2,0.02)
    floor.translate([-1,-1,-0.02])
    wall = o3d.geometry.TriangleMesh.create_box(2,0.02,1)
    wall.translate([-1, -0.22, 0])
    return DistFieldEnv([floor, wall])

def create_laptop_with_y_wall_df():
    floor = o3d.geometry.TriangleMesh.create_box(2, 2, 0.02)
    floor.translate([-1, -1, -0.02])
    wall = o3d.geometry.TriangleMesh.create_box(0.02, 2, 1)
    wall.translate([-0.22, -1, 0])
    return DistFieldEnv([floor, wall])

def create_laptop_with_table():
    table = o3d.geometry.TriangleMesh.create_box(3, 3, 0.02)
    table.translate([-2.7, -1.5, -0.02])
    return DistFieldEnv([table])

def create_bookshelf_df():
    floor = o3d.geometry.TriangleMesh.create_box(2, 2, 0.02)
    floor.translate([-1, -1, -0.02])
    box_left = o3d.geometry.TriangleMesh.create_box(0.4, 0.1, 0.4)
    box_left.translate([-0.2, -0.16, 0])
    box_right = o3d.geometry.TriangleMesh.create_box(0.4, 0.1, 0.4)
    box_right.translate([-0.2, 0.06, 0])
    return DistFieldEnv([floor, box_right, box_left])

env_lists = [create_laptop_df, create_laptop_with_x_wall_df, create_laptop_with_y_wall_df, create_bookshelf_df, create_laptop_with_table]

if __name__ == "__main__":
    floor = o3d.geometry.TriangleMesh.create_box(2, 2, 0.02)
    floor.translate([-1, -1, -0.02])
    box_left = o3d.geometry.TriangleMesh.create_box(0.4, 0.1, 0.4)
    box_left.translate([-0.2, -0.16, 0])
    box_right = o3d.geometry.TriangleMesh.create_box(0.4, 0.1, 0.4)
    box_right.translate([-0.2, 0.06, 0])
    floor.compute_triangle_normals()
    floor.compute_vertex_normals()
    box_left.compute_triangle_normals()
    box_left.compute_vertex_normals()
    box_right.compute_triangle_normals()
    box_right.compute_vertex_normals()
    o3d.visualization.draw_geometries([floor, box_left, box_right])