import numpy as np
import open3d as o3d
import pygeodesic.geodesic as geo
from sklearn.neighbors import NearestNeighbors

class SmallBlockRegionDummy:
    def __init__(self,region_data_path="data/regions/small_block_dummy_region.npz"):
        self.x_range = [-0.2, 0.2]
        self.y_range = [-0.2, 0.2]
        self.z_range = [-0.05, 0.05]
        region_data = np.load(region_data_path)
        self.regions = region_data["regions"]
        self.centers = np.zeros((len(self.regions),3))
        for i in range(len(self.regions)):
            self.centers[i,0] = self.regions[i,:2].mean()
            self.centers[i,1] = self.regions[i,2:4].mean()
            self.centers[i,2] = self.regions[i,4:].mean()
        self.fixed_axis = region_data["fixed_axes"]
        self.surface_norm = region_data["surface_norm"]

    def create_geodesic_agent(self):
        mesh_box = o3d.geometry.TriangleMesh.create_box(0.4,0.4,0.1)
        mesh_box.compute_triangle_normals()
        mesh_box.compute_triangle_normals()
        mesh_box.compute_vertex_normals()
        mesh_box.translate([-0.2,-0.2,-0.05])
        pcd_box = mesh_box.sample_points_poisson_disk(10000,use_triangle_normal=True)
        rec_mesh_box,_ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_box)
        self.vtx = rec_mesh_box.vertices
        faces = rec_mesh_box.triangles
        self.knn_searcher = NearestNeighbors(n_neighbors=1,algorithm="ball_tree").fit(self.vtx)
        self.geod_agent = geo.PyGeodesicAlgorithmExact(self.vtx, faces)

    def distance(self, point_a, point_b):
        _, index_a = self.knn_searcher.kneighbors(point_a.reshape(1,-1))
        _, index_b = self.knn_searcher.kneighbors(point_b.reshape(1,-1))
        index_a = index_a[0,0]
        index_b = index_b[0,0]
        distance,_ = self.geod_agent.geodesicDistance(index_a,index_b)
        return distance

    def construct_distance_matrix(self):
        distance_matrix = np.zeros((len(self.regions),len(self.regions)))
        for i in range(len(self.regions)):
            for j in range(i, len(self.regions)):
                if i!=j:
                    dist = self.distance(self.centers[i], self.centers[j])
                    distance_matrix[i,j] = dist
                    distance_matrix[j,i] = dist
        return distance_matrix

    def parse_action(self, state_id, action, csg):
        """
        state_id: int id of contact state
        action: np.ndarray[4,2] each element is bounded within [-1,1]
        """
        assert (np.abs(action)<=1).all()
        state = csg.getState(state_id)-1 # Region ID
        print(state)
        finger_regions = self.regions[state]
        fixed_axes = self.fixed_axis[state]
        print(finger_regions)
        print(fixed_axes)
        scaled_action = (action+1) * 0.5 # Mapped to [0, 1]
        finger_tip_pos = []
        finger_tip_norm = []

        for i,region in enumerate(finger_regions):
            sub_a = scaled_action[i]
            fixed_axis = fixed_axes[i]
            if fixed_axis == 0:
                x = region[0]
                y_range = region[3] - region[2]
                y_start = region[2]
                y = y_range * sub_a[0] + y_start
                z_range = region[5] - region[4]
                z_start = region[4]
                z = z_range * sub_a[1] + z_start
            elif fixed_axis == 1:
                x_range = region[1] - region[0]
                x_start = region[0]
                x = x_range * sub_a[0] + x_start
                y = region[2]
                z_range = region[5] - region[4]
                z_start = region[4]
                z = z_range * sub_a[1] + z_start
            else:
                x_range = region[1] - region[0]
                x_start = region[0]
                x = x_range * sub_a[0] + x_start
                y_range = region[3] - region[2]
                y_start = region[2]
                y = y_range * sub_a[1] + y_start
                z = region[4]
            finger_tip_pos.append(np.array([x,y,z]))
            finger_tip_norm.append(self.surface_norm[state[i]])
        return finger_tip_pos

    def parse_sub_action(self, state_id, finger_id, sub_action, csg):
        region_id = csg.getState(state_id)[finger_id] - 1
        scaled_action = (sub_action+1) * 0.5 # Mapped to [0, 1]
        fixed_axis = self.fixed_axis[region_id]
        region = self.regions[region_id]
        if fixed_axis == 0:
            x = region[0]
            y_range = region[3] - region[2]
            y_start = region[2]
            y = y_range * scaled_action[0] + y_start
            z_range = region[5] - region[4]
            z_start = region[4]
            z = z_range * scaled_action[1] + z_start
        elif fixed_axis == 1:
            x_range = region[1] - region[0]
            x_start = region[0]
            x = x_range * scaled_action[0] + x_start
            y = region[2]
            z_range = region[5] - region[4]
            z_start = region[4]
            z = z_range * scaled_action[1] + z_start
        else:
            x_range = region[1] - region[0]
            x_start = region[0]
            x = x_range * scaled_action[0] + x_start
            y_range = region[3] - region[2]
            y_start = region[2]
            y = y_range * scaled_action[1] + y_start
            z = region[4]
        return np.array([x,y,z]), self.surface_norm[region_id]

if __name__ == "__main__":
    test_region = SmallBlockRegionDummy(region_data_path="../data/regions/small_block_dummy_region.npz")
    test_region.create_geodesic_agent()
    distance_metrix = test_region.construct_distance_matrix()
    print(distance_metrix)
    
