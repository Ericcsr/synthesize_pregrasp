import numpy as np
import open3d as o3d
import pygeodesic.geodesic as geo
from sklearn.neighbors import NearestNeighbors

class SmallBlockRegionDummy:
    def __init__(self,scale,region_data_path="data/regions/small_block_dummy_region.npz"):
        region_data = np.load(region_data_path)
        self.regions = region_data["regions"] * scale.repeat(2)
        self.centers = np.zeros((len(self.regions),3))
        for i in range(len(self.regions)):
            self.centers[i,0] = self.regions[i,:2].mean()
            self.centers[i,1] = self.regions[i,2:4].mean()
            self.centers[i,2] = self.regions[i,4:].mean()
        self.fixed_axis = region_data["fixed_axes"]
        self.surface_norm = region_data["surface_norm"]

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

    def sample_points(self, region_id, n_samples=5):
        region = self.regions[region_id]
        fixed_axis = self.fixed_axis[region_id]

        results = np.zeros((n_samples, 3))
        coord = np.random.random((n_samples, 2))
        for i in range(n_samples):
            if fixed_axis == 0:
                results[i,0] = region[0]
                y_range = region[3] - region[2]
                y_start = region[2]
                results[i,1] = y_range * coord[i,0] + y_start
                z_range = region[5] - region[4]
                z_start = region[4]
                results[i,2] = z_range * coord[i,1] + z_start
            elif fixed_axis == 1:
                x_range = region[1] - region[0]
                x_start = region[0]
                results[i,0] = x_range * coord[i,0] + x_start
                results[i,1] = region[2]
                z_range = region[5] - region[4]
                z_start = region[4]
                results[i,2] = z_range * coord[i,1] + z_start
            else:
                x_range = region[1] - region[0]
                x_start = region[0]
                results[i,0] = x_range * coord[i,0] + x_start
                y_range = region[3] - region[2]
                y_start = region[2]
                results[i,1] = y_range * coord[i,1] + y_start
                results[i,2] = region[4]
        return results


if __name__ == "__main__":
    test_region = SmallBlockRegionDummy(region_data_path="../data/regions/small_block_dummy_region.npz")
    test_region.create_geodesic_agent()
    distance_metrix = test_region.construct_distance_matrix()
    print(distance_metrix)
    
