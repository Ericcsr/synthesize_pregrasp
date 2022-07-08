import numpy as np

class SmallBlockRegionDummy:
    def __init__(self, contact_state_graph):
        self.x_range = [-0.2, 0.2]
        self.y_range = [-0.2, 0.2]
        self.z_range = [-0.05, 0.05]
        self.csg = contact_state_graph
        region_data = np.load("../data/regions/small_block_dummy_region.npz")
        self.regions = region_data["regions"]
        self.fixed_axis = region_data["fixed_axes"]

    def parse_action(self, state_id, action):
        """
        state_id: int id of contact state
        action: np.ndarray[4,2] each element is bounded within [-1,1]
        """
        assert (np.abs(action)<=1).all()
        state = self.csg.getState(state_id)-1 # Region ID
        print(state)
        finger_regions = self.regions[state]
        fixed_axes = self.fixed_axis[state]
        print(finger_regions)
        print(fixed_axes)
        scaled_action = (action+1) * 0.5 # Mapped to [0, 1]
        finger_tip_pos = []

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
                z = x_range * sub_a[1] + z_start
            else:
                x_range = region[1] - region[0]
                x_start = region[0]
                x = x_range * sub_a[0] + x_start
                y_range = region[3] - region[2]
                y_start = region[2]
                y = y_range * sub_a[0] + y_start
                z = region[4]
            finger_tip_pos.append(np.array([x,y,z]))
        return finger_tip_pos

    def parse_sub_action(self, state_id, finger_id, sub_action):
        region_id = self.csg.getState(state_id)[finger_id] - 1
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
            z = x_range * scaled_action[1] + z_start
        else:
            x_range = region[1] - region[0]
            x_start = region[0]
            x = x_range * scaled_action[0] + x_start
            y_range = region[3] - region[2]
            y_start = region[2]
            y = y_range * scaled_action[0] + y_start
            z = region[4]
        return [x,y,z]