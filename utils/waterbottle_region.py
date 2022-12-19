import numpy as np
import open3d as o3d

class WaterbottleRegion:
    def __init__(self,radius=0.045, length=0.22):
        self.radius = radius
        self.length = length

    def parse_sub_action(self, state_id, finger_id, sub_action, csg): # state id can only be zero
        region_id = csg.getState(state_id)[finger_id] - 1 # start from 0
        scaled_action = (sub_action+1) * 0.5 # Mapped to [0, 1]
        if region_id == 0:
            r = self.radius * np.sqrt(scaled_action[0])
            theta = scaled_action[1] * 2*np.pi
            x = -r * np.sin(theta)
            y = -r * np.cos(theta)
            z = self.length/2
            surface_norm = np.array([0.0,0.0,1.0])
        elif region_id == 1:
            z = scaled_action[0] * self.length - self.length/2
            theta = scaled_action[1] * np.pi
            x = -self.radius * np.sin(theta)
            y = -self.radius * np.cos(theta)
            surface_norm = np.array([x,y,0])
            surface_norm /= np.linalg.norm(surface_norm)
        elif region_id == 2:
            z = scaled_action[0] * self.length - self.length/2
            theta = scaled_action[1] * np.pi + np.pi
            x = -self.radius * np.sin(theta)
            y = -self.radius * np.cos(theta)
            surface_norm = np.array([x,y,0])
            surface_norm /= np.linalg.norm(surface_norm)
        elif region_id == 3:
            r = self.radius * np.sqrt(scaled_action[0])
            theta = scaled_action[1] * 2*np.pi
            x = -r * np.sin(theta)
            y = -r * np.cos(theta)
            z = -self.length/2
            surface_norm = np.array([0.0,0.0,-1.0])
        return np.array([x,y,z]), surface_norm
