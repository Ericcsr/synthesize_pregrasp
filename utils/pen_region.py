import numpy as np
import open3d as o3d

class GroovePenRegion:
    def __init__(self,radius=0.0125, length=0.2):
        self.radius = radius
        self.length = length

    def parse_sub_action(self, state_id, finger_id, sub_action, csg): # state id can only be zero
        region_id = csg.getState(state_id)[finger_id] - 1 # start from 0
        scaled_action = (sub_action+1) * 0.5 # Mapped to [0, 1]
        if region_id == 0:
            r = self.radius * np.sqrt(scaled_action[0])
            theta = scaled_action[1] * 2*np.pi
            y = -r * np.sin(theta)
            z = -r * np.cos(theta)
            x = self.length/2
            surface_norm = np.array([1.0,0.0,0.0])
        elif region_id == 1:
            x = scaled_action[0] * self.length - self.length/2
            theta = scaled_action[1] * np.pi
            y = -self.radius * np.sin(theta)
            z = -self.radius * np.cos(theta)
            surface_norm = np.array([y,z,0])
            surface_norm /= np.linalg.norm(surface_norm)
        elif region_id == 2:
            x = scaled_action[0] * self.length - self.length/2
            theta = scaled_action[1] * np.pi + np.pi
            y = -self.radius * np.sin(theta)
            z = -self.radius * np.cos(theta)
            surface_norm = np.array([y,z,0])
            surface_norm /= np.linalg.norm(surface_norm)
        elif region_id == 3:
            r = self.radius * np.sqrt(scaled_action[0])
            theta = scaled_action[1] * 2*np.pi
            y = -r * np.sin(theta)
            z = -r * np.cos(theta)
            x = -self.length/2
            surface_norm = np.array([-1.0,0.0,0.0])
        return np.array([x,y,z]), surface_norm
