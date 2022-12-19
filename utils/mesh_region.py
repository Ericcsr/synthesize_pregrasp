import numpy as np
import open3d as o3d

class MeshRegion:
    def __init__(self,mesh, mode="uniform"):
        """
        Assume mesh is o3d mesh, with well defined triangle normal vector
        """
        self.mode = mode
        self.mesh = mesh
        self.triangles = np.asarray(mesh.triangles)
        self.triangle_normals = np.asarray(mesh.triangle_normals)
        self.vertices = np.asarray(mesh.vertices)

    @staticmethod
    def calculate_barycentry_weight(x,y,z=None,method="uniform"):
        w = np.zeros(3)
        if method=="uniform":
            w[0] = 1 - np.sqrt(x)
            w[1] = np.sqrt(x)*(1-y)
            w[2] =  np.sqrt(x) * y
        elif method=="normalize":
            s = x+y+z
            w[0] = x/s
            w[1] = y/s
            w[2] = z/s
        elif method=="simple":
            w[0] = 1-x
            w[1] = x*(1-y)
            w[2] = x*y
        return w

    def parse_sub_action(self, state_id, finger_id, sub_action, csg):
        """Sub_action should have 2 element"""
        
        region_id = csg.getState(state_id)[finger_id] - 1
        scaled_action = (sub_action+1) * 0.5 # Mapped to [0, 1]
        region = self.triangles[region_id]
        vertex = self.vertices[region]
        # TODO: Barycentric interpolation based on weight
        result = np.zeros(3)
        w = self.calculate_barycentry_weight(x=scaled_action[0],
                                             y=scaled_action[1],
                                             z=None if self.mode!="normalize" else scaled_action[2],
                                             method=self.mode)
        result += vertex[0] * w[0]
        result += vertex[1] * w[1]
        result += vertex[2] * w[2]
        #print("Finger:",finger_id,"region:", region_id,"W:",w,"result:",result)
        return result, self.triangle_normals[region_id]    
