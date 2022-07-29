import pybullet as p
import time
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
import pybullet_data
import cv2 as cv
import numpy as np
import utils.rigidBodySento as rb

class PyBulletRenderer:
    def __init__(self, width=640, height=440, showImage=True):
        self.windowName = "image"
        self.width = width
        self.height = height
        self.showImage = showImage
        if self.showImage:
            null_image = np.zeros((self.width, self.height))
            cv.imshow(self.windowName, null_image)
        self.yaw = 20
        self.pitch = 10
        self.distance = 8
        self.view_height = 0
        self.upAxisIndex=2
        self.projection_matrix = [
            1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
            -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0]
        if self.showImage:
            cv.createTrackbar("yaw", self.windowName, 0, 360, self.changeYaw)
            cv.createTrackbar("pitch", self.windowName, 0, 90, self.changePitch)
            cv.createTrackbar("Distance", self.windowName, 1, 20, self.changeDistance)
            cv.createTrackbar("Height", self.windowName,0, 10, self.changeHeight)
            cv.setTrackbarPos("yaw",self.windowName, 20)
            cv.setTrackbarPos("pitch",self.windowName, 10)
            cv.setTrackbarPos("Distance",self.windowName, 8)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Initialize others
        self.plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.reset()

    def reset(self):
        self.x_ind = rb.create_primitive_shape(p, 0, p.GEOM_CYLINDER, (0.005, 1.0), (0,0,0.6,0.8),False,(0.5,0,0), (0, 0.7071068, 0, 0.7071068))
        self.y_ind = rb.create_primitive_shape(p, 0, p.GEOM_CYLINDER, (0.005, 1.0), (0,0.6,0,0.8),False,(0,0.5,0), (0.7071068, 0, 0, 0.7071068))
        self.z_ind = rb.create_primitive_shape(p, 0, p.GEOM_CYLINDER, (0.005, 1.0), (0.6,0,0,0.8),False,(0,0,0.5))
        #self.ground = p.loadURDF("plane.urdf",useFixedBase=1)
    
    def changeYaw(self, value):
        self.yaw = value
    
    def changePitch(self, value):
        self.pitch = value

    def changeDistance(self, value):
        self.distance = value

    def changeHeight(self, value):
        self.view_height = value

    def render(self, blocking=False):
        image = np.zeros((self.width, self.height))
        while True:
            view_matrix = p.computeViewMatrixFromYawPitchRoll([0, 0, 0.1 * self.view_height], self.distance*0.1, self.yaw, self.pitch-45, 0, self.upAxisIndex)
            image = p.getCameraImage(self.width, 
                                    self.height, 
                                    view_matrix, 
                                    projectionMatrix=self.projection_matrix,
                                    shadow=1, 
                                    lightDirection=[1,1,1])[2]
            if self.showImage:
                cv.imshow(self.windowName, image)
                key = cv.waitKey(1)
                if not blocking or key == ord('q'):
                    break
            else:
                break
        return image
    
    def stop(self):
        p.unloadPlugin(self.plugin)

if __name__ == "__main__":
    import pytorch_kinematics as pk
    import helper
    import torch
    import math
    p.connect(p.DIRECT)
    renderer = PyBulletRenderer()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.loadURDF("plane.urdf", [0, 0, -1])
    hand = p.loadURDF("/home/sirius/sirui/contact_planning_dexterous_hand/model/resources/allegro_hand_description/urdf/allegro_arm.urdf", useFixedBase=1)
    

    
    while (p.isConnected()):
        for y in range(0, 360, 10):
            start = time.time()
            #p.stepSimulation()
            stop = time.time()
            print("stepSimulation %f" % (stop - start))
            start = time.time()
            p.resetJointState(hand, 3, math.sin(0.5*y/180 * math.pi))
            renderer.render()
            stop = time.time()
            print("renderImage %f" % (stop - start))
            cv.waitKey(1)
    renderer.stop()