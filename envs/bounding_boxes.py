import numpy as np
import open3d as o3d
from envs.scales import SCALES

BOUNDING_BOXES = {
    "bookshelf":[o3d.geometry.AxisAlignedBoundingBox(min_bound = np.array([-0.205, -0.045, 0.005]),max_bound = np.array([0.205, 0.045, 100])),
                 o3d.geometry.AxisAlignedBoundingBox(min_bound = np.array([0.205, -100, 0.005]),max_bound = np.array([100, 100, 100]))],
    "waterbottle":[o3d.geometry.AxisAlignedBoundingBox(min_bound = np.array([-0.04, -0.045, 0.005]),max_bound = np.array([0.05, 0.045, 100])),
                   o3d.geometry.AxisAlignedBoundingBox(min_bound = np.array([0.05, -100, 0.005]),max_bound = np.array([100, 100, 100]))],
    "laptop":[o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-0.11-0.2*SCALES["laptop"][0], -100, 0.005]),max_bound=np.array([100, 100, 100]))],
    "groovepen":[o3d.geometry.AxisAlignedBoundingBox(min_bound = np.array([-100, -100, 0.005]),max_bound = np.array([100, 100, 100]))],
    "cardboard":[o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-1.5, -0.02, 0.0005]),max_bound=np.array([1.5, 3.00, 100])),
                 o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-1.5, -100, -100]),max_bound=np.array([1.5, -0.02, 100]))],
    "ruler":[o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-1.5, -0.02, 0.0005]),max_bound=np.array([1.5, 3.00, 100])),
                 o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-1.5, -100, -100]),max_bound=np.array([1.5, -0.02, 100]))],
    "foodbox":[o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-100, -100, 0.005]),max_bound=np.array([100, 100, 100]))],
    "plate":[o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-100, -100, 0.005]),max_bound=np.array([100, 100, 100]))],
    "keyboard": [o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-0.31,-100,0.005]), max_bound=np.array([0.5,100,100])),
                 o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([0.5,-100,-100]), max_bound=np.array([100,100,100]))],
    "cshape":[o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-0.07, -100, 0.005]),max_bound=np.array([100, 100, 100]))],
    "tape":[o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-0.12, -100, 0.005]),max_bound=np.array([100, 100, 100]))]
}