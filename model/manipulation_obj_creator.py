import model.param as model_param
import model.manipulation.scenario as scenario
from pydrake.geometry import Box, Sphere
from pydrake.common import eigen_geometry
from pydrake.math import RigidTransform

def create_laptop_env(plant, has_floor = False):
    scenario.AddShape(plant, Box(0.4, 0.4, 0.1), "manipulated_object",
                      collidable=True)
    if has_floor:
        scenario.AddShape(plant, Box(2.0, 2.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.012])) # Add slightly more clearance to prevent problem in collision detection

def create_book_shelf_env(plant, has_floor = False):
    scenario.AddShape(plant, Box(0.4, 0.4, 0.1), "manipulated_object",
                     colliable=True)
    scenario.AddShape(plant, Box(0.4, 0.4, 0.1), "left_book",
                      collidable=True)
    scenario.AddShape(plant, Box(0.4, 0.4, 0.1), "right_book",
                      collidable=True)

    # Pre-fix coordinates of left and right book
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("left_book"),
                     RigidTransform(quaternion=eigen_geometry.Quaternion([0.7071068, 0, 0, 0.7071068]),
                                    p = [0., 0.11, 0.2]))
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("right_book"),
                     RigidTransform(quaternio=eigen_geometry.Quaternion([0.7071068, 0, 0, 0.7071068]),
                                    p = [0., -0.11, 0.2]))
    if has_floor:
        scenario.AddShape(plant, Box(2.0, 2.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.012])) # Add slightly more clearance to prevent problem in collision detection
