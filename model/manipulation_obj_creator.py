import numpy as np
import model.param as model_param
import model.manipulation.scenario as scenario
from pydrake.geometry import Box, Sphere, Convex, Cylinder
from pydrake.common import eigen_geometry
from pydrake.math import RigidTransform
from pydrake.all import (SpatialInertia, UnitInertia, CoulombFriction)

#TODO: change scale of other environments
def create_laptop_env(plant, has_floor = False, scale=[1,1,1]):
    scenario.AddShape(plant, Box(0.4*scale[0], 0.4*scale[1], 0.1*scale[2]), "manipulated_object",
                      collidable=True)
    if has_floor:
        scenario.AddShape(plant, Box(10.0, 10.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.012])) # Add slightly more clearance to prevent problem in collision detection

def create_bookshelf_env(plant, has_floor = False, scale=[1,1,1]): # Need to confirm whether this is right
    scenario.AddShape(plant, Box(0.4, 0.4, 0.1), "manipulated_object",
                     collidable=True)
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
                     RigidTransform(quaternion=eigen_geometry.Quaternion([0.7071068, 0, 0, 0.7071068]),
                                    p = [0., -0.11, 0.2]))
    if has_floor:
        scenario.AddShape(plant, Box(2.0, 2.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.012])) # Add slightly more clearance to prevent problem in collision detection


def create_waterbottle_env(plant, has_floor = False):
    scenario.AddShape(plant, Cylinder(0.045,0.22), "manipulated_object",
                     collidable=True)
    scenario.AddShape(plant, Cylinder(0.045,0.22), "left_bottle",
                      collidable=True)
    scenario.AddShape(plant, Cylinder(0.045,0.22), "right_bottle",
                      collidable=True)

    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("left_bottle"),
                     RigidTransform(p = [0., -0.1, 0.11]))
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("right_bottle"),
                     RigidTransform(p = [0., 0.1, 0.11]))
    
    if has_floor:
        scenario.AddShape(plant, Box(2.0, 2.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.012])) # Add slightly more clearance to prevent problem in collision detection


def create_tablebox_env(plant, has_floor=False):
    scenario.AddShape(plant, Box(0.4, 0.4, 0.1), "manipulated_object", 
                      collidable=True)
    if has_floor:
        scenario.AddShape(plant, Box(10.0, 10.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.012])) # Add slightly more clearance to prevent problem in collision detection

def create_wallbox_env(plant, has_floor=False):
    scenario.AddShape(plant, Box(0.4, 0.4, 0.1), "manipulated_object", 
                      collidable=True)
    if has_floor:
        scenario.AddShape(plant, Box(10.0, 10.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.012])) # Add slightly more clearance to prevent problem in collision detection

def create_plate_env(plant, has_floor=False):
    # TODO: Add convex based mesh
    scenario.AddConvexMesh(plant, model_param.plate_mesh_path, "manipulated_object",
                           collidable=True)
    if has_floor:
        scenario.AddShape(plant, Box(10.0, 10.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.012]))

def create_handle_env(plant, has_floor=False):
    scenario.AddURDFModel(plant, "envs/assets/handle_floating.urdf", fixed_body_name="handleBody")
    scenario.AddURDFModel(plant, "envs/assets/handle_shell.urdf", fixed_body_name="root")
    if has_floor:
        scenario.AddShape(plant, Box(10.0, 10.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),plant.GetFrameByName("floor"), 
                         RigidTransform(p=[0.,0., -0.012]))

def create_groovepen_env(plant, has_floor=False):
    scenario.AddURDFModel(plant, "envs/assets/pen.urdf")
    # Add groove
    scenario.AddShape(plant, Box(0.4,0.04,0.03), "groove_1", collidable=True)
    plant.WeldFrames(plant.world_frame(), 
                     plant.GetFrameByName("groove_1"),
                     RigidTransform(p=[0.2, -0.04, 0.03]))
    scenario.AddShape(plant, Box(0.4,0.04,0.03), "groove_2", collidable=True)
    plant.WeldFrame(plant.world_frame(),
                    plant.GetFrameByName("groove_2"),
                    RigidTransform(p=[0.2,0.04,0.03]))
    scenario.AddShape(plant, Box(0.04,0.12,0.03),"groove_3", collidable=True)
    plant.WeldFrame(plant.world_frame(),
                    plant.GetFrameByName("groove_3"),
                    RigidTransform(p=[-0.02,0,0.03]))
    if has_floor:
        scenario.AddShape(plant, Box(10.0, 10.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),plant.GetFrameByName("floor"), 
                         RigidTransform(p=[0.,0., -0.012]))


object_creators = {
    "laptop":create_laptop_env,
    "bookshelf":create_bookshelf_env,
    "tablebox": create_tablebox_env,
    "wallbox": create_wallbox_env,
    "plate": create_plate_env,
    "handle": create_handle_env,
    "waterbottle":create_waterbottle_env,
    "groovepen":create_groovepen_env
}