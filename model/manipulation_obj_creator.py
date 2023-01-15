import numpy as np
import model.param as model_param
import model.manipulation.scenario as scenario
from pydrake.geometry import Box, Sphere, Convex, Cylinder
from pydrake.common import eigen_geometry
from pydrake.math import RigidTransform
from pydrake.all import (SpatialInertia, UnitInertia, CoulombFriction)
from envs.scales import SCALES

#TODO: change scale of other environments
def create_foodbox_env(plant, has_floor = False, scale=SCALES["foodbox"]):
    scenario.AddShape(plant, Box(0.4*scale[0], 0.4*scale[1], 0.1*scale[2]), "manipulated_object",
                      collidable=True)
    if has_floor:
        scenario.AddShape(plant, Box(10.0, 10.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.012])) # Add slightly more clearance to prevent problem in collision detection

def create_bookshelf_env(plant, has_floor = False, scale=SCALES["bookshelf"]): # Need to confirm whether this is right
    scenario.AddShape(plant, Box(0.4*scale[0], 0.4*scale[1], 0.1*scale[2]), "manipulated_object",
                     collidable=True)
    scenario.AddShape(plant, Box(0.4*scale[0], 0.4*scale[1], 0.1*scale[2]), "left_book",
                      collidable=True)
    scenario.AddShape(plant, Box(0.4*scale[0], 0.4*scale[1], 0.1*scale[2]), "right_book",
                      collidable=True)

    # Pre-fix coordinates of left and right book
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("left_book"),
                     RigidTransform(quaternion=eigen_geometry.Quaternion([0.7071068, 0.7071068, 0, 0]),
                                    p = [0., 0.11*scale[2], 0.2*scale[1]]))
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("right_book"),
                     RigidTransform(quaternion=eigen_geometry.Quaternion([0.7071068, 0.7071068, 0, 0]),
                                    p = [0., -0.11*scale[2], 0.2]))
    if has_floor:
        scenario.AddShape(plant, Box(2.0, 2.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.012])) # Add slightly more clearance to prevent problem in collision detection


def create_waterbottle_env(plant, has_floor = False, scale=SCALES["waterbottle"]):
    scenario.AddShape(plant, Cylinder(0.041,0.22), "manipulated_object",
                     collidable=True)
    scenario.AddShape(plant, Box(0.08, 0.05, 0.24), "left_bottle",
                      collidable=True)
    scenario.AddShape(plant, Box(0.08, 0.05, 0.24), "right_bottle",
                      collidable=True)

    scenario.AddShape(plant, Box(5,5,0.01), "ceiling",collidable=True)

    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("left_bottle"),
                     RigidTransform(p = [0., -0.075, 0.12]))
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("right_bottle"),
                     RigidTransform(p = [0., 0.075, 0.12]))
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("ceiling"),
                     RigidTransform(p= [0., 0., 0.30]))
    
    if has_floor:
        scenario.AddShape(plant, Box(2.0, 2.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.012])) # Add slightly more clearance to prevent problem in collision detection

def create_laptop_env(plant, has_floor=False, scale=SCALES["laptop"]):
    # scenario.AddShape(plant, Box(0.4*scale[0], 0.4*scale[1], 0.1*scale[2]), "manipulated_object", 
    #                   collidable=True)
    scenario.AddConvexMesh(plant, "model/resources/meshes/keyboard_cvx.obj", 
                           "manipulated_object",
                           collidable=True)
    if has_floor:
        scenario.AddShape(plant, Box(10.0, 10.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.012])) # Add slightly more clearance to prevent problem in collision detection

def create_plate_env(plant, has_floor=False, scale=SCALES["plate"]):
    # TODO: Add convex based mesh
    scenario.AddConvexMesh(plant, "model/resources/meshes/plate_cvx_simple.obj", 
                           "manipulated_object",
                           collidable=True)
    if has_floor:
        scenario.AddShape(plant, Box(10.0, 10.0, 0.5), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("floor"),
                         RigidTransform(p=[0., 0., -0.252]))

def create_handle_env(plant, has_floor=False):
    scenario.AddURDFModel(plant, "envs/assets/handle_floating.urdf", fixed_body_name="handleBody")
    scenario.AddURDFModel(plant, "envs/assets/handle_shell.urdf", fixed_body_name="root")
    if has_floor:
        scenario.AddShape(plant, Box(10.0, 10.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),plant.GetFrameByName("floor"), 
                         RigidTransform(p=[0.,0., -0.012]))

def create_groovepen_env(plant, has_floor=False,scale=SCALES["groovepen"]):
    #scenario.AddURDFModel(plant, "envs/assets/pen.urdf", fixed_body_name="manipulated_object")
    scenario.AddConvexMesh(plant, 
                           "model/resources/meshes/pen_cvx.obj",
                           "manipulated_object",
                           collidable=True)
    # Add groove
    scenario.AddShape(plant, Box(0.4,0.04,0.03), "groove_1", collidable=True)
    plant.WeldFrames(plant.world_frame(), 
                     plant.GetFrameByName("groove_1"),
                     RigidTransform(p=[0.2, -0.04, 0.03]))
    scenario.AddShape(plant, Box(0.4,0.04,0.03), "groove_2", collidable=True)
    plant.WeldFrames(plant.world_frame(),
                    plant.GetFrameByName("groove_2"),
                    RigidTransform(p=[0.2,0.04,0.03]))
    scenario.AddShape(plant, Box(0.04,0.12,0.03),"groove_3", collidable=True)
    plant.WeldFrames(plant.world_frame(),
                    plant.GetFrameByName("groove_3"),
                    RigidTransform(p=[-0.02,0,0.03]))
    if has_floor:
        scenario.AddShape(plant, Box(10.0, 10.0, 0.02), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(),plant.GetFrameByName("floor"), 
                         RigidTransform(p=[0.,0., -0.012]))

def create_ruler_env(plant, has_floor = False, scale=SCALES["ruler"]):
    scenario.AddShape(plant, Box(0.4*scale[0], 0.4*scale[1], 0.1*scale[2]), "manipulated_object",
                      collidable=True)
    if has_floor:
        scenario.AddShape(plant, Box(3,3,0.05), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("floor"),
                         RigidTransform(p=[0,1.48,-0.035]))

def create_cardboard_env(plant, has_floor=False, scale=SCALES["cardboard"]):
    scenario.AddShape(plant, Box(0.4*scale[0], 0.4*scale[1], 0.1*scale[2]), "manipulated_object",
                      collidable=True)
    if has_floor:
        scenario.AddShape(plant, Box(3,3,0.05), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("floor"),
                         RigidTransform(p=[0,1.48,-0.035]))

def create_keyboard_env(plant, has_floor=False, scale=SCALES["keyboard"]):
    scenario.AddConvexMesh(plant, "model/resources/meshes/keyboard_cvx.obj", 
                           "manipulated_object",
                           collidable=True)
    if has_floor:
        scenario.AddShape(plant, Box(3,3,0.05), "floor", collidable=True)
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("floor"),
                         RigidTransform(p=[-1, 0,-0.035]))
object_creators = {
    "foodbox":create_foodbox_env,
    "bookshelf":create_bookshelf_env,
    "laptop": create_laptop_env,
    "plate": create_plate_env,
    "waterbottle":create_waterbottle_env,
    "groovepen":create_groovepen_env,
    "ruler":create_ruler_env,
    "cardboard":create_cardboard_env,
    "keyboard":create_keyboard_env
}