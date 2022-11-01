#  Copyright 2020 Stanford University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pybullet
import numpy as np


def create_primitive_shape(pb, mass, shape, dim, color=(0.6, 0, 0, 1), collidable=True, init_xyz=(0, 0, 0),
                           init_quat=(0, 0, 0, 1)):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder, (radius) for sphere
    # init_xyz vec3 being initial obj location, init_quat being initial obj orientation
    visual_shape_id = None
    collision_shape_id = -1
    if shape == pybullet.GEOM_BOX:
        visual_shape_id = pb.createVisualShape(shapeType=shape, halfExtents=dim, rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shapeType=shape, halfExtents=dim)
    elif shape == pybullet.GEOM_CYLINDER:
        visual_shape_id = pb.createVisualShape(shape, dim[0], [1, 1, 1], dim[1], rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
    elif shape == pybullet.GEOM_SPHERE:
        visual_shape_id = pb.createVisualShape(shape, radius=dim[0], rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shape, radius=dim[0])

    sid = pb.createMultiBody(baseMass=mass, baseInertialFramePosition=[0, 0, 0],
                             baseCollisionShapeIndex=collision_shape_id,
                             baseVisualShapeIndex=visual_shape_id,
                             basePosition=init_xyz, baseOrientation=init_quat)
    return sid


def get_link_com_xyz_orn(pb, body_id, link_id):
    # get the world transform (xyz and quaternion) of the Center of Mass of the link
    # We *assume* link CoM transform == link shape transform (the one you use to calculate fluid force on each shape)
    assert link_id >= -1
    if link_id == -1:
        link_com, link_quat = pb.getBasePositionAndOrientation(body_id)
    else:
        link_com, link_quat, *_ = pb.getLinkState(body_id, link_id, computeForwardKinematics=1)
    return list(link_com), list(link_quat)


def apply_external_world_force_on_local_point(pb, body_id, link_id, world_force, local_com_offset):
    link_com, link_quat = get_link_com_xyz_orn(pb, body_id, link_id)
    _, inv_link_quat = pybullet.invertTransform([0., 0, 0], link_quat)  # obj->world
    local_force, _ = pybullet.multiplyTransforms([0., 0, 0], inv_link_quat, world_force, [0, 0, 0, 1])
    pb.applyExternalForce(body_id, link_id, local_force, local_com_offset, flags=pybullet.LINK_FRAME)


def get_link_com_linear_velocity(pb, body_id, link_id):
    # get the Link CoM linear velocity in the world coordinate frame
    assert link_id >= -1
    if link_id == -1:
        vel, _ = pb.getBaseVelocity(body_id)
    else:
        vel = pb.getLinkState(body_id, link_id, computeLinkVelocity=1, computeForwardKinematics=1)[6]
    return list(vel)


def get_link_angular_velocity(pb, body_id, link_id):
    # get the Link angular velocity in the world coordinate frame
    assert link_id >= -1
    if link_id == -1:
        _, ang_v = pb.getBaseVelocity(body_id)
    else:
        ang_v = pb.getLinkState(body_id, link_id, computeLinkVelocity=1, computeForwardKinematics=1)[7]
    return list(ang_v)


def get_dim_of_box_shape(pb, body_id, link_id):
    # get the dimension (length-3 list of width, depth, height) of the input link
    # We *assume* each link is a box
    # p.getCollisionShapeData() might be useful to you
    # hint: Check out getCollisionShapeData function in PyBullet Tutorial Document
    dim = pb.getCollisionShapeData(body_id, link_id)[0][3]
    return dim
