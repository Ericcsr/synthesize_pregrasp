import model.param as model_param
import utils.math_utils as math_utils

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R


def create_primitive_object(_p, mass, shape, dim, color=(0.6, 0, 0, 0.7),
                            collidable=True, base_position=(0, 0, 0),
                            base_quaternion=(0, 0, 0, 1)):
    '''

    :param _p:
    :param mass:
    :param shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    :param dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder,
                (radius) for sphere
    :param color:
    :param collidable:
    :param base_position: vec3 initial obj location
    :param base_quaternion: initial obj orientation
    :return:
    '''
    visual_shape_id = None
    collision_shape_id = -1
    if shape == p.GEOM_BOX:
        visual_shape_id = _p.createVisualShape(shapeType=shape,
                                               halfExtents=dim,
                                               rgbaColor=color)
        if collidable:
            collision_shape_id = _p.createCollisionShape(shapeType=shape,
                                                         halfExtents=dim)
    elif shape == p.GEOM_CYLINDER:
        visual_shape_id = p.createVisualShape(shape, dim[0], [1, 1, 1], dim[1],
                                              rgbaColor=color)
        if collidable:
            collision_shape_id = _p.createCollisionShape(shape, dim[0],
                                                         [1, 1, 1], dim[1])
    elif shape == p.GEOM_SPHERE:
        visual_shape_id = _p.createVisualShape(shape, radius=dim[0],
                                               rgbaColor=color)
        if collidable:
            collision_shape_id = _p.createCollisionShape(shape, radius=dim[0])

    sid = _p.createMultiBody(baseMass=mass,
                             baseInertialFramePosition=[0, 0, 0],
                             baseCollisionShapeIndex=collision_shape_id,
                             baseVisualShapeIndex=visual_shape_id,
                             basePosition=base_position,
                             baseOrientation=base_quaternion)
    # Constraint the location of the object

    return sid


class PrimitiveObject:
    def __init__(self, _p, base_position=(0, 0, 0),
                 base_quaternion=(0, 0, 0, 1),
                 fix_base=True,
                 **kwargs):
        self._p = _p
        kwargs['base_position'] = base_position
        kwargs['base_quaternion'] = base_quaternion
        self.object_id = create_primitive_object(_p, **kwargs)
        self.dim = kwargs['dim']
        if fix_base:
            self.base_pose_constraint_id = _p.createConstraint(
                self.object_id, -1,
                -1, -1, jointType=p.JOINT_FIXED,
                jointAxis=[
                    0, 1, 0],
                parentFramePosition=[
                    0, 0, 0],
                childFramePosition=base_position,
                childFrameOrientation=base_quaternion)
        self.normal_map = None
        self._p.stepSimulation()

    def get_normal_map(self, num_normals, no_step_simulation=False,
                       return_in_world_frame=False,
                       padding=model_param.object_padding):
        '''
        A normals map is a Nx6 vector containing the normals directions
        on the object. The first Nx3 are positions, the second Nx3 are
        unit vectors pointing outwards on the normals direction.
        :param num_normals:
        :return: num_normals x 6 matrix representing the normals map
        '''
        # For now, only use uniform sampling
        if self.normal_map is not None:
            if self.normal_map.shape[0] == num_normals:
                return self.normal_map
        # TODO(wualbert): better sampling strategy
        xyz = math_utils.fibonacci_sphere(num_normals)
        # FIXME(wualbert): sphere radius is arbitrary
        np.testing.assert_equal(xyz.shape, [num_normals, 3])
        normal_map = np.zeros([num_normals, 6])
        if not no_step_simulation:
            self._p.stepSimulation()
        # Start from object origin
        result = self._p.rayTestBatch(xyz, np.zeros(xyz.shape),
                                      parentObjectUniqueId=self.object_id,
                                      parentLinkIndex=-1, numThreads=0)
        assert(len(result) == num_normals)
        base_position, base_quaternion = \
            self._p.getBasePositionAndOrientation(
                self.object_id
            )
        base_position = np.array(base_position)
        base_quaternion = np.array(base_quaternion)
        base_rotation = R.from_quat(base_quaternion)
        for ri, r in enumerate(result):
            object_id, link_id, _, world_position, world_normal = r
            assert(object_id == self.object_id)
            assert(link_id == -1)
            world_position = np.array(world_position)
            world_normal = np.array(world_normal)
            if return_in_world_frame:
                normal_map[ri, :3] = world_position
                normal_map[ri, 3:] = world_normal
            else:
                normal_map[ri, :3] = world_position - base_position
                normal_map[ri, 3:] = base_rotation.inv().as_matrix() @\
                    world_normal
        return normal_map


class Floor(PrimitiveObject):
    def __init__(self, _p, half_length=10, half_width=10, center=(0., 0., 0.)):
        # Floor thickness = 0.1
        center = np.array(center)
        center[-1] = -0.1
        super(). __init__(_p, base_position=center,
                          mass=1e3,
                          base_quaternion=(0, 0, 0, 1),
                          color=(0.5, 0.5, 0.5, 0.5),
                          shape=p.GEOM_BOX,
                          dim=np.array([half_length, half_width, 0.1]),
                          fix_base=True)
