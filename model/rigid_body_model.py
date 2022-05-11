import model.param as model_param
from model.param import *
import model.manipulation.scenario as scenario

import utils.math_utils as math_utils

import cvxpy as cp
from functools import partial
import numpy as np
import torch
import os
import pybullet as p
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from qpth.qp import QPFunction, QPSolvers

########## Drake stuff ##########
import pydrake
from pydrake.all import eq
from pydrake.autodiffutils import ExtractValue, ExtractGradient, InitializeAutoDiff, AutoDiffXd
from pydrake.common import FindResourceOrThrow
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import DrakeVisualizer, Sphere, Box, CollisionFilterManager, CollisionFilterDeclaration, GeometrySet
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.multibody.tree import JacobianWrtVariable
import pydrake.solvers.mathematicalprogram as mp
from pydrake.systems.analysis import Simulator
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer

#################################
def compute_pybullet_T_WL(_p, body_id, link_index,
                          computeForwardKinematics=True):
    """
    Obtain the link's pose in the world directly from Pybullet
    :param _p:
    :param body_id:
    :param link_index:
    :param computeForwardKinematics:
    :return:
    """
    if link_index == -1:
        link_state = _p.getBasePositionAndOrientation(body_id)
    else:
        link_state = _p.getLinkState(body_id, link_index,
                                     computeForwardKinematics)
    world_position = np.array(link_state[0])
    world_quaternion = np.array(link_state[1])
    T_world = math_utils.construct_T_from_position_quaterion(
        world_position, world_quaternion
    )
    return T_world

class Link:
    '''
    Wrapper for Pybullet's rigid body model
    '''

    def __init__(self, model, joint_index, pcollision_L=None):
        self.model = model
        self.joint_index = joint_index
        self._p = self.model._p
        joint_info = self._p.getJointInfo(self.model.body_id, self.joint_index)
        # Parse and memoize
        self.joint_name = joint_info[1]
        # Currently only support single-axis revolute joint
        assert(joint_info[2] ==
               p.JOINT_REVOLUTE or joint_info[2] == p.JOINT_FIXED)
        self.joint_type = joint_info[2]
        self.joint_limit = np.array([joint_info[8], joint_info[9]])
        self.link_name = joint_info[12]
        self.joint_axis = np.array(joint_info[13])
        self.parent_index = joint_info[-1]
        self.parent_frame_position = np.array(joint_info[14])
        self.parent_frame_quaternion = np.array(joint_info[15])
        # Construct the transforms
        self.T_PJ = \
            math_utils.construct_T_from_position_quaterion(
                self.parent_frame_position, self.parent_frame_quaternion
            )
        self.T_PJ[:3,:3] = self.T_PJ[:3,:3].T
        self.T_PJ_tensor = torch.tensor(self.T_PJ)
        link_info = self._p.getLinkState(self.model.body_id,
                                         self.joint_index,
                                         computeForwardKinematics=True)
        self.link_pcom_L = np.array(link_info[2])
        self.link_qcom_L = np.array(link_info[3])
        self.link_Tcom_L = math_utils.construct_T_from_position_quaterion(
            self.link_pcom_L, self.link_qcom_L)

        # Collision checking point expressed in the link frame
        # TODO: currently only support 1 collision checking point per link
        if pcollision_L is not None:
            pcollision_L = np.asarray(pcollision_L)
        self.pcollision_L = pcollision_L
        self.pcollision_L_tensor = None

    def compute_T_PL(self, joint_angle, p_L=None, q_L=None):
        """
        T @ parent = link, T is expressed in the world frame
        :param joint_angle:
        :param p_L:
        :param q_L:
        :return: T
        """
        if p_L is None and q_L is None:
            link_Tcom_L = self.link_Tcom_L
        else:
            link_Tcom_L = math_utils.construct_T_from_position_quaterion(
                p_L, q_L)
        joint_T = math_utils.compute_joint_T(joint_angle,
                                             self.joint_axis)
        if type(joint_angle) == torch.Tensor:
            return (self.T_PJ_tensor.type_as(joint_angle) @
                    joint_T.type_as(joint_angle) @
                    torch.tensor(link_Tcom_L).type_as(joint_angle)).type_as(joint_angle)
        else:
            return self.T_PJ @ joint_T @ link_Tcom_L# @ np.linalg.pinv(base_rotation_T)

    def compute_p_P_collision(self, joint_angle):
        if isinstance(joint_angle, torch.Tensor):
            if self.pcollision_L_tensor is None:
                self.pcollision_L_tensor = torch.tensor(
                    self.pcollision_L, device=joint_angle.device).type(
                    joint_angle.dtype)
            return self.compute_T_PL(joint_angle, self.pcollision_L_tensor,
                                     torch.tensor([0.,0.,0.,1.],
                                                  device=joint_angle.device).type(joint_angle.dtype))
        else:
            return self.compute_T_PL(joint_angle,self.pcollision_L, np.array([0.,0.,0.,1.]))

class AllegroHandModel:
    def __init__(self, _p, urdf_path=None,
                 base_position=None, base_quaternion=None,
                 backend=AllegroHandBackend.BULLET,
                 time_step=1. / 240,
                 collision_points_dict = None):
        if urdf_path is None:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            urdf_path = current_dir + '/' + model_param.allegro_urdf_path
        self.urdf_path = urdf_path
        self.backend = backend
        self._p = _p
        self._ts = time_step
        self.body_id = self._p.loadURDF(urdf_path, useFixedBase=False)
        # sdf_path = pydrake.getDrakePath()+"/manipulation/models/"\
        # "allegro_hand_description/sdf/allegro_hand_description_right.sdf"
        # print(sdf_path)
        # self.body_id = self._p.loadSDF(sdf_path)[0]
        # Create constraints to fix base
        if base_position is None:
            base_position = np.array([0., 0., 0.])
        if base_quaternion is None:
            base_quaternion = np.array([0., 0., 0., 1.])
        self.base_pose_constraint_id = None
        self.set_base_pose(base_position, base_quaternion)
        self.num_fingers = 4
        self.dof_per_finger = 4
        # Memoize transformations
        self.links = {}
        self.collision_points_dict = collision_points_dict
        # Traverse the hand to get information
        for joint_id in range(self._p.getNumJoints(self.body_id)):
            if self.collision_points_dict is not None:
                pcollision = self.collision_points_dict[joint_id] if joint_id in self.collision_points_dict else None
                self.links[joint_id] = Link(self, joint_id, pcollision)
            else:
                self.links[joint_id] = Link(self, joint_id)

    def get_bullet_fingertip_p_WT(self, q):
        ans = {}
        diagram_context, plant_context = self.create_context()
        self.plant.SetPositions(plant_context, q)
        for finger in AllegroHandFinger:
            X_WT = self.fingertip_frame[finger].CalcPoseInWorld(plant_context)
            ans[finger] = X_WT.translation()
        return ans

    def set_plant_context_with_drake_q(self, q):
        diagram_context, plant_context = self.create_context()
        self.plant.SetPositions(plant_context, q)
        return diagram_context, plant_context

    def get_finger_joint_limits(self, finger):
        ans = np.zeros([self.dof_per_finger, 2])
        joint_id = finger.value-1
        for i in range(self.num_fingers-1, -1, -1):
            ans[i, :] = self.links[joint_id].joint_limit
            joint_id = self.links[joint_id].parent_index
        assert joint_id == -1
        return ans

    def set_base_pose(self, base_position, base_quaternion,
                      step_simulation=True):
        # TODO(wualbert): find a better implementation,
        #  perhaps with changeConstraint?
        if self.base_pose_constraint_id is not None:
            self._p.removeConstraint(self.base_pose_constraint_id)
        # reset the base pose
        self._p.resetBasePositionAndOrientation(self.body_id, base_position, base_quaternion)
        # FIXME(wualbert): constraint addition is broken
        self.base_pose_constraint_id = self._p.createConstraint(
            self.body_id, -1,
            -1, -1, jointType=p.JOINT_FIXED,
            jointAxis=[
                0, 0, 0],
            parentFramePosition=[
                0, 0, model_param.allegro_base_inertia_offset],
            childFramePosition=base_position,
            childFrameOrientation=base_quaternion)
        if step_simulation:
            self._p.stepSimulation()

    def get_collision_points_world_positions(self, joint_angles, base_position,
                                      base_rotation_matrix):
        """
        :param joint_angles:
        :param base_position:
        :param base_rotation_matrix:
        :return: num_finger x 3 tensor representing the world position of
                the fingers
        """
        # joint angles should be provided in the order of the fingers
        # i.e. first 4 angles belong to first finger
        computed_finger_count = 0
        current_finger_i = 0
        if type(joint_angles) == torch.Tensor:
            ans = torch.zeros([len(self.collision_points_dict.keys()), 3])
            base_T = math_utils.construct_T_from_position_matrix(
                base_position, base_rotation_matrix)
            for ci, collision_link_id in enumerate(self.collision_points_dict.keys()):
                T = base_T.to(joint_angles.device).type(joint_angles.dtype)
                if collision_link_id == -1:
                    T = T @ math_utils.construct_T_from_position_matrix(
                        torch.tensor(self.collision_points_dict[collision_link_id], device=joint_angles.device).type(joint_angles.dtype),
                        torch.eye(3))
                else:
                    finger_i = collision_link_id//5
                    if finger_i > current_finger_i:
                        current_finger_i = finger_i
                        computed_finger_count += 1
                    link_i = collision_link_id%5
                    # finger_index_i is 4*num_fingers
                    for parent_link in range(link_i):
                        T = T @ self.links[parent_link+finger_i*5].compute_T_PL(
                            joint_angles[parent_link + computed_finger_count*4]
                        )
                    T = T @ self.links[collision_link_id].compute_p_P_collision(
                        joint_angles[link_i + computed_finger_count * 4]
                    )
                ans[ci, :] = T[:3, 3]
        else:
            ans = np.zeros([len(self.collision_points_dict.keys()), 3])
            base_T = math_utils.construct_T_from_position_matrix(
                base_position, base_rotation_matrix)
            for ci, collision_link_id in enumerate(self.collision_points_dict.keys()):
                T = np.copy(base_T)
                if collision_link_id == -1:
                    T = T @ math_utils.construct_T_from_position_matrix(
                        self.collision_points_dict[collision_link_id],
                        np.eye(3))
                else:
                    finger_i = collision_link_id//5
                    if finger_i > current_finger_i:
                        current_finger_i = finger_i
                        computed_finger_count += 1
                    link_i = collision_link_id%5
                    for parent_link in range(link_i):
                        T = T @ self.links[parent_link+finger_i*5].compute_T_PL(
                            joint_angles[parent_link + computed_finger_count*4]
                        )
                    T = T @ self.links[collision_link_id].compute_p_P_collision(
                        joint_angles[link_i + computed_finger_count * 4]
                    )
                ans[ci, :] = T[:3, 3]
        return ans

    def visualize_collision_points(self):
        # Compute the collision points
        angle_indices = np.asarray([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13,
                                    15, 16, 17, 18], dtype=int)
        joint_angles = np.zeros(16)  # np.random.rand(16) * 0.2
        joint_angles_sim = self._p.getJointStates(self.body_id,
                                       angle_indices,
                                       )
        for f_idx in range(self.num_fingers):
            for j_idx in range(4):
                joint_angles[4*f_idx+j_idx] = joint_angles_sim[4*f_idx+j_idx][0]
        base_position, base_quaternion = self._p.getBasePositionAndOrientation(
            self.body_id)
        base_matrix = p.getMatrixFromQuaternion(
                base_quaternion)
        base_matrix = np.array(base_matrix).reshape([3, 3])
        collision_points = self.get_collision_points_world_positions(joint_angles, base_position, base_matrix)
        # Plot the collision points in Pybullet
        for cp in collision_points:
            self._visualize_crosshair_at_point(cp, lifeTime=1., color=(1.,0.,0.))

    def _visualize_crosshair_at_point(self, point, width=0.002, color=(1.,0.,0.), **kwargs):
        for idx in range(3):
            delta = np.zeros(3)
            delta[idx] = width
            self._p.addUserDebugLine(point-delta,
                                point+delta,
                                lineColorRGB=color, lineWidth=1., **kwargs)

class AllegroHandPlantDrake:
    def __init__(self, object_path=None, object_base_link_name=None,
                 object_world_pose=None, meshcat_open_brower=True,
                 num_viz_spheres=3, viz_sphere_colors=None, viz_sphere_radius=0.015,
                 object_collidable=True):
        self.num_fingers = 4
        self.dof_per_finger = 4
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, MultibodyPlant(time_step=0.01))

        drake_allegro_path = FindResourceOrThrow(model_param.drake_allegro_path)
        parser = Parser(self.plant)
        self.hand_model = parser.AddModelFromFile(drake_allegro_path)
        # Add object
        if object_path is not None:
            drake_object_path = FindResourceOrThrow(object_path)
            parser.AddModelFromFile(drake_object_path)
            self.object_body = self.plant.GetBodyByName(object_base_link_name)
            self.object_frame = self.plant.GetFrameByName(object_base_link_name)
        else:
            # Add primitive shape from scenario
            self.box = scenario.AddShape(
                self.plant, 
                Box(0.2 * SCALE *2, 0.2 * SCALE*2, 0.05 * SCALE*2),
                "manipulated_object",
                collidable=object_collidable,
                color=(0.6,0,0,0.8))
            self.object_body = self.plant.GetBodyByName("manipulated_object")
            self.object_frame = self.plant.GetFrameByName("manipulated_object")
        if object_world_pose is None:
            raise NotImplementedError
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.object_frame, 
            object_world_pose)
        self.hand_body = self.plant.GetBodyByName("hand_root") # hand root
        self.hand_body_drake_frame = self.hand_body.body_frame()
        # TODO: (Eric) The frame may be flipped wrt to drake's frame
        self.hand_base_bullet_frame = pydrake.multibody.tree.FixedOffsetFrame_[float](
                name=f"pybullet_base",
                P=self.hand_body_drake_frame,
                X_PF=pydrake.math.RigidTransform(p=np.array([0.0,0.0,0.145])))
        self.plant.AddFrame(self.hand_base_bullet_frame)
        self.fingertip_frames = {}
        self.fingertip_bodies = {}
        self.fingertip_sphere_collision = {}
        for i, finger in enumerate(AllAllegroHandFingers):
            X_FT = pydrake.math.RigidTransform(p=AllegroHandFingertipDrakeLinkOffset[finger])
            self.fingertip_bodies[finger] = self.plant.GetBodyByName(f"link_{AllegroHandFingertipDrakeLink[finger]}")
            self.fingertip_frames[finger] = pydrake.multibody.tree.FixedOffsetFrame_[float](
                name=f"fingertip_{finger}",
                P=self.fingertip_bodies[finger].body_frame(),
                X_PF=X_FT)
            self.plant.AddFrame(self.fingertip_frames[finger])
        self.viz_spheres = []
        self.viz_sphere_names = []
        self.viz_sphere_positions_idx_start = []
        for i in range(num_viz_spheres):
            # Add visualization spheres
            viz_name = f'viz_sphere_{i}'
            if viz_sphere_colors is not None:
                color = viz_sphere_colors[i]
            else:
                color = np.zeros(4)
                color[i%3] = 1.
                color[-1] = 1.
            # sphere = scenario.AddShape(self.plant, Sphere(0.015), viz_name, color=color, collidable=False)
            sphere = scenario.AddShape(self.plant, Sphere(viz_sphere_radius), viz_name, color=color, collidable=False)
            self.viz_spheres.append(sphere)
            self.viz_sphere_names.append(viz_name)
            # self.viz_spheres_frames[finger] = self.plant.GetBodyByName(viz_name, sphere).body_frame()
            # # Make the fingertips non-colllidable
            fingertip_collsions = self.plant.GetCollisionGeometriesForBody(self.fingertip_bodies[finger])
            # Extract the collsions
            for fc in fingertip_collsions:
                if isinstance(self.scene_graph.model_inspector().GetShape(fc), Sphere):
                    self.fingertip_sphere_collision[finger] = fc

        # Extract all collision candidates
        all_collision_candidates = set()
        for c in self.scene_graph.model_inspector().GetCollisionCandidates():
            all_collision_candidates.add(c[0])
            all_collision_candidates.add(c[1])

        object_collision_candidates = self.plant.GetCollisionGeometriesForBody(self.object_body)
        non_object_collision_candidates = all_collision_candidates.difference(object_collision_candidates)
        hand_nontip_collision_candidates = non_object_collision_candidates.difference(
            self.fingertip_sphere_collision.values())
        all_geometry_set = GeometrySet(self.scene_graph.model_inspector().GetAllGeometryIds())
        # other_geometry_set = GeometrySet(list(non_object_collision_candidates))
        hand_nontip_geometry_set = GeometrySet(list(hand_nontip_collision_candidates))
        object_geometry_set = GeometrySet(list(object_collision_candidates))
        # print(self.plant.GetCollisionGeometriesForBody(self.object_body))
        # print('col', list(self.fingertip_spheres.values()))
        # self.scene_graph.collision_filter_manager().Apply(CollisionFilterDeclaration().ExcludeWithin(all_geometry_set).AllowBetween(hand_nontip_geometry_set,object_geometry_set))
        self.scene_graph.collision_filter_manager().Apply(CollisionFilterDeclaration().ExcludeWithin(all_geometry_set).
                                                          AllowBetween(hand_nontip_geometry_set,object_geometry_set).
                                                          AllowWithin(hand_nontip_geometry_set))
        # Don't allow hand to collide with itself
        self.plant.Finalize()
        if not meshcat_open_brower:
            self.meshcat_viz = None
        else:
            self.meshcat_viz = ConnectMeshcatVisualizer(builder, self.scene_graph,
                                                        open_browser=meshcat_open_brower)

        self.diagram = builder.Build()
        Simulator(self.diagram).Initialize()

        diagram_context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, diagram_context)
        self.plant.SetPositions(
            plant_context, np.zeros(self.plant.num_positions(), ))
        # Store the indices of the visualization spheres
        for _, name in enumerate(self.viz_sphere_names):
            self.viz_sphere_positions_idx_start.append(self.plant.GetBodyByName(name).floating_positions_start()+4)
        # print('FB', self.plant.GetFloatingBaseBodies())
        # body_idx = self.plant.GetFloatingBaseBodies().pop()
        # print('Body idx', body_idx)
        # body = self.plant.get_body(body_idx)
        # print('has_quat', body.has_quaternion_dofs())
        # print('float_start', body.floating_positions_start())
        # body_frame_id = self.plant.GetBodyFrameIdOrThrow(body_idx)
        self.num_positions = self.plant.num_positions()
        # Note the object is welded, so it has no dof
        assert self.num_positions == 23+7*len(self.viz_sphere_names)
        # print('MI',self.plant.GetModelInstanceName)
        # print('Joint 0 idx', self.plant.GetJointByName("joint_0").position_start())
        self.base_quaternion_idx_start = self.hand_body.floating_positions_start()
        self.base_position_idx_start = self.hand_body.floating_positions_start() + 4
        self.finger_joints_idx = {}
        for finger in AllegroHandFinger:
            self.finger_joints_idx[finger] = []
            for link_idx in range(AllegroHandFingertipDrakeLink[finger] - 3,
                                  AllegroHandFingertipDrakeLink[finger] + 1):
                link_name = f'joint_{link_idx}'
                self.finger_joints_idx[finger].append(
                    self.plant.GetJointByName(link_name).position_start()
                )
            self.finger_joints_idx[finger] = np.array(self.finger_joints_idx[finger], dtype=int)
        self.finger_joint_idx_start = self.plant.GetJointByName('joint_0').position_start()
        # print(self.plant.get_body(self.viz_spheres[AllegroHandFinger.MIDDLE]).floating_positions_start())

    def get_bullet_hand_config_from_drake_q(self, q,unused_fingers={}):
        """
        Note that Drake quaternions are [w, x, y, z], while Pybullet
        quaternions are [x, y, z, w]
        Moreover, the finger ordering is different in drake and Pybullet
        Finally, there is an offset between Drake and Pybullet's base of
        allegro_drake_pybullet_base_offset
        """
        # Return the base and position
        diagram_context, plant_context = self.create_context()
        self.plant.SetPositions(plant_context, q)
        X_WB = self.hand_base_bullet_frame.CalcPoseInWorld(plant_context)
        base_position = X_WB.translation()
        # q is in [w,x,y,z]
        base_quaternion = X_WB.rotation().ToQuaternion().wxyz()
        base_quaternion = base_quaternion[np.array([1, 2, 3, 0], dtype=int)]
        
        finger_angles = {}
        for finger in AllegroHandFinger:
            if finger in unused_fingers:
                finger_angles[finger] = np.zeros_like(q[self.finger_joints_idx[finger]])
            else:
                finger_angles[finger] = q[self.finger_joints_idx[finger]]
        
        order = ['ifinger','mfinger','rfinger','thumb']
        bullet_joint_state = np.zeros(20)
        for i in range(4):
            bullet_joint_state[5*i:5*i+4] = finger_angles[NameToFingerIndex[order[i]]]
        return base_position, base_quaternion, bullet_joint_state

    def populate_q_with_viz_sphere_positions(self, positions, q):
        for i in range(len(positions)):
            start_index = self.viz_sphere_positions_idx_start[i] # quaternion excluded
            # Populate the quaternions so it's not all zeros
            q[start_index-4] = 1.
            q[start_index:start_index+3] = positions[i]
        return None

    def create_context(self):
        """
        This is a syntax sugar function. It creates a new pair of diagram
        context and plant context.
        """
        diagram_context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, diagram_context)
        return (diagram_context, plant_context)

    def convert_q_to_hand_configuration(self, q):
        """
        Note that Drake's quaternion is o
        """
        base_position = q[self.base_position_idx_start:
                       self.base_position_idx_start+3]
        base_quaternion = q[self.base_quaternion_idx_start:
                       self.base_quaternion_idx_start+4]
        finger_angles = {}
        for finger in AllegroHandFinger:
            finger_angles[finger] = q[self.finger_joints_idx[finger]]
        return base_position, base_quaternion, finger_angles

    def convert_hand_configuration_to_q(self, base_position, base_quaternion, finger_angles):
        q = np.zeros(self.plant.num_positions())
        q[self.base_position_idx_start:
          self.base_position_idx_start + 3] = base_position
        q[self.base_quaternion_idx_start:
          self.base_quaternion_idx_start + 4] = base_quaternion
        for finger in AllegroHandFinger:
            q[self.finger_joints_idx[finger]] = finger_angles[finger]
        return q

    def get_finger_joint_limits(self, finger):
        ans = np.zeros([self.dof_per_finger, 2])
        joint_id = finger.value - 1
        for i in range(self.num_fingers - 1, -1, -1):
            ans[i, :] = self.links[joint_id].joint_limit
            joint_id = self.links[joint_id].parent_index
        assert joint_id == -1
        return ans

    def construct_ik_given_fingertip_normals(self, 
                                             plant_context,
                                             finger_normal_map,
                                             padding=model_param.object_padding,
                                             collision_distance=0.,
                                             has_normals=True,
                                             allowed_deviation=np.ones(3)*0.01):
        ik = InverseKinematics(self.plant, plant_context)
        #     ik.AddPositionConstraint(
        # dut.plant.get_body(
        #     dut.finger_link2_indices[planar_gripper.Finger.kFinger1])
        # .body_frame(), np.zeros(3), dut.plant.world_frame(),
        # np.array([-0.5, -0.1, -0.1]), np.array([0.5, 0.1, 0.1]))
        unused_fingers = set(AllegroHandFinger)
        constraints_on_finger = {}
        desired_positions = {}
        for finger in finger_normal_map.keys():
            # TODO: permute finger
            # frame_name = f'link_{AllegroHandFingertipDrakeLink[finger]}'
            # # link_offset = AllegroHandFingertipDrakeLinkOffset[finger]
            # [:3] is position [3:] is the direction
            contact_position = np.squeeze(finger_normal_map[finger][:3])
            if has_normals:
                contact_normal = finger_normal_map[finger][3:]
                contact_normal /= np.linalg.norm(contact_normal)
                desired_position = contact_position + contact_normal * padding
            else:
                desired_position = contact_position
            desired_positions[finger] = desired_position
            constraints_on_finger[finger] = [ik.AddPositionConstraint(
                self.fingertip_frames[finger],
                np.zeros(3),
                self.plant.world_frame(),
                # desired_position,
                desired_position-allowed_deviation,
                desired_position+allowed_deviation
            )]
            # Add angle constraints
            if has_normals and False:
                constraints_on_finger[finger].append(ik.AddAngleBetweenVectorsConstraint(self.plant.world_frame(),
                                                 contact_normal,
                                                 self.fingertip_frames[finger],
                                                 np.array([0.,0.,-1.]),
                                                 0., np.pi/3.))
            unused_fingers.remove(finger)

        prog = ik.get_mutable_prog()
        q = ik.q()
        for finger in unused_fingers:
            # keep unused fingers straight
            # FIXME(wualbert): why is this never satisfied? (Eric)Thumb never satisfied??
            if finger != AllegroHandFinger.THUMB:
                constraints_on_finger[finger] = prog.AddBoundingBoxConstraint(
                    0., 0., q[self.finger_joints_idx[finger]])
        # Collision constraints
        if collision_distance is None:
            collision_constr = None
        else:
            collision_constr = ik.AddMinimumDistanceConstraint(collision_distance, 1.)
        return ik, constraints_on_finger, collision_constr, desired_positions

    def get_collision_points_world_positions(self, joint_angles, base_position,
                                             base_rotation_matrix):
        """
        :param joint_angles:
        :param base_position:
        :param base_rotation_matrix:
        :return: num_finger x 3 tensor representing the world position of
                the fingers
        """
        # joint angles should be provided in the order of the fingers
        # i.e. first 4 angles belong to first finger
        computed_finger_count = 0
        current_finger_i = 0
        if type(joint_angles) == torch.Tensor:
            ans = torch.zeros([len(self.collision_points_dict.keys()), 3])
            base_T = math_utils.construct_T_from_position_matrix(
                base_position, base_rotation_matrix)
            for ci, collision_link_id in enumerate(self.collision_points_dict.keys()):
                T = base_T.to(joint_angles.device).type(joint_angles.dtype)
                if collision_link_id == -1:
                    T = T @ math_utils.construct_T_from_position_matrix(
                        torch.tensor(self.collision_points_dict[collision_link_id], device=joint_angles.device).type(
                            joint_angles.dtype),
                        torch.eye(3))
                else:
                    finger_i = collision_link_id // 5
                    if finger_i > current_finger_i:
                        current_finger_i = finger_i
                        computed_finger_count += 1
                    link_i = collision_link_id % 5
                    # finger_index_i is 4*num_fingers
                    for parent_link in range(link_i):
                        T = T @ self.links[parent_link + finger_i * 5].compute_T_PL(
                            joint_angles[parent_link + computed_finger_count * 4]
                        )
                    T = T @ self.links[collision_link_id].compute_p_P_collision(
                        joint_angles[link_i + computed_finger_count * 4]
                    )
                ans[ci, :] = T[:3, 3]
        else:
            ans = np.zeros([len(self.collision_points_dict.keys()), 3])
            base_T = math_utils.construct_T_from_position_matrix(
                base_position, base_rotation_matrix)
            for ci, collision_link_id in enumerate(self.collision_points_dict.keys()):
                T = np.copy(base_T)
                if collision_link_id == -1:
                    T = T @ math_utils.construct_T_from_position_matrix(
                        self.collision_points_dict[collision_link_id],
                        np.eye(3))
                else:
                    finger_i = collision_link_id // 5
                    if finger_i > current_finger_i:
                        current_finger_i = finger_i
                        computed_finger_count += 1
                    link_i = collision_link_id % 5
                    for parent_link in range(link_i):
                        T = T @ self.links[parent_link + finger_i * 5].compute_T_PL(
                            joint_angles[parent_link + computed_finger_count * 4]
                        )
                    T = T @ self.links[collision_link_id].compute_p_P_collision(
                        joint_angles[link_i + computed_finger_count * 4]
                    )
                ans[ci, :] = T[:3, 3]
        return ans

    def visualize_collision_points(self):
        # Compute the collision points
        angle_indices = np.asarray([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13,
                                    15, 16, 17, 18], dtype=int)
        joint_angles = np.zeros(16)  # np.random.rand(16) * 0.2
        joint_angles_sim = self._p.getJointStates(self.body_id,
                                                  angle_indices,
                                                  )
        for f_idx in range(self.num_fingers):
            for j_idx in range(4):
                joint_angles[4 * f_idx + j_idx] = joint_angles_sim[4 * f_idx + j_idx][0]
        base_position, base_quaternion = self._p.getBasePositionAndOrientation(
            self.body_id)
        base_matrix = p.getMatrixFromQuaternion(
            base_quaternion)
        base_matrix = np.array(base_matrix).reshape([3, 3])
        collision_points = self.get_collision_points_world_positions(joint_angles, base_position, base_matrix)
        # Plot the collision points in Pybullet
        for cp in collision_points:
            self._visualize_crosshair_at_point(cp, lifeTime=1., color=(1., 0., 0.))

    def _visualize_crosshair_at_point(self, point, width=0.002, color=(1., 0., 0.), **kwargs):
        for idx in range(3):
            delta = np.zeros(3)
            delta[idx] = width
            self._p.addUserDebugLine(point - delta,
                                     point + delta,
                                     lineColorRGB=color, lineWidth=1., **kwargs)


def construct_default_allegro_model(mode=p.DIRECT, collidable=True):
    _p = bullet_client.BulletClient(mode)
    if not collidable:
        model = AllegroHandModel(_p)
    else:
        model = AllegroHandModel(_p, collision_points_dict=model_param.collision_points_dict)
    return _p, model
