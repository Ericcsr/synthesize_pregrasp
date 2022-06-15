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
from pydrake.symbolic import Variable
from pydrake.common import FindResourceOrThrow
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import DrakeVisualizer, Sphere, Box, CollisionFilterManager, CollisionFilterDeclaration, GeometrySet
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics, MinimumDistanceConstraint, PositionConstraint
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.multibody.tree import JacobianWrtVariable
import pydrake.solvers.mathematicalprogram as mp
from pydrake.systems.analysis import Simulator
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer

#################################
def norm_cost(q):
    return q.dot(q)

class AllegroHandPlantDrake:
    def __init__(self, 
                 object_path=None, 
                 object_base_link_name=None,
                 object_world_pose=None, 
                 meshcat_open_brower=True,
                 num_finger_tips=3, 
                 object_collidable=True, 
                 useFixedBase=False,
                 base_pose=None,
                 robot_path=model_param.allegro_hand_urdf_path,
                 baseName="hand_root",
                 baseOffset=model_param.allegro_hand_offset,
                 all_fingers = AllAllegroHandFingers,
                 interp_mode = False):
        self.interp_mode = interp_mode
        self.baseOffset = baseOffset
        self.all_fingers = all_fingers
        self.useFixedBase = useFixedBase
        if all_fingers == AllAllegroHandFingers:
            self.tip_drake_link_map = AllegroHandFingertipDrakeLink
            self.tip_offset_map = AllegroHandFingertipDrakeLinkOffset
            self.finger_map = AllegroHandFinger
            self.name_to_idx = NameToFingerIndex
        else:
            self.tip_drake_link_map = AllegroArmFingertipDrakeLink
            self.tip_offset_map = AllegroArmFingertipDrakeLinkOffset
            self.finger_map = AllegroArmFinger
            self.name_to_idx = NameToArmFingerIndex
        self.num_fingers = 4
        self.dof_per_finger = 4
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, MultibodyPlant(time_step=0.01))

        #robot_path = FindResourceOrThrow(robot_path)
        parser = Parser(self.plant)
        self.hand_model = parser.AddModelFromFile(robot_path)
        # Add object
        if object_path is not None:
            drake_object_path = FindResourceOrThrow(object_path)
            parser.AddModelFromFile(drake_object_path)
            self.object_body = self.plant.GetBodyByName(object_base_link_name)
            self.object_frame = self.plant.GetFrameByName(object_base_link_name)
        else:
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
            object_world_pose) # Object Frame cannot be changed .. which may be problematic

        self.hand_body = self.plant.GetBodyByName(baseName) # hand root
        self.hand_body_drake_frame = self.hand_body.body_frame()
        if useFixedBase:
            self.plant.WeldFrames(
                self.plant.world_frame(),
                self.hand_body_drake_frame, # Will the name change after using the arm?
                RigidTransform(p=np.zeros(3)) if base_pose==None else base_pose)
        
        self.hand_base_bullet_frame = pydrake.multibody.tree.FixedOffsetFrame_[float](
                name=f"pybullet_base",
                P=self.hand_body_drake_frame,
                X_PF=pydrake.math.RigidTransform(p=np.array(baseOffset[:3]))) #TODO: May need to change for whole arm
        self.plant.AddFrame(self.hand_base_bullet_frame)
        self.fingertip_frames = {}
        self.fingertip_bodies = {}
        self.fingertip_sphere_collision = {}
        for _, finger in enumerate(all_fingers):
            X_FT = pydrake.math.RigidTransform(p=self.tip_offset_map[finger])
            self.fingertip_bodies[finger] = self.plant.GetBodyByName(f"link_{self.tip_drake_link_map[finger]}")
            self.fingertip_frames[finger] = pydrake.multibody.tree.FixedOffsetFrame_[float](
                name=f"fingertip_{finger}",
                P=self.fingertip_bodies[finger].body_frame(),
                X_PF=X_FT)
            self.plant.AddFrame(self.fingertip_frames[finger])
        for _ in range(num_finger_tips):
            # Make the fingertips non-colllidable
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
        hand_nontip_geometry_set = GeometrySet(list(hand_nontip_collision_candidates))
        object_geometry_set = GeometrySet(list(object_collision_candidates))
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
        self.num_positions = self.plant.num_positions()
        # Note the object is welded, so it has no dof which we should do the same for base right? TODO: (Eric)
        assert self.num_positions == 23 + (5 if self.all_fingers == AllAllegroArmFingers else 0) - \
                        (7 if self.useFixedBase else 0)
        # print('MI',self.plant.GetModelInstanceName)
        # print('Joint 0 idx', self.plant.GetJointByName("joint_0").position_start())
        self.base_quaternion_idx_start = self.hand_body.floating_positions_start()
        self.base_position_idx_start = self.hand_body.floating_positions_start() + 4

        if self.all_fingers == AllAllegroArmFingers:
            self.arm_joints_idx = [self.plant.GetJointByName("shoulder_0").position_start(),
                                   self.plant.GetJointByName("shoulder_1").position_start(),
                                   self.plant.GetJointByName("elbow_0").position_start(),
                                   self.plant.GetJointByName("elbow_1").position_start(),
                                   self.plant.GetJointByName("wrist_0").position_start()]
        self.finger_joints_idx = {}
        for finger in self.finger_map:
            self.finger_joints_idx[finger] = []
            for link_idx in range(self.tip_drake_link_map[finger] - 3,
                                  self.tip_drake_link_map[finger] + 1):
                link_name = f'joint_{link_idx}'
                self.finger_joints_idx[finger].append(
                    self.plant.GetJointByName(link_name).position_start()
                )
            self.finger_joints_idx[finger] = np.array(self.finger_joints_idx[finger], dtype=int)
        self.finger_joint_idx_start = self.plant.GetJointByName('joint_0').position_start()

    def get_bullet_hand_config_from_drake_q(self, q,unused_fingers={}, rest_angles={}):
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
        for finger in self.finger_map:
            if finger in unused_fingers:
                if rest_angles=={}:
                    finger_angles[finger] = np.zeros_like(q[self.finger_joints_idx[finger]])
                else:
                    finger_angles[finger] = rest_angles[finger] # rest angle for finger not in contact
            else:
                finger_angles[finger] = q[self.finger_joints_idx[finger]]
        
        order = ['ifinger','mfinger','rfinger','thumb']
        bullet_joint_state = np.zeros(21+self.baseOffset[3])
        if self.all_fingers == AllAllegroArmFingers:
            for i in range(5):
                bullet_joint_state[i] = q[self.arm_joints_idx[i]]
        for i in range(4):
            bullet_joint_state[5*i+1+self.baseOffset[3]:5*i+5+self.baseOffset[3]] = finger_angles[self.name_to_idx[order[i]]]
        return base_position, base_quaternion, bullet_joint_state

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
        for finger in self.finger_map:
            finger_angles[finger] = q[self.finger_joints_idx[finger]]
        return base_position, base_quaternion, finger_angles

    def convert_hand_configuration_to_q(self, base_position, base_quaternion, finger_angles):
        q = np.zeros(self.plant.num_positions())
        q[self.base_position_idx_start:
          self.base_position_idx_start + 3] = base_position
        q[self.base_quaternion_idx_start:
          self.base_quaternion_idx_start + 4] = base_quaternion
        for finger in self.finger_map:
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
        ik = InverseKinematics(self.plant, plant_context, with_joint_limits=True)
        unused_fingers = set(self.finger_map)
        constraints_on_finger = {}
        desired_positions = {}
        for finger in finger_normal_map.keys():
            # TODO: permute finger
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
            if has_normals:
                constraints_on_finger[finger].append(ik.AddAngleBetweenVectorsConstraint(self.plant.world_frame(),
                                                 contact_normal,
                                                 self.fingertip_frames[finger],
                                                 np.array([0.,0.,-1.]),
                                                 0, np.pi/2.))
            unused_fingers.remove(finger)

        prog = ik.get_mutable_prog()
        q = ik.q()
        if not model_param.USE_SOFT_BOUNDING:
            prog.AddCost(norm_cost, vars = q) # plant_context is q
        for finger in unused_fingers:
            # keep unused fingers straight
            # FIXME(wualbert): why is this never satisfied? (Eric)Thumb never satisfied??
            if finger != self.finger_map.THUMB and not self.interp_mode:
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

    def get_joint_limits(self):
        joint_lower_limits = self.plant.GetPositionLowerLimits()
        joint_upper_limits = self.plant.GetPositionUpperLimits()
        return joint_lower_limits, joint_upper_limits

    def getNumDofs(self):
        return len(self.plant.GetPositionLowerLimits())