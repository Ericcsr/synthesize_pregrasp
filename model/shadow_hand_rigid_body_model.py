import os
from typing import Dict
import numpy as np
import model.param as model_param
from model.param import *
import model.manipulation.scenario as scenario

########## Drake stuff ##########
import pydrake 
from pydrake.common import FindResourceOrThrow
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import Capsule, Box, CollisionFilterDeclaration, GeometrySet
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.parsing import Parser


def norm_cost(q):
    return q.dot(q)

class ShadowHandPlantDrake:
    def __init__(self, 
                 object_path=None,
                 object_base_link_name=None,
                 object_world_pose=None,
                 object_creator = None,
                 threshold_distance=1.,
                 drake_model_path=model_param.shadow_hand_urdf_collision_path):
        self.num_fingers = 4
        self.dof_per_finger = 4
        self.threshold_distance=threshold_distance
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, MultibodyPlant(time_step=0.01))
        
        drake_shadow_path = drake_model_path
        parser = Parser(self.plant)
        self.hand_model = parser.AddModelFromFile(drake_shadow_path)
        # Add object
        if not (object_path is None):
            drake_object_path = FindResourceOrThrow(object_path)
            parser.AddModelFromFile(drake_object_path)
        elif not (object_creator is None):
            object_creator(self.plant)
        else:
            scenario.AddShape(self.plant, Box(0.2 * 2, 0.2 * 2, 0.05 * 2),
                object_base_link_name, collidable=True, color=(0.6,0,0,0.8))
        self.object_body = self.plant.GetBodyByName(object_base_link_name)
        self.object_frame = self.plant.GetFrameByName(object_base_link_name)
        if object_world_pose is None:
            raise NotImplementedError
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.object_frame, object_world_pose)

        self.hand_body = self.plant.GetBodyByName("hand_root")
        self.hand_drake_frame = self.hand_body.body_frame()
        self.hand_base_bullet_frame = pydrake.multibody.tree.FixedOffsetFrame_[float](
                name=f"pybullet_base",
                P=self.hand_drake_frame,
                X_PF=pydrake.math.RigidTransform(p=np.array([0, 0, 0])))
        self.plant.AddFrame(self.hand_base_bullet_frame)

        self.fingertip_frames = {}
        self.fingertip_bodies = {}
        self.fingertip_sphere_collisions = {}
        self.env_collisions = []
        for i, finger in enumerate(AllShadowHandFingers):
            X_FT = pydrake.math.RigidTransform(p=ShadowHandFingertipDrakeLinkOffset[finger])
            self.fingertip_bodies[finger] = self.plant.GetBodyByName(model_param.ShadowHandIndex2Link[finger])
            self.fingertip_frames[finger] = pydrake.multibody.tree.FixedOffsetFrame_[float](
                name=f"fingertip_{finger}",
                P=self.fingertip_bodies[finger].body_frame(),
                X_PF=X_FT)
            self.plant.AddFrame(self.fingertip_frames[finger])
        
        # # Make the fingertips non-colllidable
        for finger in model_param.ActiveShadowHandFingers:
            fingertip_collsions = self.plant.GetCollisionGeometriesForBody(self.fingertip_bodies[finger])            
            # Extract the collsions
            for fc in fingertip_collsions:
                    if isinstance(self.scene_graph.model_inspector().GetShape(fc), Capsule):
                        self.fingertip_sphere_collisions[finger] = fc
        all_collision_candidates = set()
        for c in self.scene_graph.model_inspector().GetCollisionCandidates():
            all_collision_candidates.add(c[0])
            all_collision_candidates.add(c[1])
        
        object_collision_candidates = set(self.plant.GetCollisionGeometriesForBody(self.object_body))
        non_object_collision_candidates = all_collision_candidates.difference(object_collision_candidates)
        hand_nontip_collision_candidates = non_object_collision_candidates.difference(
            self.fingertip_sphere_collisions.values())
        all_geometry_set = GeometrySet(self.scene_graph.model_inspector().GetAllGeometryIds())
        hand_nontip_geometry_set = GeometrySet(list(hand_nontip_collision_candidates))
        object_geometry_set = GeometrySet(list(object_collision_candidates))
        self.scene_graph.collision_filter_manager().Apply(CollisionFilterDeclaration().ExcludeWithin(all_geometry_set).
                                                          AllowBetween(hand_nontip_geometry_set,object_geometry_set).
                                                          AllowWithin(hand_nontip_geometry_set))
        # Don't allow hand to collide with itself
        self.plant.Finalize()
        self.diagram = builder.Build()
        diagram_context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, diagram_context)
        self.plant.SetPositions(
            plant_context, np.zeros(self.plant.num_positions(), ))
        self.num_positions = self.plant.num_positions()
        # Note the object is welded, so it has no dof
        assert self.num_positions == 7 + 22
        self.base_quaternion_idx_start = self.hand_body.floating_positions_start()
        self.base_position_idx_start = self.hand_body.floating_positions_start() + 4
        self.finger_joints_idx = {}
        finger_map = [ShadowHandFinger.THUMB, 
                      ShadowHandFinger.INDEX, 
                      ShadowHandFinger.MIDDLE, 
                      ShadowHandFinger.RING,
                      ShadowHandFinger.LITTLE]
        for finger in finger_map:
            self.finger_joints_idx[finger] = []
            for link_idx in range(1, 6 if finger in [ShadowHandFinger.THUMB, ShadowHandFinger.LITTLE] else 5):
                joint_name = f'{model_param.ShadowHandIndex2Name[finger]}_joint{link_idx}'
                self.finger_joints_idx[finger].append(
                    self.plant.GetJointByName(joint_name).position_start()
                )
            self.finger_joints_idx[finger] = np.array(self.finger_joints_idx[finger], dtype=int)
        self.finger_joint_idx_start = self.plant.GetJointByName('thumb_joint1').position_start()

    def get_bullet_hand_config_from_drake_q(self, q):
        """
        Note that Drake quaternions are [w, x, y, z], while Pybullet
        quaternions are [x, y, z, w]
        Moreover, the finger ordering is different in drake and Pybullet
        Finally, there is an offset between Drake and Pybullet's base of
        shadow_drake_pybullet_base_offset
        """
        # Return the base and position
        _, plant_context = self.create_context()
        self.plant.SetPositions(plant_context, q)
        X_WB = self.hand_base_bullet_frame.CalcPoseInWorld(plant_context)
        base_position = X_WB.translation()
        # q is in [w,x,y,z]
        base_quaternion = X_WB.rotation().ToQuaternion().wxyz()
        base_quaternion = base_quaternion[np.array([1, 2, 3, 0], dtype=int)]
        hand_q = [0]
        finger_map = [ShadowHandFinger.THUMB,
                      ShadowHandFinger.INDEX, 
                      ShadowHandFinger.MIDDLE, 
                      ShadowHandFinger.RING, 
                      ShadowHandFinger.LITTLE]
        for finger in finger_map:
            hand_q += q[self.finger_joints_idx[finger]].tolist()
        assert len(hand_q) == 23
        return base_position, base_quaternion, np.array(hand_q)

    def create_context(self):
        """
        This is a syntax sugar function. It creates a new pair of diagram
        context and plant context.
        """
        diagram_context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, diagram_context)
        return (diagram_context, plant_context)

    # Which can be used for initial guess
    def convert_q_to_hand_configuration(self, q):
        """
        Note that Drake's quaternion is o
        """
        base_position = q[self.base_position_idx_start:
                       self.base_position_idx_start+3]
        base_quaternion = q[self.base_quaternion_idx_start:
                       self.base_quaternion_idx_start+4][[1,2,3,0]]
        finger_angles = {}
        finger_map = [ShadowHandFinger.THUMB, 
                      ShadowHandFinger.INDEX, 
                      ShadowHandFinger.MIDDLE, 
                      ShadowHandFinger.RING,
                      ShadowHandFinger.LITTLE]
        for finger in finger_map:
            finger_angles[finger] = q[self.finger_joints_idx[finger]]
        return base_position, base_quaternion, finger_angles

    def convert_hand_configuration_to_q(self, base_position, base_quaternion, finger_angles):
        """
        base_quaternion: Follow Bullet's convention
        finger_angles: Follow Drake's convention
        """
        q = np.zeros(self.plant.num_positions())
        q[self.base_position_idx_start:
          self.base_position_idx_start + 3] = base_position
        q[self.base_quaternion_idx_start:
          self.base_quaternion_idx_start + 4] = base_quaternion[[3,0,1,2]]
        finger_map = [ShadowHandFinger.THUMB, 
                      ShadowHandFinger.INDEX, 
                      ShadowHandFinger.MIDDLE, 
                      ShadowHandFinger.RING,
                      ShadowHandFinger.LITTLE]
        for finger in finger_map:
            q[self.finger_joints_idx[finger]] = finger_angles[finger]
        return q

    def get_joint_limits(self):
        # TODO: The Joint limits for hand base may be problematic
        lb = self.plant.GetPositionLowerLimits()
        ub = self.plant.GetPositionUpperLimits()
        return lb, ub

    def construct_ik_given_fingertip_normals(self,
                                             finger_tip_poses,
                                             padding=model_param.object_padding,
                                             collision_distance=0.,
                                             has_normals=True,
                                             allowed_deviation=np.ones(3)*0.01,
                                             straight_unused_fingers=True):
        """
        By default, we assume finger_tip_poses as a np.ndarray [N_finger, 6]
        Finger order: THUMB, INDEX, MIDDLE, RING
        Assume Unused finger have pos [100, 100, 100]
        """
        _, plant_context = self.create_context()
        ik = InverseKinematics(self.plant, plant_context)
        unused_fingers = set(ActiveShadowHandFingers)
        constraints_on_finger = {}
        desired_positions = get_desired_position(finger_tip_poses, padding, has_normals)

        finger_map = [ShadowHandFinger.THUMB, 
                      ShadowHandFinger.INDEX, 
                      ShadowHandFinger.MIDDLE, 
                      ShadowHandFinger.RING]
        if len(finger_tip_poses)==5:
            finger_map.append(ShadowHandFinger.LITTLE)
        for i, finger in enumerate(finger_map):
            contact_position = np.squeeze(finger_tip_poses[i, :3])
            if contact_position[0]+contact_position[1]+contact_position[2] > 20:
                desired_positions[finger] = np.array([100., 100., 100.])
                continue
            constraints_on_finger[finger] = [ik.AddPositionConstraint(
                self.fingertip_frames[finger],
                np.zeros(3),
                self.plant.world_frame(),
                desired_positions[finger]-allowed_deviation,
                desired_positions[finger]+allowed_deviation
            )]
            # Add angle constraints
            if has_normals:
                contact_normal = finger_tip_poses[i, 3:]
                contact_normal /= np.linalg.norm(contact_normal)
                constraints_on_finger[finger].append(ik.AddAngleBetweenVectorsConstraint(self.plant.world_frame(),
                                                 contact_normal,
                                                 self.fingertip_frames[finger],
                                                 np.array([0.,0.,-1.]),
                                                 0, np.pi/2))
            unused_fingers.remove(finger)

        prog = ik.get_mutable_prog()
        q = ik.q()
        prog.AddCost(norm_cost, vars = q)
        if straight_unused_fingers:
            for finger in unused_fingers:
                # if finger != ShadowHandFinger.THUMB:
                constraints_on_finger[finger] = prog.AddBoundingBoxConstraint(
                    DEFAULT_SHADOW[finger]-DEFAULT_JOINT_ALLOWANCE, 
                    DEFAULT_SHADOW[finger]+DEFAULT_JOINT_ALLOWANCE, 
                    q[self.finger_joints_idx[finger]])
                # else:
                #     constraints_on_finger[finger] = None
        else:
            for finger in unused_fingers:
                constraints_on_finger[finger] = None
        # Collision constraints
        if collision_distance is None:
            collision_constr = None
        else:
            collision_constr = ik.AddMinimumDistanceConstraint(collision_distance, self.threshold_distance)
        # Need to add base pose constraints
        return ik, constraints_on_finger, collision_constr, desired_positions

    
def get_desired_position(finger_tip_poses, padding, has_normals)->Dict:
    """
    finger_tip_poses: dictionary of finger tip poses expressed in world coordinate
    return a dictionary of modified tip position in the world coordinate
    """
    desired_positions = {}
    finger_map = [ShadowHandFinger.THUMB, 
                    ShadowHandFinger.INDEX, 
                    ShadowHandFinger.MIDDLE, 
                    ShadowHandFinger.RING]
    if len(finger_tip_poses)==5:
        finger_map.append(ShadowHandFinger.LITTLE)
    for i, finger in enumerate(finger_map):
        contact_position = np.squeeze(finger_tip_poses[i, :3])
        if contact_position[0] > 50:
            desired_positions[finger] = np.array([100., 100., 100.])
            continue
        if has_normals:
            contact_normal = finger_tip_poses[i, 3:].copy()
            contact_normal /= np.linalg.norm(contact_normal)
            desired_position = contact_position + contact_normal * padding
        else:
            desired_position = contact_position
        desired_positions[finger] = desired_position
    return desired_positions
