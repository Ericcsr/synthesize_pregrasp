from cv2 import destroyAllWindows, threshold
import model.param as model_param
from model.param import *
import model.manipulation.scenario as scenario
import utils.math_utils as math_utils

import cvxpy as cp
from functools import partial
import numpy as np
import torch
import os
from scipy.spatial.transform import Rotation as R
from qpth.qp import QPFunction, QPSolvers
import copy

import open3d as o3d

########## Drake stuff ##########
import pydrake 
from pydrake.all import Solve
from pydrake.autodiffutils import ExtractValue, ExtractGradient, InitializeAutoDiff, AutoDiffXd
from pydrake.common import FindResourceOrThrow
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import Sphere, Box, Cylinder, CollisionFilterDeclaration, GeometrySet
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant, CoulombFriction
from pydrake.multibody.parsing import Parser
from pydrake.multibody.tree import JacobianWrtVariable, SpatialInertia, UnitInertia
import pydrake.solvers.mathematicalprogram as mp
import pydrake.perception as drake_perception
from pydrake.common.eigen_geometry import AngleAxis

# from pydrake.systems.analysis import Simulator
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer, MeshcatPointCloudVisualizer

def norm_cost(q):
    return q.dot(q)

class AllegroHandPlantDrake:
    def __init__(self, 
                 object_path=None, 
                 object_base_link_name=None,
                 object_world_pose=None,
                 threshold_distance=1.):
        self.num_fingers = 4
        self.dof_per_finger = 4
        self.threshold_distance=threshold_distance
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, MultibodyPlant(time_step=0.01))

        #drake_allegro_path = FindResourceOrThrow(model_param.drake_allegro_path)
        drake_allegro_path = model_param.allegro_hand_urdf_path
        parser = Parser(self.plant)
        self.hand_model = parser.AddModelFromFile(drake_allegro_path)
        # Add object
        if object_path is not None:
            drake_object_path = FindResourceOrThrow(object_path)
            parser.AddModelFromFile(drake_object_path)
            self.object_body = self.plant.GetBodyByName(object_base_link_name)
            self.object_frame = self.plant.GetFrameByName(object_base_link_name)
            if object_world_pose is None:
                raise NotImplementedError
            self.plant.WeldFrames(
                self.plant.world_frame(),
                # FIXME(wualbert): cleaner
                self.object_frame, object_world_pose)
            # Setup o3d stuff
            split_path = drake_object_path.split("/")
            drake_obj_name = split_path[-1].split(".")[0]
            drake_mesh_path = "/"+os.path.join(*split_path[:-1])+"/../meshes/"+drake_obj_name+"_textured.obj"
            o3d_mesh = o3d.io.read_triangle_mesh(drake_mesh_path)
            self.o3d_mesh = copy.deepcopy(o3d_mesh).transform(object_world_pose.GetAsMatrix4())
            self.o3d_mesh.compute_triangle_normals()
            self.o3d_mesh.compute_vertex_normals()
            assert self.o3d_mesh.has_triangle_normals() and self.o3d_mesh.has_vertex_normals()
            # assert self.o3d_mesh.is_watertight()
            self.o3d_scene = o3d.t.geometry.RaycastingScene()
            self.o3d_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.o3d_mesh))
        else:
            scenario.AddShape(
                self.plant, 
                Box(0.2 * 2, 0.2 * 2, 0.05 * 2),
                "manipulated_object",
                collidable=True,
                color=(0.6,0,0,0.8))
            self.object_body = self.plant.GetBodyByName("manipulated_object")
            self.object_frame = self.plant.GetFrameByName("manipulated_object")
            if object_world_pose is None:
                raise NotImplementedError
            self.plant.WeldFrames(
                self.plant.world_frame(),
                self.object_frame, object_world_pose)
            o3d_mesh = o3d.geometry.TriangleMesh.create_box(0.4, 0.4, 0.1)
            o3d_mesh.translate([-0.2, -0.2, -0.05])
            o3d_mesh.compute_triangle_normals()
            o3d_mesh.compute_vertex_normals()
            self.o3d_mesh = copy.deepcopy(o3d_mesh).transform(object_world_pose.GetAsMatrix4())
            self.o3d_scene = o3d.t.geometry.RaycastingScene()
            self.o3d_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.o3d_mesh))

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
        for i, finger in enumerate(AllAllegroHandFingers):
            X_FT = pydrake.math.RigidTransform(p=AllegroHandFingertipDrakeLinkOffset[finger])
            self.fingertip_bodies[finger] = self.plant.GetBodyByName(f"link_{AllegroHandFingertipDrakeLink[finger]}")
            self.fingertip_frames[finger] = pydrake.multibody.tree.FixedOffsetFrame_[float](
                name=f"fingertip_{finger}",
                P=self.fingertip_bodies[finger].body_frame(),
                X_PF=X_FT)
            self.plant.AddFrame(self.fingertip_frames[finger])
        
        # # Make the fingertips non-colllidable
        for finger in model_param.ActiveAllegroHandFingers:
            fingertip_collsions = self.plant.GetCollisionGeometriesForBody(self.fingertip_bodies[finger])            
            # Extract the collsions
            for fc in fingertip_collsions:
                    if isinstance(self.scene_graph.model_inspector().GetShape(fc), Sphere):
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
        assert self.num_positions == 23
        self.base_quaternion_idx_start = self.hand_body.floating_positions_start()
        self.base_position_idx_start = self.hand_body.floating_positions_start() + 4
        self.finger_joints_idx = {}
        finger_map = [AllegroHandFinger.INDEX, AllegroHandFinger.MIDDLE, AllegroHandFinger.RING, AllegroHandFinger.THUMB]
        for finger in finger_map:
            self.finger_joints_idx[finger] = []
            for link_idx in range(AllegroHandFingertipDrakeLink[finger] - 3,
                                  AllegroHandFingertipDrakeLink[finger] + 1):
                link_name = f'joint_{link_idx}'
                self.finger_joints_idx[finger].append(
                    self.plant.GetJointByName(link_name).position_start()
                )
            self.finger_joints_idx[finger] = np.array(self.finger_joints_idx[finger], dtype=int)
        self.finger_joint_idx_start = self.plant.GetJointByName('joint_0').position_start()

    def get_bullet_hand_config_from_drake_q(self, q):
        """
        Note that Drake quaternions are [w, x, y, z], while Pybullet
        quaternions are [x, y, z, w]
        Moreover, the finger ordering is different in drake and Pybullet
        Finally, there is an offset between Drake and Pybullet's base of
        allegro_drake_pybullet_base_offset
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
        finger_map = [AllegroHandFinger.INDEX, AllegroHandFinger.MIDDLE, AllegroHandFinger.RING, AllegroHandFinger.THUMB]
        for finger in finger_map:
            hand_q += q[self.finger_joints_idx[finger]].tolist() + [0]
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
                       self.base_quaternion_idx_start+4]
        finger_angles = {}
        finger_map = [AllegroHandFinger.INDEX, AllegroHandFinger.MIDDLE, AllegroHandFinger.RING, AllegroHandFinger.THUMB]
        for finger in finger_map:
            finger_angles[finger] = q[self.finger_joints_idx[finger]]
        return base_position, base_quaternion, finger_angles

    def convert_hand_configuration_to_q(self, base_position, base_quaternion, finger_angles):
        """
        base_quaternion: Follow Drake's convention
        finger_angles: Follow Drale's convention
        """
        q = np.zeros(self.plant.num_positions())
        q[self.base_position_idx_start:
          self.base_position_idx_start + 3] = base_position
        q[self.base_quaternion_idx_start:
          self.base_quaternion_idx_start + 4] = base_quaternion
        finger_map = [AllegroHandFinger.INDEX, AllegroHandFinger.MIDDLE, AllegroHandFinger.RING, AllegroHandFinger.THUMB]
        for finger in finger_map:
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
                                             finger_tip_poses,
                                             padding=model_param.object_padding,
                                             collision_distance=0.,
                                             has_normals=True,
                                             allowed_deviation=np.ones(3)*0.01,
                                             straight_unused_fingers=False,
                                             bounding_box=None):
        """
        By default, we assume finger_tip_poses as a np.ndarray [N_finger, 6]
        Finger order: THUMB, INDEX, MIDDLE, RING
        Assume Unused finger have pos [100, 100, 100]
        """
        _, plant_context = self.create_context()
        ik = InverseKinematics(self.plant, plant_context)
        unused_fingers = set(AllegroHandFinger)
        constraints_on_finger = {}
        desired_positions = {}

        finger_map = [AllegroHandFinger.THUMB, AllegroHandFinger.INDEX, AllegroHandFinger.MIDDLE, AllegroHandFinger.RING]
        for i, finger in enumerate(finger_map):
            contact_position = np.squeeze(finger_tip_poses[i, :3])
            if contact_position[0] > 50:
                continue
            if has_normals:
                contact_normal = finger_tip_poses[i, 3:]
                contact_normal /= np.linalg.norm(contact_normal)
                desired_position = contact_position + contact_normal * padding
            else:
                desired_position = contact_position
            desired_positions[finger] = desired_position
            constraints_on_finger[finger] = [ik.AddPositionConstraint(
                self.fingertip_frames[finger],
                np.zeros(3),
                self.plant.world_frame(),
                desired_position-allowed_deviation,
                desired_position+allowed_deviation
            )]
            # Add angle constraints
            # if has_normals:
            #     constraints_on_finger[finger].append(ik.AddAngleBetweenVectorsConstraint(self.plant.world_frame(),
            #                                      contact_normal,
            #                                      self.fingertip_frames[finger],
            #                                      np.array([0.,0.,-1.]),
            #                                      0., np.pi/3.))
            unused_fingers.remove(finger)

        prog = ik.get_mutable_prog()
        q = ik.q()
        prog.AddCost(norm_cost, vars = q)
        if straight_unused_fingers:
            for finger in unused_fingers:
                # keep unused fingers straight
                # FIXME(wualbert): why is this never satisfied?
                if finger != self.finger_map.THUMB:
                    constraints_on_finger[finger] = prog.AddBoundingBoxConstraint(
                        0., 0., q[self.finger_joints_idx[finger]])
        # Collision constraints
        if collision_distance is None:
            collision_constr = None
        else:
            collision_constr = ik.AddMinimumDistanceConstraint(collision_distance, self.threshold_distance)
        # Need to add base pose constraints
        if not (bounding_box is None):
            constraint_on_base = ik.AddPositionConstraint(
                                self.hand_drake_frame, 
                                np.zeros(3), 
                                self.bounding_box["upper"],
                                self.bounding_box["lower"])
            return ik, constraints_on_finger, collision_constr, constraint_on_base, desired_positions
        else:
            return ik, constraints_on_finger, collision_constr, desired_positions

    
