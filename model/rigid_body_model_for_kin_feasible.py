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
import copy

import open3d as o3d

########## Drake stuff ##########
import pydrake 
from pydrake.all import eq, Simulator, Solve
from pydrake.autodiffutils import ExtractValue, ExtractGradient, InitializeAutoDiff, AutoDiffXd
from pydrake.common import FindResourceOrThrow
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import DrakeVisualizer, Sphere, Box, Cylinder, CollisionFilterManager, CollisionFilterDeclaration, GeometrySet
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

class AllegroHandPlantDrake:
    def __init__(self, object_path=None, object_base_link_name=None,
                 object_world_pose=None, object_setup_fn=None, has_tabletop=True, meshcat=True,
                 meshcat_open_brower=True, meshcat_show_point_cloud=False,
                 num_viz_spheres=3, viz_sphere_colors=None, viz_sphere_radius=0.015,
                 num_viz_triads=5, viz_triad_length=.05, viz_triad_radius=0.003,
                 viz_triad_colors=None,
                 X_EH = None
                 ):
        self.num_fingers = 4
        self.dof_per_finger = 4
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, MultibodyPlant(time_step=0.01))

        #drake_allegro_path = FindResourceOrThrow(model_param.drake_allegro_path)
        drake_allegro_path = model_param.allegro_hand_urdf_path
        parser = Parser(self.plant)
        self.hand_model = parser.AddModelFromFile(drake_allegro_path)
        self.has_object = object_path is not None or object_setup_fn is not None
        self.has_tabletop = has_tabletop
        # Add object
        if object_setup_fn is not None:
            object_setup_fn(self)
        elif object_path is not None:
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

        # Add tabletop
        if self.has_tabletop:
            # Create the tabletop with
            box = Box(width=model_param.tabletop_dim[0], 
                        depth=model_param.tabletop_dim[1], 
                        height=model_param.tabletop_dim[2])
            spatial_inertia = SpatialInertia(mass=1.,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(0.,0.,0.)
            )
            tabletop_model_instance = self.plant.AddModelInstance("tabletop_model_instance")
            box_body = self.plant.AddRigidBody(name="tabletop_box",
                                M_BBo_B=spatial_inertia, model_instance=tabletop_model_instance)
            body_X_BG = RigidTransform([0.,0.,0.])
            # Highly frictionous
            body_friction = CoulombFriction(static_friction=model_param.tabletop_mu_static,
                                    dynamic_friction=model_param.tabletop_mu_dynamic)
            self.plant.RegisterVisualGeometry(
                body=box_body, X_BG=body_X_BG, shape=box, name="tabletop_box_visual",
                diffuse_color=[0.8, 0.8, 0.8, 1.])
                # diffuse_color=[1., 0.64, 0.0, 1.])
                
            self.plant.RegisterCollisionGeometry(
                body=box_body, X_BG=body_X_BG, shape=box,
                name="tabletop_box_collision", coulomb_friction=body_friction)
            # Create tabletop frame and weld to world
            self.tabletop_body = self.plant.GetBodyByName("tabletop_box")
            self.tabletop_frame = self.tabletop_body.body_frame()
            # Tabletop is at xy plane
            p_W_tabletop = np.array([0., 0., -model_param.tabletop_dim[2]/2.])
            r_W_tabletop = np.zeros(3)
            tabletop_world_pose = RigidTransform(RollPitchYaw(*r_W_tabletop), p_W_tabletop)
            self.plant.WeldFrames(
                    self.plant.world_frame(),
                    self.tabletop_frame, tabletop_world_pose)

        self.hand_body = self.plant.GetBodyByName("hand_root")
        self.H_drake_frame = self.hand_body.body_frame()
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
        if X_EH is not None:
            self.X_EH = RigidTransform(X_EH)
        # if self.X_EH is not None:
        #     self.E_drake_frame = pydrake.multibody.tree.FixedOffsetFrame_[float](
        #         name="franka_flange_frame",
        #         P=self.hand_body, # Not sure why this segfaults with H_drake_frame.
        #         X_PF=self.X_EH.inverse())
        # else:
        #     self.E_drake_frame = None
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
            try:
                iterator = iter(viz_sphere_radius)
            except TypeError:
                viz_sphere_radius = [viz_sphere_radius]*num_viz_spheres
            # sphere = scenario.AddShape(self.plant, Sphere(0.015), viz_name, color=color, collidable=False)
            sphere = scenario.AddShape(self.plant, Sphere(viz_sphere_radius[i]), viz_name, color=color, collidable=False)
            self.viz_spheres.append(sphere)
            self.viz_sphere_names.append(viz_name)
            # self.viz_spheres_frames[finger] = self.plant.GetBodyByName(viz_name, sphere).body_frame()
        self.num_viz_triads = num_viz_triads
        self.viz_triads = []
        self.viz_triad_names = []
        self.viz_triad_pose_idx_start = []
        self.viz_triad_length = viz_triad_length
        # Create the vizualization triads
        for i in range(num_viz_triads):
            # Add visualization triads
            viz_name = f'viz_triad_{i}'
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                axis_name = viz_name+'/'+axis
                if viz_triad_colors is None:
                    color = np.zeros(4)
                    color[axis_idx%3] = 1.
                    color[-1] = 1.
                    cylinder = scenario.AddShape(self.plant, Cylinder(viz_triad_radius, viz_triad_length), 
                                                                    axis_name, color=color, collidable=False)
                else:
                    cylinder = scenario.AddShape(self.plant, Cylinder(viz_triad_radius, viz_triad_length), 
                                                                    axis_name, color=viz_triad_colors[i][axis_idx], collidable=False)
                self.viz_triads.append(cylinder)
                self.viz_triad_names.append(axis_name)
            # self.viz_spheres_frames[finger] = self.plant.GetBodyByName(viz_name, sphere).body_frame()
        
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
        if self.has_object:
            object_collision_candidates = set(self.plant.GetCollisionGeometriesForBody(self.object_body))
        else:
            object_collision_candidates = set()
        if self.has_tabletop:
            object_collision_candidates.update(self.plant.GetCollisionGeometriesForBody(self.tabletop_body))
        non_object_collision_candidates = all_collision_candidates.difference(object_collision_candidates)
        hand_nontip_collision_candidates = non_object_collision_candidates.difference(
            self.fingertip_sphere_collisions.values())
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
        if not meshcat:
            self.meshcat_viz = None
        else:
            self.meshcat_viz = ConnectMeshcatVisualizer(builder, self.scene_graph,
                                                        open_browser=meshcat_open_brower)
        if meshcat_show_point_cloud:
            self.pc_viz = builder.AddSystem(
                MeshcatPointCloudVisualizer(self.meshcat_viz))
        else:
            self.pc_viz = None

        self.diagram = builder.Build()

        diagram_context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, diagram_context)
        self.plant.SetPositions(
            plant_context, np.zeros(self.plant.num_positions(), ))
        # Store the indices of the visualization spheres
        for _, name in enumerate(self.viz_sphere_names):
            self.viz_sphere_positions_idx_start.append(self.plant.GetBodyByName(name).floating_positions_start()+4)
        for _, name in enumerate(self.viz_triad_names):
            self.viz_triad_pose_idx_start.append(self.plant.GetBodyByName(name).floating_positions_start())
        # print('FB', self.plant.GetFloatingBaseBodies())
        # body_idx = self.plant.GetFloatingBaseBodies().pop()
        # print('Body idx', body_idx)
        # body = self.plant.get_body(body_idx)
        # print('has_quat', body.has_quaternion_dofs())
        # print('float_start', body.floating_positions_start())
        # body_frame_id = self.plant.GetBodyFrameIdOrThrow(body_idx)
        self.num_positions = self.plant.num_positions()
        # Note the object is welded, so it has no dof
        assert self.num_positions == 23+7*len(self.viz_sphere_names)+7*len(self.viz_triad_names)
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

    def get_bullet_hand_config_from_drake_q(self, q):
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
        print('bullet', self.hand_base_bullet_frame.CalcPoseInWorld(plant_context))
        print('base', self.H_drake_frame.CalcPoseInWorld(plant_context))
        base_position = X_WB.translation()
        # q is in [w,x,y,z]
        base_quaternion = X_WB.rotation().ToQuaternion().wxyz()
        base_quaternion = base_quaternion[np.array([1, 2, 3, 0], dtype=int)]
        hand_q = q[self.finger_joint_idx_start:]  # np.zeros(16)
        # for finger in AllegroHandFinger:
        #     finger_order = int(finger.value//4)
        #     bullet_q_idx = np.arange(4*finger_order, 4*finger_order+4)
        #     print(finger, bullet_q_idx)
        #     hand_q[bullet_q_idx] = q[self.finger_joints_idx[finger]]
        return base_position, base_quaternion, hand_q

    def compute_p_WF(self, drake_q, plant_context=None):
        """
        Compute the fingertip locations in the world frame p_WF given
        a hand configuration drake_q
        :param drake_q:
        :param plant_context:
        :return: dictionary of 3x1 vectors
        """
        ans = {}
        if plant_context is None:
            diagram_context, plant_context = self.create_context()
        T = drake_q.dtype
        q_val = ExtractValue(drake_q) if T == object else drake_q
        if (not np.array_equal(self.plant.GetPositions(plant_context), q_val)):
            self.plant.SetPositions(plant_context, q_val)
        for finger in AllegroHandFinger:
            X_WT = self.fingertip_frames[finger].CalcPoseInWorld(plant_context)
            if T == object:
                # Construct autodiff
                Jv_q_WFi = self.plant.CalcJacobianTranslationalVelocity(
                    context=plant_context,
                    with_respect_to=JacobianWrtVariable.kQDot,
                    frame_B=self.fingertip_frames[finger],
                    p_BoBi_B=np.zeros(3),
                    frame_A=self.plant.world_frame(),
                    frame_E=self.plant.world_frame())
                ans[finger] = InitializeAutoDiff(value=X_WT.translation(),
                                                 gradient=Jv_q_WFi)
            else:
                ans[finger] = X_WT.translation().reshape(-1,1)
        return ans

    def compute_p_W_distance_to_object_with_o3d(self, p_W):
        T = p_W.dtype
        p_W_val = ExtractValue(p_W) if T == object else p_W
        if T == object:
            # Perform a batch query for finite differencing
            all_p_W_val = np.tile(np.squeeze(p_W_val), (7,1)) # 7x3
            # Perturb
            # Order is (x-, y-, z-, x+, y+, z+, original)
            for sgn_idx, sgn in enumerate([-1.,1.]):
                for coord_idx in range(3):
                    all_p_W_val[sgn_idx*3+coord_idx, coord_idx] += sgn*model_param.mesh_distance_finite_difference_step/2.
            query_points = o3d.core.Tensor(all_p_W_val, dtype=o3d.core.Dtype.Float32)
            # Compute nearest point with o3d
            all_signed_dist = np.squeeze(self.o3d_scene.compute_signed_distance(query_points).numpy())
            Jp_d = np.zeros((1, 3))
            for coord_idx in range(3):
                Jp_d[:,coord_idx] = (all_signed_dist[coord_idx+3]-all_signed_dist[coord_idx])/model_param.mesh_distance_finite_difference_step
            return InitializeAutoDiff(value=np.atleast_2d(all_signed_dist[-1]), gradient=Jp_d)
        else:                
            query_point = o3d.core.Tensor(p_W_val.reshape((1,-1)), dtype=o3d.core.Dtype.Float32)
            # Compute nearest point with o3d
            signed_dist = np.squeeze(self.o3d_scene.compute_signed_distance(query_point).numpy())
            return signed_dist

    def compute_p_WF_distance_to_object_with_o3d_from_q(self, q, plant_context=None):
        T = q.dtype
        p_WF = self.compute_p_WF(q, plant_context)
        if T == object:
            ans = None
            for fi, finger in enumerate(model_param.ActiveAllegroHandFingers):
                di = self.compute_p_W_distance_to_object_with_o3d(p_WF[finger])
                di_val = ExtractValue(di)
                Jp_di = ExtractGradient(di) # 3x3
                Jq_p = ExtractGradient(p_WF[finger]) # 3xlen(q)
                Jq_di = Jp_di @ Jq_p # 3xlen(q)
                if ans is None:
                    ans = InitializeAutoDiff(value=np.atleast_2d(di_val), gradient=Jq_di)
                else:
                    ans = np.vstack([ans, InitializeAutoDiff(value=np.atleast_2d(di_val), gradient=Jq_di)])
            return ans
        else:
            return np.vstack([self.compute_p_W_distance_to_object_with_o3d(p_WF[finger])
                                    for finger in model_param.ActiveAllegroHandFingers])

    def populate_q_with_viz_sphere_positions(self, positions, q):
        for i in range(len(positions)):
            start_index = self.viz_sphere_positions_idx_start[i] # quaternion excluded
            # Populate the quaternions so it's not all zeros
            q[start_index-4] = 1.
            q[start_index:start_index+3] = positions[i]
        return None

    def populate_q_with_viz_triad_X_WT(self, X_WTs, q, start_triad_idx=0):
        for triad_idx, X_WT in enumerate(X_WTs):
            triad_idx_in_q = start_triad_idx + triad_idx
            for triad_axis_idx in range(3):
                start_index = self.viz_triad_pose_idx_start[3*triad_idx_in_q+triad_axis_idx]
                # Compute the rigid trasnform
                    # x-axis
                if triad_axis_idx == 0:
                    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2),
                                        [self.viz_triad_length / 2., 0, 0])
                elif triad_axis_idx == 1:
                    # y-axis
                    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2),
                                        [0, self.viz_triad_length / 2., 0])
                else:
                    # z-axis
                    X_TG = RigidTransform([0, 0, self.viz_triad_length / 2.])
                X_WG = X_WT.multiply(X_TG)
                q[start_index:start_index+4] = X_WG.rotation().ToQuaternion().wxyz()
                q[start_index+4:start_index+7] = X_WG.translation()
        return None
    
    def populate_q_with_viz_triad_grasp_conf(self, p_WF_dict, C_WF_dict, f_N_dict, q):
        """
        Sugar function for visualizing contact basis
        """
        if f_N_dict is not None:
            assert self.num_viz_triads >= 6
        X_WTs = []
        for f_idx, finger in enumerate(model_param.ActiveAllegroHandFingers):
            # Compute X_WT
            X_NF_drake = RigidTransform()
            X_NF_drake.set_rotation(RotationMatrix(C_WF_dict[finger].T))
            X_NF_drake.set_translation(p_WF_dict[finger])
            X_WTs.append(X_NF_drake)
        # Compute the normals
        if f_N_dict is None:
            return q
        for f_idx, finger in enumerate(model_param.ActiveAllegroHandFingers):
            # Compute X_WT
            X_NF_drake = RigidTransform()
            # Compute the rotation matrix of the contact force in contact frame N 
            # rot_F_N = np.zeros((3,3))
            # rot_F_N[0,:] = f_W_dict[finger] / np.linalg.norm(f_W_dict[finger])
            # rot_F_N[1,:] = np.cross(np.array([1.,0.,0.]), rot_F_N[0,:])
            # rot_F_N[2,:] = np.cross(rot_F_N[0,:],rot_F_N[1,:])
            # rot_F_N = np.eye(3)
            # Compute using axis angle
            x_hat = np.array([1.,0.,0.])
            f_hat = -f_N_dict[finger] / np.linalg.norm(f_N_dict[finger])
            axis = np.cross(x_hat, f_hat)
            # For numerical stability
            axis[0] += 1e-7
            axis /= np.linalg.norm(axis)
            angle = np.arccos(np.dot(x_hat, f_hat))
            X_NF_drake.set_rotation(AngleAxis(angle, axis))
            X_NF_drake.set_translation(np.zeros(3))
            X_WTs.append(X_WTs[f_idx].multiply(X_NF_drake))
        self.populate_q_with_viz_triad_X_WT(X_WTs, q)
        return q

    def create_context(self):
        """
        This is a syntax sugar function. It creates a new pair of diagram
        context and plant context.
        """
        diagram_context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, diagram_context)
        return (diagram_context, plant_context)

    def visualize_point_cloud(self, point_cloud, diagram_context,  point_cloud_rgbs=None):
        assert self.pc_viz is not None
        # Convert to drake datatype if necessary
        if not isinstance(point_cloud, drake_perception.PointCloud):
            if point_cloud_rgbs is not None:
                pc_drake = drake_perception.PointCloud(len(point_cloud), drake_perception.Fields(drake_perception.BaseField.kXYZs | drake_perception.BaseField.kRGBs))
                pc_drake.mutable_rgbs()[:] = point_cloud_rgbs.T*255.
            else:
                pc_drake = drake_perception.PointCloud(len(point_cloud), drake_perception.Fields(drake_perception.BaseField.kXYZs))
            # Drake's convention is 3xn for an n-point cloud
            pc_drake.mutable_xyzs()[:] = point_cloud.T
        else:
            pc_drake = point_cloud
        context = self.diagram.GetMutableSubsystemContext(
            self.pc_viz, diagram_context)
        self.pc_viz.GetInputPort("point_cloud_P").FixValue(
            context, AbstractValue.Make(pc_drake))

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
    
    def convert_hand_configuration_matrix_to_q(self, base_position, base_orn_matrix, finger_angles):
        q = np.zeros(self.plant.num_positions())
        q[self.base_position_idx_start:
          self.base_position_idx_start + 3] = base_position
        base_quaternion = R.from_matrix(base_orn_matrix).as_quat()[[3,0,1,2]]
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

    def construct_vanilla_collision_aware_ik(self, plant_context, 
                                            collision_distance=0.,
                                            threshold_distance=0.1,
                                            max_zneg_E_angle_from_z_P=None,
                                            constrain_p_PE=True,
                                            p_PE_min=None,
                                            p_PE_max=None):
        ik = InverseKinematics(self.plant, plant_context)
        collision_constr = ik.AddMinimumDistanceConstraint(collision_distance, threshold_distance)
        if max_zneg_E_angle_from_z_P:
            angle_constr = self.add_max_zneg_E_angle_from_z_P_constraint_to_ik(ik, max_zneg_E_angle_from_z_P)
        else:
            angle_constr = None
        if constrain_p_PE:
            p_PE_constr = self.add_p_PE_constraint_to_ik(ik, p_PE_min, p_PE_max)
        return ik, collision_constr, angle_constr, p_PE_constr

    def construct_ik_given_fingertip_normals(self, plant_context,
                                             finger_normal_map,
                                             padding=model_param.object_padding,
                                             collision_distance=0.,
                                             has_normals=True,
                                             allowed_deviation=np.ones(3)*0.01,
                                             threshold_distance=1.,
                                             straight_unused_fingers=False):
        ik = InverseKinematics(self.plant, plant_context)
        unused_fingers = set(AllegroHandFinger)
        constraints_on_finger = {}
        desired_positions = {}
        for finger in finger_normal_map.keys():
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
                                                 0., np.pi/2.))
            unused_fingers.remove(finger)

        prog = ik.get_mutable_prog()
        q = ik.q()
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
            collision_constr = ik.AddMinimumDistanceConstraint(collision_distance, threshold_distance)
        return ik, constraints_on_finger, collision_constr, desired_positions

    def compute_n_t0_t1_from_p_WFi_with_mesh(self, p_WFi):
        assert p_WFi.dtype != object
        query_point = o3d.core.Tensor(p_WFi.reshape((1,-1)), dtype=o3d.core.Dtype.Float32)
        # Compute nearest point with o3d
        query_result = self.o3d_scene.compute_closest_points(query_point)
        nearest_feature_idx = query_result['primitive_ids'][0].item()
        # Get the normal from this feature
        n_W = self.o3d_mesh.triangle_normals[nearest_feature_idx]
        n_W /= np.linalg.norm(n_W)
        t0_W = np.asarray([-n_W[1], n_W[0], 0.])
        t0_W /= np.linalg.norm(t0_W)
        t1_W = np.cross(n_W, t0_W)
        return np.vstack([n_W, t0_W, t1_W])

    def compute_n_t0_t1_from_q_with_mesh(self, q, plant_context):
        """
        Compute [n_W.T, t0_W.T, t1_W].T, an orthonormal basis that defines the object surface frame using the object mesh
        n_W is the contact normal direction that points out of the object. t0_W, t1_W are two tangent
        directions
        :param q:
        :param plant_context:
        :return: [n_W.T, t0_W.T, t1_W].T
        """
        ans = {}
        if plant_context is None:
            diagram_context, plant_context = self.create_context()
        T = q.dtype
        q_val = ExtractValue(q) if T == object else q
        if (not np.array_equal(self.plant.GetPositions(plant_context), q_val)):
            self.plant.SetPositions(plant_context, q_val)
        p_WF = self.compute_p_WF(q, plant_context)
        # For brevity, the finger index i are dropped in the following comments.
        # e.g. R_WFi → R_WF
        if T != object:
            for fi, finger in enumerate(model_param.ActiveAllegroHandFingers):
                ans[finger] = self.compute_n_t0_t1_from_p_WFi_with_mesh(p_WF[finger])
            return ans
        else:
            # Compute the Jacobian w.r.t p
            for fi, finger in enumerate(model_param.ActiveAllegroHandFingers):
                p_WFi_val = ExtractValue(p_WF[finger])
                Jq_p_WFi = ExtractGradient(p_WF[finger]) # 3xlen(q)
                # First sqeeze it to 9
                n_t0_t1_finger = np.squeeze(self.compute_n_t0_t1_from_p_WFi_with_mesh(p_WFi_val))
                Jp_WFi_n_t0_t1_T = np.zeros((9,3))
                # Compute the 
                for p_coord in range(3):
                    p_WFi_plus = np.copy(p_WFi_val)
                    p_WFi_plus[p_coord] += model_param.mesh_normal_finite_difference_step/2.
                    # Compute n
                    p_WFi_minus = np.copy(p_WFi_val)
                    p_WFi_minus[p_coord] -= model_param.mesh_normal_finite_difference_step/2.
                    n_t0_t1_finger_plus = np.ndarray.flatten(self.compute_n_t0_t1_from_p_WFi_with_mesh(p_WFi_plus))
                    n_t0_t1_finger_minus = np.ndarray.flatten(self.compute_n_t0_t1_from_p_WFi_with_mesh(p_WFi_minus))
                    Jp_WFi_n_t0_t1_T[:,p_coord] = (n_t0_t1_finger_plus-n_t0_t1_finger_minus)/model_param.mesh_normal_finite_difference_step
                Jq_n_t0_t1_T = Jp_WFi_n_t0_t1_T @ Jq_p_WFi
                ans[finger] = InitializeAutoDiff(value=np.vstack([n_t0_t1_finger[0], n_t0_t1_finger[1], n_t0_t1_finger[2]]), 
                                                gradient=Jq_n_t0_t1_T) # Note that Drake gradient is 2D
            return ans

    def add_grasp_dynamic_constraint_to_prog(self, prog, q_ik, f_W, plant_context, finger_surface_map,
                                             min_normal_force=model_param.min_normal_force):
        """
        Create new decision variables representing the contact forces. Add a constraint that the contact
        forces must satisfy
        1. In the friction cone
        2. External force & torque sum to zero
        3. Normal force is at least 1.
        Remarks:
        1. We do not consider the internal force in this formulation
        2. We assume the normal directions are consistent in finger_surface_map, i.e. the z direction points
        away from the center of the object and the resulting normal forces should be negative
        are negative.
        :param prog:
        :param q:
        :param plant_context:
        :param finger_surface_map:
        :return:
        """
        p_diff_WF = {}
        p_WF = {}
        pbar_WF = 0.
        normal_force_constraints = {}
        friction_cone_constraints = {}
        for finger in model_param.ActiveAllegroHandFingers:
            prog.SetInitialGuess(f_W[finger], np.array([0., 0., -min_normal_force]))
            # Normal force magnitude constraint
            raise AssertionError
        return normal_force_constraints, friction_cone_constraints

    @staticmethod
    def construct_and_solve_wrench_closure_qp(p_WF, C_WF, mu=model_param.friction_coefficient):
        Q, P, G, h, A, b = \
            AllegroHandPlantDrake.get_min_external_wrench_qp_matrices(p_WF, C_WF,
                                                    mu)
        try:
            if isinstance(p_WF, torch.Tensor):
                ans = QPFunction(eps=1e-12, verbose=0, notImprovedLim=3,
                                        maxIter=20, solver=QPSolvers.CVXPY,
                                        check_Q_spd=False)(Q, P, G, h, A, b)
                # TODO: better error handling
                if ans is None:
                    return None, torch.Tensor([[np.inf]], requires_grad=True)
                z_hat = ans.type_as(Q).view(-1)
                obj = 0.5*z_hat@Q@z_hat + P@z_hat
            else:
                z_cp = cp.Variable(Q.shape[0])
                prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(z_cp, Q)),
                                [G @ z_cp <= h])
                prob.solve()
                z_hat = z_cp.value
                obj = prob.value
        except AssertionError:
            # prob.status isn't optimal
            print('Likely failed qpth package "qpth/solvers/cvxpy.py" line 22')
            print("assert('optimal' in prob.status)")
            print("Please comment that out")
            raise AssertionError
        except cp.SolverError:
            if isinstance(p_WF, torch.Tensor):
                return None, torch.Tensor([[np.inf]], requires_grad=True)
            else:
                return None, np.inf
        return z_hat, obj

    def compute_wrench_closure_obj_from_q(self, q, plant_context):
        """
        Construct and solve the lower-level QP to minimize the external wrench squared L2 norm
        :param q: Drake autodiff representing the hand configuration
        :param plant_context:
        :return: Drake
        """
        T = q.dtype
        p_WF_dict = self.compute_p_WF(q)
        p_WF = np.hstack([p_WF_dict[finger] for finger in model_param.ActiveAllegroHandFingers]).T
        # C_WF is the same as R_WF
        n_t0_t1_dict = self.compute_n_t0_t1_from_q_with_mesh(q, plant_context)
        # assert p_WF.shape == (3, 3)
        # assert C_WF.shape == (3, 3, 3)
        # Convert p_WF and C_WF to torch
        p_WF_torch = torch.tensor(ExtractValue(p_WF))
        # C_WF: 3x(3x3) orthonormal basis Cᵢ = [ᵂn̂ᵢ,ᵂt̂ᵢ₀,ᵂt̂ᵢ₁]
        C_WF_torch = torch.zeros(3,3,3, dtype=p_WF_torch.dtype)
        for f_idx, finger in enumerate(model_param.ActiveAllegroHandFingers):
            n_W = np.squeeze(ExtractValue(n_t0_t1_dict[finger][0]))
            t0_W = np.squeeze(ExtractValue(n_t0_t1_dict[finger][1]))
            t1_W = np.squeeze(ExtractValue(n_t0_t1_dict[finger][2]))
            C_WF_torch[f_idx, :, 0] = torch.from_numpy(n_W)
            C_WF_torch[f_idx, :, 1] = torch.from_numpy(t0_W)
            C_WF_torch[f_idx, :, 2] = torch.from_numpy(t1_W)
        def construct_and_solve_wc_qp_obj(p_WF_torch, C_WF_torch):
            # Only return the objective
            return self.construct_and_solve_wrench_closure_qp(p_WF_torch, C_WF_torch)[1]
        z_hat_torch, obj_torch = self.construct_and_solve_wrench_closure_qp(p_WF_torch, C_WF_torch)
        if z_hat_torch is None:
            # Didn't solve
            z_hat_torch = np.inf
        obj = obj_torch.detach().numpy()
        if T == object:
            # Extract the Jacobian of f_S w.r.t. p_WF and C_WF
            Jp_WF_obj_torch, JC_WF_obj_torch = torch.autograd.functional.jacobian(construct_and_solve_wc_qp_obj,
                                                                                  (p_WF_torch, C_WF_torch),
                                                                                  create_graph=True, strict=True)
            # Jp_WF_obj is 3x3: Jacobian of objective w.r.t each entry of p_WF ∈ ℝ³ˣ³
            Jp_WF_obj = np.squeeze(Jp_WF_obj_torch.detach().numpy())
            # JC_WF_obj is 3x3x3: Jacobian of objective w.r.t each entry of C_WF ∈ ℝ³ˣ³ˣ³
            JC_WF_obj = np.squeeze(JC_WF_obj_torch.detach().numpy())

            # Jq_p_WF is (3x3)xlen(q)
            Jq_p_WF = np.stack([ExtractGradient(p_WF_dict[finger]) for finger in model_param.ActiveAllegroHandFingers])
            # Jq_C_WF is 3x(3x3)xlen(q): 3 fingers x [ᵂn̂ᵢ,ᵂt̂ᵢ₀,ᵂt̂ᵢ₁] x q
            Jq_C_WF = np.zeros((3,3,3,len(q)))
            for f_idx, finger in enumerate(model_param.ActiveAllegroHandFingers):
                Jq_n_W = np.squeeze(ExtractGradient(n_t0_t1_dict[finger][0]))
                Jq_t0_W = np.squeeze(ExtractGradient(n_t0_t1_dict[finger][1]))
                Jq_t1_W = np.squeeze(ExtractGradient(n_t0_t1_dict[finger][2]))
                # Notice that since ᵂn̂ᵢ,ᵂt̂ᵢ₀,ᵂt̂ᵢ₁ are column vectors, we are filling in dim 1 & 3
                Jq_C_WF[f_idx, :, 0, :] = Jq_n_W
                Jq_C_WF[f_idx, :, 1, :] = Jq_t0_W
                Jq_C_WF[f_idx, :, 2, :] = Jq_t1_W
            # Use chain rule to compute the total gradient w.r.t. q
            # Jq_obj = np.sum(np.multiply(np.atleast_3d(Jp_WF_obj), Jq_p_WF), axis=(0,1))+ \
            #          np.sum(np.multiply(np.expand_dims(JC_WF_obj,-1), Jq_C_WF), axis=(0,1,2))
            J_mul_p_WF = np.multiply(Jp_WF_obj.reshape((3, 3, 1)), Jq_p_WF)
            Jq_obj = np.sum(np.multiply(Jp_WF_obj.reshape((3,3,1)), Jq_p_WF), axis=(0,1))+ \
                     np.sum(np.multiply(JC_WF_obj.reshape((3,3,3,1)), Jq_C_WF), axis=(0,1,2))
            return InitializeAutoDiff(np.atleast_2d(obj), np.atleast_2d(Jq_obj).reshape(1,-1))
        else:
            return np.atleast_2d(obj)

    def add_fingertip_distance_to_mesh_constraint_to_prog(self, prog, q, plant_context, 
                                                        min_dist=model_param.fingertip_radius*0.6, 
                                                        max_dist=model_param.fingertip_radius*0.9):
        objective_fn = partial(self.compute_p_WF_distance_to_object_with_o3d_from_q, plant_context=plant_context)
        return prog.AddConstraint(
            objective_fn,
            min_dist*np.ones(3), max_dist*np.ones(3), vars=q)

    def add_bilevel_grasp_dynamic_constraint_to_prog(self, prog, q, plant_context, eps=1e-7):
        """
        Add a constraint that construct_and_solve_wrench_closure_qp returns 0.
        Recall that this objective is the sum of external force L2 norm and external torque L2 norm
        :param prog:
        :param q:
        :param plant_context:
        :param finger_surface_map:
        :param eps:
        :return:
        """
        objective_fn = partial(self.compute_wrench_closure_obj_from_q, plant_context=plant_context)
        # First extract the fingertip positions and normals
        return prog.AddConstraint(
            objective_fn,
            -eps*np.ones((1,1)), eps*np.ones((1,1)), vars=q)

    def add_max_zneg_E_angle_from_z_P_constraint_to_ik(self, ik, max_angle):
        """
        Constrain the franka 
        """
        xneg_H = np.asarray([-1., 0. , 0.]).reshape([3,1]) # Negative x of hand is parallel to negative z of flange frame
        z_P = np.asarray([0., 0. , 1.]).reshape([3,1])
        return ik.AddAngleBetweenVectorsConstraint(frameA=self.plant.world_frame(),
                                            na_A=z_P,
                                            frameB=self.H_drake_frame,
                                            nb_B=xneg_H,
                                            angle_lower=0.,
                                            angle_upper=max_angle
                                            )
        
    def add_p_PE_constraint_to_ik(self, ik, p_PE_min=None, p_PE_max=None):
        """
        Constrain the franka 
        """
        if p_PE_min is None:
            p_PE_min = model_param.p_PE_min
        if p_PE_max is None:
            p_PE_max = model_param.p_PE_max
        return ik.AddPositionConstraint(
            self.H_drake_frame, self.X_EH.inverse().translation(), 
            self.plant.world_frame(),
            p_PE_min, p_PE_max
            )

    @staticmethod
    def get_min_external_wrench_qp_matrices(p_WF, C_WF, mu=model_param.friction_coefficient,
                                            psd_offset=1e-7, min_normal_force=model_param.min_normal_force):
        """
        Construct standard form QP
        min zᵀQz + Pᵀz s.t. Az=b, Gz≤h
        where z = [[αᵢ, βᵢ₁, βᵢ₂] for i =1,2,3]] ∈ ℝ⁹.
        This encodes the QP
        min_{αᵢ, βᵢ₁, βᵢ₂} ‖∑ᵢᵂFᵢ‖² + ‖∑ᵢᵂτᵢ‖²,
        subjected to
        L1 friction cone constraints: βᵢ₁, βᵢ₂ ≤ |αᵢ|t̂ᵢ₁
        nonzero normals: αᵢ ≤ -1.
        αᵢ, βᵢ₁, βᵢ₂ parameterizes the contact force in the contact frame, i.e.
        ᵂFᵢ = Cᵢ[αᵢ, βᵢ₁, βᵢ₂]ᵀ
        ᵂτᵢ = ᵂpᵢF × ᵂFᵢ
        In particular
        Q = [[CᵢᵀCⱼ for j=1,2,3] for i=1,2,3] + [[([ᵂpᵢF]ₓCᵢ)ᵀ([ᵂpⱼF]ₓCⱼ) for j=1,2,3] for i=1,2,3]
        P = 0₁ₓ₉
        G = [Gf; Gα] where
        Gf = block_diag([[friction_coeff, -1., 0.],
                        [friction_coeff, 0., -1.],
                        [friction_coeff, 1., 0.],
                        [friction_coeff, 0., 1.]]) ∈ ℝ¹²ˣ⁹
        Gα = diag([1,0,0]*3)
        h = [zeros(12), -1, -1, -1]
        A = 0₁ₓ₉
        b = 0
        All parameters may be torch tensors or numpy arrays (TODO).
        Note that this program is always feasible.
        :param p_WF: 3x3 matrix [ᵂp₁F.T; ᵂp₂F.T; ᵂp₃F]. Each row is the contact position of a finger, expressed in world frame.
        :param C_WF: 3x(3x3) orthonormal basis Cᵢ = [ᵂn̂ᵢ,ᵂt̂ᵢ₀,ᵂt̂ᵢ₁].
                    Each of which denotes the frame at contact. ᵂn̂ᵢ points out of the object.
        :param mu: friction coefficient
        :return: matrices Q, P, G, h, A, b
        """
        T = type(p_WF)
        if T == torch.Tensor:
            dtype = p_WF.dtype
            C_123_T = torch.vstack([C_WF[i,:,:].T for i in range(3)]) # 9x3
            Q = C_123_T @ (C_123_T.T)
            p_WF_cross_C = torch.hstack([torch.linalg.cross(torch.atleast_2d(p_WF[i,:]),
                                                       C_WF[i,:,:].T, dim=-1).T for i in range(3)])
            Q += p_WF_cross_C.T @ (p_WF_cross_C)
            P = torch.zeros(1,9, dtype=dtype, requires_grad=False)
            Gf = torch.block_diag(*([torch.tensor([[mu, -1., 0.],
                                                   [mu, 0., -1.],
                                                   [mu, 1., 0.],
                                                   [mu, 0., 1.]],
                                                  dtype=dtype, requires_grad=True)]*3)) # 6x9
            Ga = torch.block_diag(*([torch.tensor([1., 0., 0.], dtype=dtype,
                                                  requires_grad=False)]*3)) # 3x9
            G = torch.vstack([Gf, Ga]) # 15x9
            h = torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.], dtype=dtype, requires_grad=False)*min_normal_force
            A = torch.Tensor()
            b = torch.Tensor()
            # A = torch.zeros(1,9, dtype=dtype, requires_grad=False)
            # b = torch.tensor([0.], dtype=dtype, requires_grad=False)
            # Add a small offset in Q's diagonal entries for numerical stability
            Q += torch.eye(Q.shape[0], dtype=Q.dtype)*psd_offset
            return Q, P, G, h, A, b
        else:
            C_123_T = np.vstack([C_WF[i,:,:].T for i in range(3)]) # 9x3
            Q = C_123_T @ (C_123_T.T)
            p_WF_cross_C = np.hstack([np.cross(np.atleast_2d(p_WF[i,:]),
                                                       C_WF[i,:,:].T).T for i in range(3)])
            Q += p_WF_cross_C.T @ (p_WF_cross_C)
            P = np.zeros((1,9))
            Gf = np.zeros((12,9))
            Ga = np.zeros((3,9))
            for i in range(3):
                Gf[4*i:4*(i+1),3*i:3*(i+1)] = np.array([[mu, -1., 0.],
                                                   [mu, 0., -1.],
                                                   [mu, 1., 0.],
                                                   [mu, 0., 1.]]) # 6x9
                Ga[i, 3*i] = 1.
            G = np.vstack([Gf, Ga]) # 15x9
            h = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.])*min_normal_force
            A = np.zeros((1,9))
            b = np.zeros(1)
            # Add a small offset in Q's diagonal entries for numerical stability
            Q += np.eye(Q.shape[0])*psd_offset
            return Q, P, G, h, A, b


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

def calculate_W_W(p_WF, C_WF, z):
    """
    Given p_WF, C_WF, and z, find the world frame total wrench
    :param p_WF
    :param C_WF
    :param z
    :return (F_W, tau_W)
    """
    if isinstance(p_WF, torch.Tensor):
        raise NotImplementedError
    else:
        F_W = np.zeros(3)
        tau_W = np.zeros(3)
        for finger_idx in range(3):
            F_WF = np.squeeze(C_WF[finger_idx,:,:] @ (z[3*finger_idx:3*(finger_idx+1)].reshape(-1,1)))
            F_W += F_WF
            tau_W += np.squeeze(np.cross(p_WF[finger_idx], F_WF))
    return F_W, tau_W

def drake_mesh_object_setup_fn(drake_hand_plant, X_WO, drake_shape, o3d_mesh, diffuse_color=[1,1,1,0.7]):
    obj_model_instance = drake_hand_plant.plant.AddModelInstance("obj_model_instance")
    spatial_inertia = SpatialInertia(mass=1.,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia(0.,0.,0.)
        )
    drake_hand_plant.object_body = drake_hand_plant.plant.AddRigidBody(name="obj",
        M_BBo_B=spatial_inertia, model_instance=obj_model_instance)
    body_X_BG = RigidTransform([0.,0.,0.])
    body_friction = CoulombFriction(static_friction=model_param.tabletop_mu_static,
                            dynamic_friction=model_param.tabletop_mu_dynamic)
    drake_hand_plant.plant.RegisterVisualGeometry(
        body=drake_hand_plant.object_body, X_BG=body_X_BG, shape=drake_shape, name="obj_visual",
        diffuse_color=diffuse_color)
    drake_hand_plant.plant.RegisterCollisionGeometry(
        body=drake_hand_plant.object_body, X_BG=body_X_BG, shape=drake_shape,
        name="obj_collision", coulomb_friction=body_friction)
    drake_hand_plant.object_frame = drake_hand_plant.object_body.body_frame()
    # Tabletop is at xy plane
    drake_hand_plant.plant.WeldFrames(
            drake_hand_plant.plant.world_frame(),
            drake_hand_plant.object_frame, X_WO)
    # Add the o3d object
    drake_hand_plant.o3d_mesh = copy.deepcopy(o3d_mesh).transform(X_WO.GetAsMatrix4())
    drake_hand_plant.o3d_mesh.compute_triangle_normals()
    drake_hand_plant.o3d_mesh.compute_vertex_normals()
    assert drake_hand_plant.o3d_mesh.has_triangle_normals() and drake_hand_plant.o3d_mesh.has_vertex_normals()
    # assert drake_hand_plant.o3d_mesh.is_watertight()
    drake_hand_plant.o3d_scene = o3d.t.geometry.RaycastingScene()
    drake_hand_plant.o3d_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(drake_hand_plant.o3d_mesh))

class FrankaPlantDrake:
    def __init__(self,
                 meshcat=True,
                 meshcat_open_brower=False):
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, MultibodyPlant(time_step=0.01))
        parser = Parser(self.plant)
        drake_franka_path = FindResourceOrThrow(model_param.drake_franka_path)
        parser = Parser(self.plant)
        self.franka_model = parser.AddModelFromFile(drake_franka_path)
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.get_body(
            self.plant.GetBodyIndices(self.franka_model)[0]).body_frame())
        self.franka_joint_idx_start = int(self.plant.GetJointIndices(self.franka_model)[0])
        self.E_drake_frame = self.plant.get_body(
            self.plant.GetBodyIndices(self.franka_model)[-1]).body_frame()
        # Create the tabletop with
        box = Box(width=model_param.tabletop_dim[0], 
                    depth=model_param.tabletop_dim[1], 
                    height=model_param.tabletop_dim[2])
        spatial_inertia = SpatialInertia(mass=1.,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia(0.,0.,0.)
        )
        tabletop_model_instance = self.plant.AddModelInstance("tabletop_model_instance")
        box_body = self.plant.AddRigidBody(name="tabletop_box",
                            M_BBo_B=spatial_inertia, model_instance=tabletop_model_instance)
        body_X_BG = RigidTransform([0.,0.,0.])
        # Highly frictionous
        body_friction = CoulombFriction(static_friction=model_param.tabletop_mu_static,
                                dynamic_friction=model_param.tabletop_mu_dynamic)
        self.plant.RegisterVisualGeometry(
            body=box_body, X_BG=body_X_BG, shape=box, name="tabletop_box_visual",
            diffuse_color=[1., 0.64, 0.0, 1.])
        self.plant.RegisterCollisionGeometry(
            body=box_body, X_BG=body_X_BG, shape=box,
            name="tabletop_box_collision", coulomb_friction=body_friction)
        # Create tabletop frame and weld to world
        self.tabletop_body = self.plant.GetBodyByName("tabletop_box")
        self.tabletop_frame = self.tabletop_body.body_frame()
        # Tabletop is at xy plane
        p_W_tabletop = np.array([0., 0., -model_param.tabletop_dim[2]/2.])
        r_W_tabletop = np.zeros(3)
        tabletop_world_pose = RigidTransform(RollPitchYaw(*r_W_tabletop), p_W_tabletop)
        self.plant.WeldFrames(
                self.plant.world_frame(),
                self.tabletop_frame, tabletop_world_pose)

        self.num_viz_triads = 2
        self.viz_triad_length = 0.05
        self.viz_triads = []
        self.viz_triad_names = []
        self.viz_triad_pose_idx_start = []
        # Create the vizualization triads
        for i in range(self.num_viz_triads):
            # Add visualization triads
            viz_name = f'viz_triad_{i}'
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                color = np.zeros(4)
                color[axis_idx%3] = 1.
                color[-1] = 1.
                axis_name = viz_name+'/'+axis
                sphere = scenario.AddShape(self.plant, Cylinder(0.003, self.viz_triad_length), 
                                                                axis_name, color=color, collidable=False)
                self.viz_triads.append(sphere)
                self.viz_triad_names.append(axis_name)

        self.plant.Finalize()
        if not meshcat:
            self.meshcat_viz = None
        else:
            self.meshcat_viz = ConnectMeshcatVisualizer(builder, self.scene_graph,
                                                        open_browser=meshcat_open_brower)
        # See https://github.com/RobotLocomotion/drake/blob/master/bindings/pydrake/systems/test/meshcat_visualizer_test.py
        self.diagram = builder.Build()

        # Store the indices of the visualization spheres
        for _, name in enumerate(self.viz_triad_names):
            self.viz_triad_pose_idx_start.append(self.plant.GetBodyByName(name).floating_positions_start())

        self.num_positions = self.plant.num_positions()
        # Note the object is welded, so it has no dof

    def create_context(self):
        """
        This is a syntax sugar function. It creates a new pair of diagram
        context and plant context.
        """
        diagram_context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, diagram_context)
        return (diagram_context, plant_context)

    def construct_and_solve_X_BE_ik(self, X_BE, franka_joint_angles=None):
        q_init_ik = np.zeros(self.plant.num_positions())
        if franka_joint_angles is not None:
            q_init_ik[self.franka_joint_idx_start:self.franka_joint_idx_start+len(franka_joint_angles)] = franka_joint_angles
        diagram_context, plant_context = self.create_context()
        ik = InverseKinematics(self.plant, plant_context)
        # Add X_BE constraint
        X_BE.translation()
        ik.AddPositionConstraint(
            self.E_drake_frame, np.zeros(3), self.plant.world_frame(),
        X_BE.translation()-model_param.ik_position_tol, X_BE.translation()+model_param.ik_position_tol)
        ik.AddOrientationConstraint(
            self.E_drake_frame, RotationMatrix(),
            self.plant.world_frame(), X_BE.rotation(), model_param.ik_orn_tol
        )

        prog = ik.get_mutable_prog()
        q = ik.q()
        # prog.AddQuadraticErrorCost(np.identity(len(q)), q_init_ik, q)
        prog.SetInitialGuess(q, q_init_ik)
        result = Solve(ik.prog())
        if result.is_success():
            return result.GetSolution(ik.q())[self.franka_joint_idx_start:self.franka_joint_idx_start+7]
        else:
            print(f"ik failed")
            return None

    def populate_q_with_viz_triad_X_WT(self, X_WTs, q, start_triad_idx=0):
        for triad_idx, X_WT in enumerate(X_WTs):
            triad_idx_in_q = start_triad_idx + triad_idx
            for triad_axis_idx in range(3):
                start_index = self.viz_triad_pose_idx_start[3*triad_idx_in_q+triad_axis_idx]
                # Compute the rigid trasnform
                    # x-axis
                if triad_axis_idx == 0:
                    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2),
                                        [self.viz_triad_length / 2., 0, 0])
                elif triad_axis_idx == 1:
                    # y-axis
                    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2),
                                        [0, self.viz_triad_length / 2., 0])
                else:
                    # z-axis
                    X_TG = RigidTransform([0, 0, self.viz_triad_length / 2.])
                X_WG = X_WT.multiply(X_TG)
                q[start_index:start_index+4] = X_WG.rotation().ToQuaternion().wxyz()
                q[start_index+4:start_index+7] = X_WG.translation()
        return None
