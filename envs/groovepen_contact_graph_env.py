#  Copyright 2021 Stanford University
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

import copy
from pybullet_utils import bullet_client
import open3d as o3d
import pybullet
import utils.render as r
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import pytorch_kinematics as pk
import torch
import os
import inspect

from typing import List, Tuple

import utils.rigidBodySento as rb

import  model.param as model_param 
from utils.math_utils import rotation_matrix_from_vectors, minimumDist2dBox
from utils.helper import convert_q_bullet_to_matrix, getSGSigmapoints
from utils.pen_region import GroovePenRegion
from utils.contact_state_graph import ContactStateGraph
from neurals.distancefield_utils import create_waterbottle_df
from neurals.NPGraspNet import NPGraspNet
from envs.setup_pybullet import create_groovepen
from envs.scales import SCALES

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
CLEARANCE_H = 0.005

SCALE = SCALES["groovepen"]

class GroovePenBulletEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 control_skip=50,
                 active_finger_tips=[0, 1],
                 num_interp_f=5,
                 opt_time=False,
                 last_fins=None,
                 init_obj_pose=None,
                 task=[np.array([0, -1, 0]),np.array([0.5, 0., 0.3])],
                 train=True,
                 steps=3,
                 observe_last_action=False,
                 path=None,
                 dex_path = None,
                 sc_path = None,
                 opt_dict = None,
                 mode="decoder",
                 sigma=0.5,
                 device="cpu",
                 showImage=False,
                 has_distance_field=False,
                 max_forces = model_param.MAX_FORCE,
                 add_physics=False): # TODO: remove spherical coordinate if proved to be none feasible.
        self.train = train
        self.add_physics = add_physics
        self.has_distance_field = has_distance_field
        self.device = device
        self.observe_last_action=observe_last_action
        self.mode = mode
        self.max_forces = max_forces
        self.init_obj_pose=init_obj_pose
        self.render = render
        self.showImage = showImage
        self.sigma = sigma
        self.init_noise = init_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 250. # A constant
        self.all_fingers = [0, 1, 2, 3, 4]
        self.num_fingertips = len(active_finger_tips)
        self.active_finger_tips = active_finger_tips
        self.active_finger_tips_idx = dict()
        for i, finger in enumerate(self.active_finger_tips):
            self.active_finger_tips_idx[finger] = i
        self.num_interp_f = num_interp_f
        self.csg = ContactStateGraph(np.load("data/contact_states/groovepen_env/dummy_states.npy"))
        self.contact_region = GroovePenRegion()
        self.path = path
        if self.num_fingertips != 4:
            print(f"[Warning]: Using {self.num_fingertips}")
        self.opt_time = opt_time
        self.task = task

        self.n_steps = steps
        self.last_surface_norm = dict()
        for finger in active_finger_tips:
            self.last_surface_norm[finger] = None

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
            self.renderer = r.PyBulletRenderer(showImage=showImage)
        else:
            self._p = bullet_client.BulletClient()

        self.np_random = None
        self.o_id = None

        self.tip_ids = []
        self.tip_cids = []

        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0
        self.c_step_timer = 0

        self.floor_id = None

        self.interped_fs = [np.zeros((3, self.control_skip))] * self.num_fingertips
        # True means last control step cp was on (fixed pos) initialize
        self.last_fins = [([0.0] * 3, False)] * self.num_fingertips
        if isinstance(last_fins, np.ndarray):
            for i in range(self.num_fingertips):
                self.last_fins[i] = (last_fins[i,:3], bool(last_fins[i,3]))
        self.cur_fins = [([0.0] * 3, False)] * self.num_fingertips

        self.all_active_fins = [[[0.0] * 3] * self.num_fingertips] * self.n_steps

        self.single_action_dim = num_interp_f * 3 + 3               # f position (2) & on/off (1)
        self.action_dim = self.single_action_dim * self.num_fingertips
        if self.opt_time:
            self.action_dim += 1

        # Prepare neural network related stuff
        pred_fingers = [0, 1, 2]
        for finger in self.active_finger_tips:
            pred_fingers.remove(finger)
        
        if self.train:
            print("Using CUDA Device:", self.device, f"Using: {self.mode}")
            self.score_function = NPGraspNet(opt_dict, pred_fingers=pred_fingers,
                                             extra_cond_fingers = self.active_finger_tips,
                                             mode = self.mode, device = self.device,
                                             has_distance_field = self.has_distance_field)
            if self.mode != "only_score":
                self.score_function.load_dex_grasp_net(dex_path)
            self.score_function.load_score_function(sc_path)
        self.dist_field_env = create_waterbottle_df()

        obs = self.reset()  # and update init obs

        self.act = [0.0] * self.action_dim
        self.action_space = gym.spaces.Box(low=np.array([-1.] * self.action_dim),
                                           high=np.array([+1.] * self.action_dim))
        self.obs_dim = len(obs)
        obs_dummy = np.array([1.12234567] * self.obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf * obs_dummy, high=np.inf * obs_dummy)

    def reset(self) -> List[float]:
        # full reload
        self._p.resetSimulation()
        if self.render:
            self.renderer.reset()

        self._p.setTimeStep(self._ts)
        self._p.setGravity(0, 0, -10)

        self.mesh_obj = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0125, height=0.2).rotate(np.array([[0,0, 1],
                                                                                                              [0,1, 0],
                                                                                                              [-1,0,0]]), center=[0,0,0])
        self.mesh_obj.compute_vertex_normals()
        self.mesh_obj.compute_triangle_normals()
        self.pcd_obj = self.mesh_obj.sample_points_poisson_disk(2048, use_triangle_normal=True)
        self.points = np.asarray(self.pcd_obj.points)
        self.normals = np.asarray(self.pcd_obj.normals)
        self.pcd_kd_tree = o3d.geometry.KDTreeFlann(self.pcd_obj)
        
        # Create two walls:
        o_id, floor_id, w1_id, w2_id, w3_id = create_groovepen(self._p)
        self.o_id = o_id
        self.w1_id = w1_id
        self.w2_id = w2_id
        self.w3_id = w3_id
        self.floor_id = floor_id
        self.wall_size = np.array([0.4,0.02,0.03])
        self._p.changeDynamics(self.w1_id, -1, lateralFriction=0.0)
        self._p.changeDynamics(self.w2_id, -1, lateralFriction=0.0)
        self._p.changeDynamics(self.w3_id, -1, lateralFriction=0.0)
        self._p.changeDynamics(self.floor_id, -1,
                               lateralFriction=20.0, restitution=0.0)            # TODO
        self._p.changeDynamics(self.o_id, -1,
                               lateralFriction=70.0, restitution=0.0)             # TODO

        self.tip_ids = []
        colors = [[1.0, 1.0, 1.0, 1.0],[0.0, 0.0, 1.0, 1.0],[0.0, 1.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0],[1.0,1.0,0,1.0]]
        for finger in self.all_fingers:
            color = colors[finger]

            size = [0.01, 0.01, 0.011] if finger == 0 else [0.01, 0.01, 0.011]       # thumb larger
            
            init_xyz = (finger+2.0, finger+2.0, finger+2.0) if finger in self.active_finger_tips else (100,100,100)
            tip_id = rb.create_primitive_shape(self._p, 0.01, pybullet.GEOM_BOX, size,         # half-extend
                                               color=color, collidable=True,
                                               init_xyz=init_xyz,
                                               init_quat=(0, 0, 0, 1))
            self.tip_ids.append(tip_id)
            self._p.changeDynamics(tip_id, -1,
                                   lateralFriction=100.0, restitution=0.0)                       # TODO
            self._p.setCollisionFilterPair(tip_id, self.floor_id, -1, -1, 0)
        
        self.tip_cids = []

        # True means last control step cp was on (fixed pos)
        self.last_fins = [([0.0] * 3, False)] * self.num_fingertips
        self.cur_fins = [([0.0] * 3, False)] * self.num_fingertips
        self.all_active_fins = []

        self._p.stepSimulation()

        self.timer = 0
        self.c_step_timer = 0
        self.step_cnt = 0
        self.last_action = np.zeros(self.action_dim)
        obs = self.get_extended_observation()
        self.initialize_deocclude_module()
        self.last_surface_norm = {
            0:None,
            1:None,
            2:None,
            3:None}
        return obs

    def step(self, a: List[float], rollout_verify=False):

        #  a new small-blocks contac env:
        #  1. Collidable when turned on & clearance; non-collidable otherwise
        #  2. when block on, add a pushing force (6D) to the block center
        #  		control with delta position control / root constraint
        #  		this can interpolate as well

        # remove existing constrs if any
        # turn off collision
        # reset tip location
        # add constrs
        # simulate one step (without collision)
        # if on, turn on collision, add target velocity

        # print("stepping")
        contact_state = self.path[self.step_cnt]
        prev_contact_state = None if self.step_cnt == 0 else self.path[self.step_cnt-1]
        self.step_cnt += 1

        # TODO: during step, small block might slide away
        # Compansate for thickness of the object
        # This face should be a surface norm
        def get_small_block_location_local(surface_norm: int, this_pos_local: List[float]) -> List[float]:
            # size = [0.02, 0.02, 0.01] if i == 0 else [0.01, 0.01, 0.01]       # thumb larger
            this_pos_local = np.array(this_pos_local)
            return list(this_pos_local + 0.01 * surface_norm)

        # How action are map to change the physical world. Very important
        def transform_a_to_action_single_pc(sub_a: List[float],
                                            fin_ind: int,
                                            previous_fins: List[Tuple[List[float], bool]]) -> List[float]:
            # face = None         # 0 top, 1 side, 2 bottom face
            assert len(sub_a) == self.single_action_dim

            # ======= Notice Finger tip pose at this stage is represented as local coordinate =======
            # If a finger tip is in contact in previous step, it does not allow to change relative position
            if previous_fins[self.active_finger_tips_idx[fin_ind]][1]: # Contact Flag (pos, contact_flag)
                pos_vec = (previous_fins[self.active_finger_tips_idx[fin_ind]][0]).copy()
                assert not (self.last_surface_norm is None)
                surface_norm = self.last_surface_norm[fin_ind]
            else: # Currently we always use split region
                  # [-1, 1] -> [-0.2, 0.2]
                pos_vec, surface_norm = self.contact_region.parse_sub_action(contact_state, fin_ind, sub_a[-3:-1], self.csg)
                pos_vec = get_small_block_location_local(surface_norm, pos_vec.copy())
                self.last_surface_norm[fin_ind] = surface_norm

            pos_force_vec = []
            for num_interp in range(self.num_interp_f): # Deltas self.num_interp_f = 7 # All for force
                idx_start = num_interp * 3
                
                # Applied to all interp, no matter it will be used for force or position
                # If the "force" is pointing into the object, then force is allowed to apply otherwise no
                if sub_a[idx_start] > 0:
                    v_normal = sub_a[idx_start] * 5.0 * self._ts    # [0, 1] -> [0, 5/250]  (max 5m/s target velocity)
                    v_t1 = sub_a[idx_start + 1] * 1.0 * v_normal        # TODO: make sense??
                    v_t2 = sub_a[idx_start + 2] * 1.0 * v_normal        # The larger the normal  inward force the larger friction will be allowed
                else:
                    v_normal = v_t1 = v_t2 = 0.0 # Eric: What does it means by setting v to zero? Will remain on previous location

                # With respect to local coordinate frame
                # TODO: May need more general representation
                
                R = rotation_matrix_from_vectors(np.array([0., 0., 1.]), surface_norm)
                v = -v_normal * surface_norm + R @ np.array([v_t1, v_t2, 0])
                pos_force_vec += v.tolist()

            pos_force_vec += pos_vec
            if (prev_contact_state != None and self.csg.getState(contact_state)[fin_ind] != self.csg.getState(prev_contact_state)[fin_ind]) and previous_fins[self.active_finger_tips_idx[fin_ind]][1]:
                pos_force_vec += [0.0]
            else:
                pos_force_vec += [1.0] if sub_a[-1] > 0 else [0.0]

            # Current decision will be effective next time.
            assert len(pos_force_vec) == self.single_action_dim + 1
            return pos_force_vec

        # Linear interpolate "force" or virtual velocity between the interpolant
        def calc_force_interp_from_fvec(vec: List[float]) -> np.ndarray:
            # for a single contact point
            # f_vec is 3*num_interp_f length 1d
            # return y, which is 3-by-control_skip matrix of the forces
            # Linear segment wise interpolation
            assert len(vec) == self.num_interp_f * 3
            y_interp = np.zeros((3, self.control_skip))
            x_interp = np.array(range(self.control_skip))

            # this returns [0] if num_interp_f == 1
            x_d = np.linspace(0, float(self.control_skip), self.num_interp_f, endpoint=True) # Eric: Time?
            y_d = np.reshape(vec, (self.num_interp_f, 3)).T 

            for row in range(3):
                y_interp[row, :] = np.interp(x_interp, x_d, y_d[row, :])    # use default extrapolation

            return y_interp

        def transform_a_no_opt_time(act: List[float],
                                    previous_fins: List[Tuple[List[float], bool]]) \
                -> Tuple[
                    List[Tuple[List[float], bool]],
                    List[np.ndarray]
                ]:

            current_fins = []
            interped_fs = []

            for idx in range(self.num_fingertips):
                f_vec = transform_a_to_action_single_pc(act[idx * self.single_action_dim:
                                                            (idx + 1) * self.single_action_dim],
                                                        self.active_finger_tips_idx[idx],
                                                        previous_fins)
                if f_vec[-1]==0:
                    f_vec[:-4] = [0.0] * len(f_vec[:-4]) 
                    f_vec[-4:-1] = [100.0, 100.0, 100.0]

                if f_vec[0]==0:
                    f_vec[:-4] = [0.0] * len(f_vec[:-4])

                f_interp = calc_force_interp_from_fvec(f_vec[:-4]) # Force information comes from [:-4]
                interped_fs.append(f_interp)  # local pos, local tar vel
                current_fins.append((f_vec[-4:-1], f_vec[-1] > 0)) # A 3D vector with a flag, position information comes from [-4:-1] contact flag is last bit

            return current_fins, interped_fs

        def calc_reward_value_stage_one():
            pos_1, quat_1 = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)
            pos = np.array(pos_1)
            quat = np.array(quat_1)
            extra_cond = np.zeros((1, 3 * len(self.active_finger_tips)))

            for i, fin in enumerate(self.active_finger_tips):
                extra_cond[0, i*3:(i+1)*3] = self.project_to_pcd(self.cur_fins[fin][0])
            rot_matrix = convert_q_bullet_to_matrix(quat)
            pcd_obj = copy.deepcopy(self.pcd_obj).rotate(rot_matrix, center=[0,0,0])
            pcd_obj.translate(pos)

            pcd_obj = self.deocclude(pcd_obj)
            if self.has_distance_field:
                df = self.compute_distance_field(pcd_obj)
            else:
                df = None

            pcd_obj.translate(-pos)
            pcd_obj.rotate(rot_matrix.T, center=[0.0, 0.0, 0.0])

            ave_score = 0
            if self.mode == "only_score":
                score, _ = self.score_function.pred_score(pcd_obj, extra_cond=extra_cond, distance_field=df)
                ave_score += score
            else:
                sigma_points = getSGSigmapoints(self.score_function.get_latent_size(), self.sigma)
                self.score_function.create_kd_tree(pcd_obj)
                for i in range(1):
                    score, _ = self.score_function.pred_score(pcd_obj, sigma_points[i], extra_cond = extra_cond)
                    ave_score += score
            return ave_score

        def get_canonical_point_cloud():
            pos_1, quat_1 = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)
            pos = np.array(pos_1)
            quat = np.array(quat_1)
            rot_matrix = convert_q_bullet_to_matrix(quat)
            pcd_obj = copy.deepcopy(self.pcd_obj).rotate(rot_matrix, center=[0.0, 0.0, 0.0])
            pcd_obj.translate(pos)
            
            # Remove all parts in occlusion
            pcd_obj = self.deocclude(pcd_obj)
            pcd_obj.translate(-pos)
            pcd_obj.rotate(rot_matrix.T, center=[0.0, 0.0, 0.0])
            return pcd_obj

        # Apply constraints on new contact points..
        def pre_simulate(current_fins: List[Tuple[List[float], bool]]):
            # What is the purpose of this function?
            for cid in self.tip_cids:
                self._p.removeConstraint(cid)
            self.tip_cids = []
            for idx in range(self.num_fingertips):
                # Transform the finger tip toward the world frame, here current_fin is expressed in local frame?
                f_pos_g, _ = self._p.multiplyTransforms(pos, quat, current_fins[idx][0], [0, 0, 0, 1])
                if not self.checkClearance(f_pos_g): # If the clearance is two small before next step of simulation, then the 
                    f_pos_g = [100.0, 100.0, 100.0]
                self._p.setCollisionFilterPair(self.tip_ids[idx], self.o_id, -1, -1, 0)
                self._p.resetBasePositionAndOrientation(self.tip_ids[idx],
                                                        f_pos_g,
                                                        quat)
                # Connect finger tips to the object for this simulation step
                cid = self._p.createConstraint(self.tip_ids[idx], -1, -1, -1, pybullet.JOINT_FIXED,
                                               [0, 0, 0], [0, 0, 0],
                                               childFramePosition=f_pos_g,
                                               childFrameOrientation=quat)
                self.tip_cids.append(cid)

            self._p.stepSimulation()

            for idx in range(self.num_fingertips):
                # can turn on all collisions since disabled fingers moved to far away already
                self._p.setCollisionFilterPair(self.tip_ids[idx], self.o_id, -1, -1, 1)

        # Starting pose of the object
        pos, quat = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)
        self.last_action = a.copy()
        a = np.tanh(a)      # [-1, 1]
        if self.opt_time:
            a_time = a[-1]      # [-1, 1]
            self.control_skip = int((a_time + 1.0) * 100.0 + 25.0)

        self.cur_fins, self.interped_fs = transform_a_no_opt_time(a, self.last_fins)

        pre_simulate(self.cur_fins)

        ave_r = 0.0

        if not self.train:
            object_poses = []
            tip_poses = []
            images = []
        for t in range(self.control_skip):
            pos, quat = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)
            if t == 0 and self.c_step_timer == 0 and not self.train:
                init_pcd = get_canonical_point_cloud()
                init_df = self.compute_distance_field(init_pcd)
                o3d.io.write_point_cloud(f"{currentdir}/assets/init_pcds/groovepen.ply",init_pcd)
                np.save(f"{currentdir}/assets/init_dfs/groovepen.npy", init_df)
                np.save(f"{currentdir}/assets/init_pose/groovepen.npy", np.hstack([pos,quat]))
            if not self.train:
                tip_pose = []
                for idx in range(self.num_fingertips):
                    tip_pose_candidate = list(self._p.multiplyTransforms(pos, quat, self.cur_fins[idx][0], [0, 0, 0, 1])[0])
                    local_normals = self.compute_tip_normals(self.cur_fins[idx][0])
                    tip_pose_candidate += list(self._p.multiplyTransforms([0, 0, 0], quat, local_normals, [0, 0, 0, 1])[0])
                    tip_pose.append(tip_pose_candidate)
                tip_poses.append(tip_pose) # position only
                object_poses.append(pos+quat) # 7
            vel = rb.get_link_com_linear_velocity(self._p, self.o_id, -1)

            for i, f_tuple in enumerate(self.interped_fs):
                # "Force" in global frame
                v_g_s, _ = self._p.multiplyTransforms([0., 0., 0.], quat, f_tuple[:, t], [0, 0, 0, 1])
                v_g = np.array(v_g_s) + np.array(vel) * self._ts # Follow the rigid body movement
                tar_pos_g, _ = self._p.multiplyTransforms(pos, quat, self.cur_fins[i][0], [0, 0, 0, 1])
                tar_pos_g = np.array(tar_pos_g)
                tar_pos_g = list(tar_pos_g + v_g)
                self._p.changeConstraint(self.tip_cids[i], tar_pos_g, quat, maxForce=self.max_forces, erp=0.9)

            self._p.stepSimulation()

            ave_r += 0 #calc_reward_value()

            if self.render:
                if self.showImage:
                    time.sleep(self._ts * 20.0)
                image = self.renderer.render()
                images.append(image)
            self.timer += 1

        self.c_step_timer += 1

        com_x = 0
        final_r = 0

        pcd_output = None
        # Let the system evolve naturally Then stabilize the system for final eval
        # Only avaiable in final epoch
        if not self.train and self.add_physics and self.c_step_timer == self.n_steps:
            obj_pos, obj_orn = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)
            self.setup_grasp(obj_pos, obj_orn)
        if self.c_step_timer == self.n_steps:
            pcd_output = get_canonical_point_cloud()
            com_x = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)[0][2] * 0.5
            for _ in range(50 if self.train else 0):
                pos, quat = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)
                if not self.train and self.add_physics:
                    self.update_grasp(pos, quat)
                    tip_pose = []
                    for idx in range(len(self.all_fingers)):
                        tip_pose_candidate = list(self._p.getBasePositionAndOrientation(self.tip_ids[idx])[0])
                        # Get normal using local coordinate
                        local_normals = self.compute_tip_normals(self.cur_fins[idx][0] if idx < self.num_fingertips else self.grasp_pos[idx])
                        tip_pose_candidate += list(self._p.multiplyTransforms([0,0,0], quat, local_normals, [0, 0, 0, 1])[0])
                        tip_pose.append(tip_pose_candidate)
                    tip_poses.append(tip_pose) # position + tip_normals 
                    object_poses.append(pos+quat) # 7
                self._p.stepSimulation()
                final_r += 0 #calc_reward_value()

                if self.render:
                    if self.showImage:
                        time.sleep(self._ts * 4.0)
                    image = self.renderer.render()
            delta = np.array([0, 0, 0.2])/100
            pos, quat = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)
            if not self.train and self.add_physics:
                self.get_ref_tip_pose(pos,quat)
            for t in range(50 if self.train else 100):
                pos, quat = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)
                if not self.train and self.add_physics:
                    self.update_constraint_with_delta(quat,delta=delta)
                    tip_pose = []
                    for idx in range(len(self.all_fingers)):
                        tip_pose_candidate = list(self._p.getBasePositionAndOrientation(self.tip_ids[idx])[0])
                        # Get normal using local coordinate
                        local_normals = self.compute_tip_normals(self.cur_fins[idx][0] if idx < self.num_fingertips else self.grasp_pos[idx])
                        tip_pose_candidate += list(self._p.multiplyTransforms([0,0,0], quat, local_normals, [0, 0, 0, 1])[0])
                        tip_pose.append(tip_pose_candidate)
                    tip_poses.append(tip_pose) # position + tip_normals 
                    object_poses.append(pos+quat) # 7
                self._p.stepSimulation()
                final_r += 0 #calc_reward_value()

                if self.render:
                    if self.showImage:
                        time.sleep(self._ts * 4.0)
                    image = self.renderer.render()

            for _ in range(150 if not self.train else 200):
                self._p.stepSimulation()
                final_r += 0 #calc_reward_value()
                if self.render:
                    if self.showImage:
                        time.sleep(self._ts * 4.0)
                    self.renderer.render()
            com_x += rb.get_link_com_xyz_orn(self._p, self.o_id, -1)[0][2] * 0.5
        obs = self.get_extended_observation()

        ave_r /= self.control_skip
        #ave_r -= cost_remaining * 10000.0     # 0.01 * 10000 ~ 100 10000
        if self.train:
            stage_one_reward = calc_reward_value_stage_one()
            ave_r += stage_one_reward * 100
        if rollout_verify:
            print("Finger tip pose:", self.cur_fins)
        ave_r += final_r / 300.0 * 2.0

        fin_pos = []
        for idx in range(self.num_fingertips):
            if self.cur_fins[idx][1]:
                fin_pos.append(self.cur_fins[idx][0])
            else:
                fin_pos.append(None)
        self.all_active_fins.append(fin_pos)

        self.last_fins = self.cur_fins.copy()
        done = False
        if self.c_step_timer == self.n_steps:
            done = True
        if self.train==False:
            return obs, ave_r, done, {"finger_pos":tip_poses,"object_pose":object_poses, "last_fins":self.last_fins, "images":images, "metric":com_x, "pcd":pcd_output}
        else:
            return obs, ave_r, done, {"metric": com_x}

    def get_extended_observation(self) -> List[float]:
        # It is essentially position of all corners of the object
        # It can be replaced by point cloud for more general object
        # It contains pose for only one step
        dim = [0.4, 0.4, 0.1]
        local_corners = [[dim[0] / 2, dim[1] / 2, dim[2] / 2] for _ in range(8)]
        local_corners[1] = [-dim[0] / 2, dim[1] / 2, dim[2] / 2]
        local_corners[2] = [dim[0] / 2, -dim[1] / 2, dim[2] / 2]
        local_corners[3] = [dim[0] / 2, dim[1] / 2, -dim[2] / 2]
        local_corners[4] = [-dim[0] / 2, -dim[1] / 2, dim[2] / 2]
        local_corners[5] = [-dim[0] / 2, dim[1] / 2, -dim[2] / 2]
        local_corners[6] = [dim[0] / 2, -dim[1] / 2, -dim[2] / 2]
        local_corners[7] = [-dim[0] / 2, -dim[1] / 2, -dim[2] / 2]

        obs = []
        pos, quat = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)
        for j in range(8):
            obs += list(self._p.multiplyTransforms(pos, quat, local_corners[j], [0, 0, 0, 1])[0])

        if self.observe_last_action:
            obs += list(self.last_action)
        return obs

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def checkClearance(self, point):
        dist2floor = point[2]
        if dist2floor < CLEARANCE_H:
            return False
        elif minimumDist2dBox(self._p, self.w1_id, self.wall_size, point) < CLEARANCE_H:
            return False
        elif minimumDist2dBox(self._p, self.w2_id, self.wall_size, point) < CLEARANCE_H:
            return False
        else:
            return True

    def initialize_deocclude_module(self):
        """
        For deocclusion module, we need multiple bounding boxes
        and take the union of points in all of them
        """
        self.bounding_boxes = []
        # self.bounding_boxes.append(o3d.geometry.AxisAlignedBoundingBox(min_bound = np.array([0.0, -0.02, 0.002]),
        #                                           max_bound = np.array([0.4, 0.02, 0.03])))
        self.bounding_boxes.append(o3d.geometry.AxisAlignedBoundingBox(min_bound = np.array([-100, -100, 0.001]),
                                                  max_bound = np.array([100, 100, 100])))
        # self.bounding_boxes.append(o3d.geometry.AxisAlignedBoundingBox(min_bound = np.array([-100, -100, 0.005]),
        #                                           max_bound = np.array([-0.205, 100, 100])))
    
    def deocclude(self, pcd):
        pcd_points = []
        pcd_normals = []
        for bb in self.bounding_boxes:
            crop_pcd = copy.deepcopy(pcd).crop(bb)
            pcd_points.append(np.asarray(crop_pcd.points))
            pcd_normals.append(np.asarray(crop_pcd.normals))
        pcd_points = np.vstack(pcd_points)
        pcd_normals = np.vstack(pcd_normals)
        new_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points))
        new_pcd.normals = o3d.utility.Vector3dVector(pcd_normals)
        return new_pcd

    def project_to_pcd(self, point):
        if self.checkClearance(point):
            p = point.copy()
            idx = self.pcd_kd_tree.search_knn_vector_3d(p, 1)[1]
            return self.points[idx]
        else:
            return np.array([100, 100, 100])

    def compute_tip_normals(self, point):
        p = point.copy()
        idx = self.pcd_kd_tree.search_knn_vector_3d(point, 1)[1]
        return self.normals[idx].flatten().copy()

    def compute_distance_field(self, pcd):
        points = np.asarray(pcd.points)
        df = self.dist_field_env.get_points_distance(points)
        return df

    def load_grasp(self, grasp, finger_idx=[0,2,3,4]):
        self.grasp_pos = grasp[:,:3] + 0.01 * grasp[:,3:]
        self.grasp_idx = finger_idx

    def setup_grasp(self,pos, quat):
        self.grasp_constraint = []
        for i,idx in enumerate(self.grasp_idx):
            tip = self.grasp_pos[idx]
            tip_pose, _ = self._p.multiplyTransforms(pos, quat, tip, [0, 0, 0, 1]) # Position of finger tip in world coordinate
            self._p.setCollisionFilterPair(self.tip_ids[idx], self.o_id, -1, -1, 0)
            self._p.resetBasePositionAndOrientation(self.tip_ids[idx],
                                                    tip_pose,
                                                    quat)
            # Connect finger tips to the object for this simulation step
            c = self._p.createConstraint(self.tip_ids[idx], -1, -1, -1, pybullet.JOINT_FIXED,
                                            [0, 0, 0], [0, 0, 0],
                                            childFramePosition=tip_pose,
                                            childFrameOrientation=quat)
            self.grasp_constraint.append(c)
        self._p.stepSimulation()
        for i, idx in enumerate(self.grasp_idx):
            self._p.setCollisionFilterPair(self.tip_ids[idx],self.o_id, -1, -1, 1)

    def update_grasp(self, pos, quat):
        self._p.changeConstraint(self.tip_cids[0], maxForce=0)
        for i,idx in enumerate(self.grasp_idx):
            tip = self.grasp_pos[idx]
            tip_pose, _ = self._p.multiplyTransforms(pos, quat, tip, [0, 0, 0, 1])
            tip_pose = np.array(tip_pose)
            self._p.changeConstraint(self.grasp_constraint[i], tip_pose, quat, maxForce=self.max_forces, erp=0.9)

    def get_ref_tip_pose(self,pos,quat):
        self.ref_poses_cond = []
        self.ref_quat = copy.deepcopy(quat)
        for i in range(self.num_fingertips):
            tip_pose,_ = self._p.multiplyTransforms(pos, quat, self.cur_fins[i][0], [0, 0, 0, 1])
            self.ref_poses_cond.append(np.array(tip_pose))
        self.ref_poses_grasp = []
        for i, idx in enumerate(self.grasp_idx):
            tip = self.grasp_pos[idx]
            tip_pose, _ = self._p.multiplyTransforms(pos, quat, tip, [0, 0, 0, 1])
            self.ref_poses_grasp.append(np.array(tip_pose))
        self._p.changeDynamics(self.o_id,-1,mass=0.3) # Have very good effect!

    def update_constraint_with_delta(self,quat,delta=None):
        for i in range(self.num_fingertips):
            tip_pose = self.ref_poses_cond[i]
            if delta is not None:
                tip_pose += delta
            if i == 0:
                self._p.changeConstraint(self.tip_cids[i], maxForce=0)
            else:
                self._p.changeConstraint(self.tip_cids[i], tip_pose, quat, maxForce=self.max_forces, erp=0.9)

        for i, idx in enumerate(self.grasp_idx):
            tip_pose = self.ref_poses_grasp[i]
            if delta is not None:
                tip_pose += delta
            self._p.changeConstraint(self.grasp_constraint[i],tip_pose, quat, maxForce=self.max_forces, erp=0.9)


            
            


