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

from pybullet_utils import bullet_client
import open3d as o3d
import pybullet
import utils.render as r
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np

import os
import inspect

from typing import List, Tuple

import utils.rigidBodySento as rb

from scipy.optimize import minimize
import  model.param as model_param 
from utils.math_utils import rotation_matrix_from_vectors
from utils.small_block_region import SmallBlockRegionDummy
from utils.contact_state_graph import ContactStateGraph

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
CLEARANCE_H = 0.05

class LaptopBulletEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 control_skip=50,
                 num_fingertips=4,
                 num_interp_f=5,
                 opt_time=False,
                 last_fins=None,
                 init_obj_pose=None,
                 task=np.array([0, 0, 1]),
                 train=True,
                 steps=3,
                 observe_last_action=False,
                 path=None):
        self.train = train
        self.observe_last_action=observe_last_action
        self.max_forces = model_param.MAX_FORCE
        self.init_obj_pose=init_obj_pose
        self.render = render
        self.init_noise = init_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 250. # A constant
        self.num_fingertips = num_fingertips
        self.num_interp_f = num_interp_f
        self.csg = ContactStateGraph(np.load("data/contact_states/laptop_env/dummy_states_2.npy"))
        self.contact_region = SmallBlockRegionDummy(self.csg)
        self.path = path
        assert num_fingertips == 4         # TODO: 3/4/5?
        self.opt_time = opt_time
        self.task = task

        self.n_steps = steps
        self.last_surface_norm = {
            0:None,
            1:None,
            2:None,
            3:None}

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
            self.renderer = r.PyBulletRenderer()
        else:
            self._p = bullet_client.BulletClient()

        self.np_random = None
        self.o_id = None

        self.cps_vids = []
        self.hand_bones_vids = []

        self.tip_ids = []
        self.tip_cids = []

        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0
        self.c_step_timer = 0

        self.floor_id = None

        self.interped_fs = [np.zeros((3, self.control_skip))] * num_fingertips
        # True means last control step cp was on (fixed pos) initialize
        self.last_fins = [([0.0] * 3, False)] * num_fingertips
        if isinstance(last_fins, np.ndarray):
            for i in range(num_fingertips):
                self.last_fins[i] = (last_fins[i,:3], bool(last_fins[i,3]))
        self.cur_fins = [([0.0] * 3, False)] * num_fingertips

        self.all_active_fins = [[[0.0] * 3] * num_fingertips] * self.n_steps

        self.single_action_dim = num_interp_f * 3 + 3               # f position (2) & on/off (1)
        self.action_dim = self.single_action_dim * num_fingertips
        if self.opt_time:
            self.action_dim += 1

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

        self.floor_id = self._p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1)

        if isinstance(self.init_obj_pose, np.ndarray):
            init_xyz = self.init_obj_pose[:3]
            init_orn = self.init_obj_pose[3:]
        else:
            init_xyz = np.array([0, 0, 0.05])
            init_orn = np.array([0, 0, 0, 1])
        self.o_id = rb.create_primitive_shape(self._p, 1.0, pybullet.GEOM_BOX, (0.2, 0.2, 0.05),         # half-extend
                                              color=(0.6, 0, 0, 0.8), collidable=True,
                                              init_xyz=init_xyz,
                                              init_quat=init_orn)
        # Need to create corresponding pointcloud
        self._p.changeDynamics(self.floor_id, -1,
                               lateralFriction=70.0, restitution=0.0)            # TODO
        self._p.changeDynamics(self.o_id, -1,
                               lateralFriction=70.0, restitution=0.0)             # TODO

        self.cps_vids = []
        for i in range(10):
            color = [0.5, 0.0, 1.0, 1.0]
            visual_id = self._p.createVisualShape(self._p.GEOM_SPHERE,
                                                  radius=0.02,
                                                  rgbaColor=color,
                                                  specularColor=[1, 1, 1])
            bid = self._p.createMultiBody(0.0,
                                          -1,
                                          visual_id,
                                          [100.0, 100.0, 100.0],
                                          [0.0, 0.0, 0.0, 1.0])
            self.cps_vids.append(bid)
            # self._p.setCollisionFilterGroupMask(bid, -1, 0, 0)

        self.tip_ids = []
        colors = [[1.0, 1.0, 1.0, 1.0],[0.0, 0.0, 1.0, 1.0],[0.0, 1.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0]]
        for i in range(self.num_fingertips):
            color = colors[i]

            size = [0.02, 0.02, 0.01] if i == 0 else [0.01, 0.01, 0.01]       # thumb larger

            tip_id = rb.create_primitive_shape(self._p, 0.01, pybullet.GEOM_BOX, size,         # half-extend
                                               color=color, collidable=True,
                                               init_xyz=(i+2.0, i+2.0, i+2.0),
                                               init_quat=(0, 0, 0, 1))
            self.tip_ids.append(tip_id)
            self._p.changeDynamics(tip_id, -1,
                                   lateralFriction=30.0, restitution=0.0)                       # TODO
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
        self.last_surface_norm = {
            0:None,
            1:None,
            2:None,
            3:None}
        return obs

    def get_hand_pose(self,
                      finger_infos: List[Tuple[List[float], bool]],
                      last_finger_infos: List[Tuple[List[float], bool]])\
                      -> Tuple[List[float], List[List[float]], float]:
        # hand / fingers pose in object frame

        #assert self.use_split_region

        # offsets in object frame
        # Eric: Hard coded offset, Need to get the position of finger tip here
        offsets = np.array([
            [+0.25, -0.03, 0],  # thumb
            [+0.25, +0.05, +0.1],  # ring
            [+0.25, 0.0, +0.1],  # middle
            [+0.25, -0.05, +0.1],  # index
        ])

        proposals = np.empty((0, 3), float)
        for fin_ind, fin_info in enumerate(finger_infos):
            if fin_info[1]:
                proposals = np.vstack((proposals, np.array(fin_info[0]) + offsets[fin_ind, :]))
            elif last_finger_infos[fin_ind][1]:
                proposals = np.vstack((proposals, np.array(last_finger_infos[fin_ind][0]) + offsets[fin_ind, :]))

        if proposals.shape[0] == 0:
            # no fixed fingers
            mean = np.array([0.5, 0.0, 0.3])
            cost = 0.0
        else:
            mean = proposals.mean(axis=0)
            cost = proposals.var(axis=0).sum()
            #print(proposals.var(axis=0))

        fin_pos = []
        for fin_ind in range(self.num_fingertips):
            if finger_infos[fin_ind][1]:
                fin_pos.append(finger_infos[fin_ind][0])
            elif last_finger_infos[fin_ind][1]:
                fin_pos.append(last_finger_infos[fin_ind][0])
            else:
                fin_pos.append(list(mean - offsets[fin_ind, :]))

        return list(mean), fin_pos, cost

    def draw_cps_ground(self, cps):
        for j, cp in enumerate(cps):
            self._p.resetBasePositionAndOrientation(self.cps_vids[j],
                                                    cp[5],
                                                    [0.0, 0.0, 0.0, 1.0])
        for k in range(len(cps), 10):
            self._p.resetBasePositionAndOrientation(self.cps_vids[k],
                                                    [100.0, 100.0, 100.0],
                                                    [0.0, 0.0, 0.0, 1.0])

    def step(self, a: List[float], train=True):

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
        self.step_cnt += 1
        next_contact_state = None if self.step_cnt == len(self.path) else self.path[self.step_cnt]
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
            if previous_fins[fin_ind][1]: # Contact Flag (pos, contact_flag)
                # last contact is on, not allowed to change
                # Directly use previous position.
                # When not in contact we can control the position of the finger
                pos_vec = (previous_fins[fin_ind][0]).copy()

                # This is problematic, need to access previous surface norm..
                # if np.isclose(pos_vec[2], 0.06, 1e-5): # On the top
                #     surface_norm = np.array([0., 0., 1.])
                # elif 0.05001 > pos_vec[2] > -0.05001: # On the side
                #     surface_norm = np.array([1., 0., 0.])
                # else: # On the bottom
                #     assert np.isclose(pos_vec[2], -0.06, 1e-6)
                #     surface_norm = np.array([0., 0., -1.])
                assert not (self.last_surface_norm is None)
                surface_norm = self.last_surface_norm[fin_ind]
            else: # Currently we always use split region
                pos_vec, surface_norm = self.contact_region.parse_sub_action(contact_state, fin_ind, sub_a[-3:-1]) # It also need to output contact norm
                # Need to compute surface norm based on current pos_vec
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
            
            # if (next_contact_state is None) or (self.csg.getState(contact_state)[fin_ind] != self.csg.getState(next_contact_state)[fin_ind]):
            #     pos_force_vec += [0.0]
            # else:
            #     pos_force_vec += [1.0] if sub_a[-1] > 0 else [0.0]      # last bit on / off (non-colliding finger) if next contact region is different then directly break
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
                                                        idx,
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

        def calc_reward_value():
            # post obj config
            cps_1 = self._p.getContactPoints(self.o_id, self.floor_id, -1, -1) # contact points between floor and object
            pos_1, quat_1 = rb.get_link_com_xyz_orn(self._p, self.o_id, -1) # Object pose and orn
            vel_1 = rb.get_link_com_linear_velocity(self._p, self.o_id, -1) # object's linear velocity vec 3

            z_axis, _ = self._p.multiplyTransforms(
                [0, 0, 0], quat_1, [0, 0, 1], [0, 0, 0, 1]
            ) # Compute the object's z-axis vector in world frame
            rot_metric = np.array(z_axis).dot(self.task)        # in [-1,1] # z-axis need to tilted to a given angle cosine similarity should be raise up seems to be

            r  = - np.linalg.norm(vel_1) * 20 - len(cps_1) * 250 # As slow as possible, as less contact point as possible.
            r += - 300 * np.linalg.norm([pos_1[0], pos_1[1], pos_1[2] - 0.3]) # COM should be 0.3 meter off the ground. While x and y coordinate remain the same.
            r += 150 * rot_metric # hope the object can be rotate to a given z orientation
            return r

        # Apply constraints on new contact points..
        def pre_simulate(current_fins: List[Tuple[List[float], bool]]):
            # What is the purpose of this function?
            for cid in self.tip_cids:
                self._p.removeConstraint(cid)
            self.tip_cids = []
            for idx in range(self.num_fingertips):
                # Transform the finger tip toward the world frame, here current_fin is expressed in local frame?
                f_pos_g, _ = self._p.multiplyTransforms(pos, quat, current_fins[idx][0], [0, 0, 0, 1])
                if f_pos_g[2] < CLEARANCE_H:
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
        
        _,_, cost_remaining = self.get_hand_pose(self.cur_fins, self.last_fins)

        if not self.train:
            object_poses = []
            tip_poses = []
            images = []
        for t in range(self.control_skip):
            pos, quat = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)
            
            if not self.train:
                tip_pose = []
                for idx in range(self.num_fingertips):
                    tip_pose.append(list(self._p.multiplyTransforms(pos, quat, self.cur_fins[idx][0], [0, 0, 0, 1])[0]))
                tip_poses.append(tip_pose) # position only
                object_poses.append(pos+quat) # 7
            vel = rb.get_link_com_linear_velocity(self._p, self.o_id, -1)

            for i, f_tuple in enumerate(self.interped_fs):
                # "Force" in global frame
                v_g_s, _ = self._p.multiplyTransforms([0., 0., 0.], quat, f_tuple[:, t], [0, 0, 0, 1])
                v_g = np.array(v_g_s) + np.array(vel) * self._ts # Follow the rigid body movement
                tar_pos_g, _ = self._p.multiplyTransforms(pos, quat, self.cur_fins[i][0], [0, 0, 0, 1])
                # if tar_pos_g[2] < CLEARANCE_H: # Not allow the finger tip to penetrate the ground, this means that the break of the finger tip may happen in the middle
                #    tar_pos_g = [100.0, 100.0, 100.0]    # WARN: This may be problematic
                tar_pos_g = np.array(tar_pos_g)
                tar_pos_g = list(tar_pos_g + v_g)
                self._p.changeConstraint(self.tip_cids[i], tar_pos_g, quat, maxForce=self.max_forces, erp=0.9)

            self._p.stepSimulation()

            ave_r += calc_reward_value()

            if self.render:
                time.sleep(self._ts * 20.0)
                image = self.renderer.render()
                images.append(image)
            self.timer += 1

        self.c_step_timer += 1

        final_r = 0
        # Let the system evolve naturally Then stabilize the system for final eval
        # Only avaiable in final epoch
        if self.c_step_timer == self.n_steps:
            for _ in range(300):
                self._p.stepSimulation()
                final_r += calc_reward_value()

                if self.render:
                    time.sleep(self._ts * 4.0)
                    self.renderer.render()

        obs = self.get_extended_observation()

        ave_r /= self.control_skip
        ave_r -= cost_remaining * 10000.0     # 0.01 * 10000 ~ 100 10000
        ave_r += final_r / 300.0 * 2.0

        fin_pos = []
        for fin_ind in range(self.num_fingertips):
            if self.cur_fins[fin_ind][1]:
                fin_pos.append(self.cur_fins[fin_ind][0])
            else:
                fin_pos.append(None)
        self.all_active_fins.append(fin_pos)

        self.last_fins = self.cur_fins.copy()
        done = False
        if self.c_step_timer == self.n_steps:
            done = True
        if self.train==False:
            return obs, ave_r, done, {"finger_pos":tip_poses,"object_pose":object_poses, "last_fins":self.last_fins, "images":images}
        else:
            return obs, ave_r, done, {}

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
