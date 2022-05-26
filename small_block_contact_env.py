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
import pybullet
import render as r
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np

import os
import inspect

from typing import List, Tuple

import rigidBodySento as rb

from scipy.optimize import minimize

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

NOMINAL_POSE_BONES = [[-0.05, 0.03, 0.05],
                [-0.07, 0.01, 0.02],
                [-0.04, 0.0, 0.0],
                [-0.03, 0.0, 0.0],

                [-0.15, -0.05, -0.1],
                [-0.03, 0.0, 0.0],
                [-0.03, 0.0, 0.0],
                [-0.03, 0.0, 0.0],

                [-0.15, 0.0, -0.1],
                [-0.03, 0.0, 0.0],
                [-0.03, 0.0, 0.0],
                [-0.03, 0.0, 0.0],

                [-0.15, 0.05, -0.1],
                [-0.03, 0.0, 0.0],
                [-0.03, 0.0, 0.0],
                [-0.03, 0.0, 0.0],
                ]

NOMINAL_POSE_BONES = (np.array(NOMINAL_POSE_BONES) * 1.5).tolist()      # TODO: enlarge by 1.5

NOMINAL_POSE = NOMINAL_POSE_BONES.copy()

NOMINAL_LENS = [0.0] * 16
for ii in range(16):
    NOMINAL_LENS[ii] = np.linalg.norm(NOMINAL_POSE_BONES[ii])

for ii in range(4):
    for jj in range(1, 4):
        for kk in range(3):
            NOMINAL_POSE[ii*4 + jj][kk] += NOMINAL_POSE[ii*4 + jj - 1][kk]

CLEARANCE_H = 0.05

CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16]
]


class SmallBlockContactBulletEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 control_skip=50,
                 num_fingertips=4,
                 num_interp_f=5,
                 use_split_region=True,      # TODO: implement false
                 opt_time=False,
                 task=np.array([0, 0, 1]),
                 solve_hand=False,
                 max_forces=50):

        self.solve_hand = solve_hand
        self.max_forces = max_forces
        self.render = render
        self.init_noise = init_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 250. # A constant
        self.num_fingertips = num_fingertips
        self.num_interp_f = num_interp_f
        self.use_split_region = use_split_region
        if use_split_region:
            assert num_fingertips == 4         # TODO: 3/4/5?
        self.opt_time = opt_time
        self.task = task

        self.n_steps = 3            # TODO: hardcoded

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
            self.renderer = r.PyBulletRenderer(p_agent=self._p)
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
        # True means last control step cp was on (fixed pos)
        self.last_fins = [([0.0] * 3, False)] * num_fingertips
        self.cur_fins = [([0.0] * 3, False)] * num_fingertips

        self.all_active_fins = [[[0.0] * 3] * num_fingertips] * self.n_steps

        self.all_hand_poss = None

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

        self.o_id = rb.create_primitive_shape(self._p, 1.0, pybullet.GEOM_BOX, (0.2, 0.2, 0.05),         # half-extend
                                              color=(0.6, 0, 0, 0.8), collidable=True,
                                              init_xyz=(0, 0, 0.05),
                                              init_quat=(0, 0, 0, 1))

        self._p.changeDynamics(self.floor_id, -1,
                               lateralFriction=30.0, restitution=0.0)            # TODO
        self._p.changeDynamics(self.o_id, -1,
                               lateralFriction=30.0, restitution=0.0)             # TODO

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

        self.hand_bones_vids = []
        for l in NOMINAL_LENS:
            color = [0.2, 0.2, 0.2, 1.0]
            visual_id = self._p.createVisualShape(self._p.GEOM_CAPSULE,
                                                  0.01,
                                                  [1, 1, 1],       # dummy, seems a bullet bug
                                                  l,
                                                  rgbaColor=color,
                                                  specularColor=[1, 1, 1])
            bid = self._p.createMultiBody(0.0,
                                          -1,
                                          visual_id,
                                          [100.0, 100.0, 100.0],
                                          [0.0, 0.0, 0.0, 1.0])
            self.hand_bones_vids.append(bid)
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

        obs = self.get_extended_observation()

        return obs

    def draw_hand(self, j_local_poss:  List[List[float]]):
        # 17 * List3[Float]
        # first wrist pose, then four fingers * 4 joints

        # self._p.removeAllUserDebugItems()
        pos, quat = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)

        j_g_poss = [self._p.multiplyTransforms(pos, quat, j_local_pos, [0, 0, 0, 1])[0] for j_local_pos in j_local_poss]

        lens = []

        bone_idx = 0
        for i in range(self.num_fingertips):
            sp = j_g_poss[0]
            for j in range(4):
                ep = j_g_poss[i*4 + j + 1]
                # self._p.addUserDebugLine(sp, ep, [0.2, 0.2, 0.2], 5)
                lens.append(np.linalg.norm(np.array(ep) - sp))

                b_midpoint = (np.array(ep) + sp) / 2.0
                b_u_vec = (np.array(ep) - sp) / np.linalg.norm(np.array(ep) - sp)
                # capsule should initially point to +z
                b_quat = (-b_u_vec[1] / np.sqrt(2), b_u_vec[0] / np.sqrt(2), 0.0, (1+b_u_vec[2]) / np.sqrt(2))
                # Each link is considered as a independent object, not articulated
                # The length of each bone for render is pre-determined.
                self._p.resetBasePositionAndOrientation(self.hand_bones_vids[bone_idx],
                                                        b_midpoint,
                                                        b_quat)
                bone_idx += 1
                sp = ep

    def draw_cps_ground(self, cps):
        for j, cp in enumerate(cps):
            self._p.resetBasePositionAndOrientation(self.cps_vids[j],
                                                    cp[5],
                                                    [0.0, 0.0, 0.0, 1.0])
        for k in range(len(cps), 10):
            self._p.resetBasePositionAndOrientation(self.cps_vids[k],
                                                    [100.0, 100.0, 100.0],
                                                    [0.0, 0.0, 0.0, 1.0])

    def get_hand_pose(self,
                      finger_infos: List[Tuple[List[float], bool]],
                      last_finger_infos: List[Tuple[List[float], bool]])\
            -> Tuple[List[float], List[List[float]], float]:
        # hand / fingers pose in object frame

        assert self.use_split_region

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

        fin_pos = []
        for fin_ind in range(self.num_fingertips):
            if finger_infos[fin_ind][1]:
                fin_pos.append(finger_infos[fin_ind][0])
            elif last_finger_infos[fin_ind][1]:
                fin_pos.append(last_finger_infos[fin_ind][0])
            else:
                fin_pos.append(list(mean - offsets[fin_ind, :]))

        return list(mean), fin_pos, cost

    def get_hand_pose_full_with_optimization(self) -> Tuple[float,  np.ndarray]:
        # calc this at the end of final control step
        # require all active fintip locations (past to present) 5*List[List[float]/None]
        # optimization var: 17 joints (17*3) * 15 (in LOCAL frame)
        # Collectively they construct linearly interpolated traj for each joint

        # from which we should be able to eval bone-length violation & collision violation at traj points
        # and run optimizer on that

        def calc_bone_length_loss(h_j_vec: List[List[float]]) -> float:
            # h_j_vec in 17 * List3[Float]
            cost = 0.0

            for fin_ind in range(self.num_fingertips):
                chain = CHAINS[fin_ind].copy()
                for j in range(0, len(chain) - 1):
                    # j in 0, ..., n-2 (n == 5)
                    # The bone is defined as subtraction of control points
                    diff = np.array(h_j_vec[chain[j+1]]) - np.array(h_j_vec[chain[j]])
                    # The bone length should be close to nominal length
                    cost += (NOMINAL_LENS[fin_ind * 4 + j] - np.linalg.norm(diff)) ** 2

            return cost

        def calc_collision_loss_box(h_j_vec: List[List[float]]) -> float:
            # h_j_vec in 17 * List3[Float]

            cost = 0.0

            for fin_ind in range(self.num_fingertips):
                chain = CHAINS[fin_ind].copy()
                for j in range(0, len(chain) - 1):
                    # j in 0, ..., n-2 (n == 5)

                    # https://slidetodoc.com/3-4-contact-generation-generating-contacts-between-rigid/
                    # separating axis
                    center = (np.array(h_j_vec[chain[j+1]]) + np.array(h_j_vec[chain[j]])) / 2

                    # print(center)
                    sp_dist = np.linalg.norm(center)    # from center to box center (0)

                    center_axis = center / sp_dist
                    proj_dist = np.abs(center_axis[0]) * 0.2 + np.abs(center_axis[1]) * 0.2 + np.abs(center_axis[2]) * 0.05

                    if sp_dist > proj_dist:
                        # no collision
                        # cost += np.maximum(-0.1 * (sp_dist - proj_dist) ** 2, -0.001)
                        cost += 0.0
                    else:
                        # in interpenetration
                        cost += (sp_dist - proj_dist) ** 2

            return cost

        def obj_func(x: List[float]) -> float:
            return get_obj_value_and_hand_poses(x)[0]

        def get_obj_value_and_hand_poses(x: List[float]) -> Tuple[float,  np.ndarray]:
            # x in (3 * 5) * 17 * 3
            x_np = np.array(x)
            x_mat = x_np.reshape((3*self.n_steps, len(NOMINAL_LENS)+1, 3))

            x_dummy = np.zeros((1, len(NOMINAL_LENS)+1, 3))

            x_mat = np.concatenate((x_mat, x_dummy), axis=0)

            for t in range(self.n_steps):
                tips_loc = self.all_active_fins[t]     # List[List[float]/None]
                for fin_ind in range(self.num_fingertips):
                    if tips_loc[fin_ind] is not None:
                        tip_loc_np = np.array(tips_loc[fin_ind]).reshape((1, 1, 3))
                        tip_loc_repeat = tip_loc_np.repeat(repeats=4, axis=0)       # 4*1*3
                        fin_joint_ind = CHAINS[fin_ind][-1]
                        x_mat[t*3: t*3+4, fin_joint_ind: fin_joint_ind+1, :] = tip_loc_repeat
                        # Position of last control point need to be on the object
                        # Hard constraints

            cost_total = 0.0
            for sub_t in range(3*self.n_steps): # Eric: Is 3 substeps?
                this_h_j_vec = x_mat[sub_t, :, :].tolist()      # 17*3 list
                cost_total += calc_bone_length_loss(this_h_j_vec)
                cost_total += calc_collision_loss_box(this_h_j_vec)

            # print(cost_total)

            return cost_total, x_mat[:-1, :, :]

        # need to return minimized cost & hand poses
        # hand_poses can be drawn during next visualization

        hand_local_poss = (np.array(NOMINAL_POSE) + np.array([0.3, 0, 0])).tolist()
        x0 = [[0.3, 0, 0.0]] + hand_local_poss.copy()
        x0 = [x0] * (3*self.n_steps)
        x0 = np.array(x0).flatten()
        # print(x0)

        res = minimize(obj_func, x0, method='L-BFGS-B',
                       options={'disp': 3, 'maxiter': 40, 'maxfun': 1000000})

        return get_obj_value_and_hand_poses(res.x)

    def get_tar_fin_poss(self,
                         last_finger_infos: List[Tuple[List[float], bool]],
                         finger_infos: List[Tuple[List[float], bool]]
                         ) \
        -> Tuple[
            List[List[float]],
            List[float]
           ]:

        tar_fin_poss = []
        w_cands_init = []

        for fin_ind in range(self.num_fingertips):
            if finger_infos[fin_ind][1]:
                tar_fin_poss.append(finger_infos[fin_ind][0])

                # 1 index offset from NOMINAL_POSE & hand_poses
                w_init = np.array(finger_infos[fin_ind][0]) - NOMINAL_POSE[CHAINS[fin_ind][-1] - 1]
                w_cands_init.append(list(w_init))

            elif last_finger_infos[fin_ind][1]:
                tar_fin_poss.append(last_finger_infos[fin_ind][0])

                # 1 index offset from NOMINAL_POSE & hand_poses
                w_init = np.array(last_finger_infos[fin_ind][0]) - NOMINAL_POSE[CHAINS[fin_ind][-1] - 1]
                w_cands_init.append(list(w_init))
            else:
                tar_fin_poss.append(None)

        if np.array(w_cands_init).shape[0] == 0:
            w_local = [+0.25, 0.0, +0.03]
        else:
            # consolidate the wrist candidates by taking centroid
            w_local = list(np.array(w_cands_init).mean(axis=0))

        return tar_fin_poss, w_local

    @staticmethod
    def push_outside_box(p_local: List[float]):

        p_local = p_local.copy()

        margins = [-0.23, 0.23, -0.23, 0.23, -0.08, 0.08]
        dists = [p_local[0] + 0.2, 0.2 - p_local[0],
                 p_local[1] + 0.2, 0.2 - p_local[1],
                 p_local[2] + 0.05, 0.05 - p_local[2]]

        if (dists[0] <= 0 or dists[1] <= 0) or (dists[2] <= 0 or dists[3] <= 0) or (dists[4] <= 0 or dists[5] <= 0):
            return p_local

        ind = int(np.argmin(dists))

        if ind == 0 or ind == 1:
            p_local[0] = margins[ind]
        elif ind == 2 or ind == 3:
            p_local[1] = margins[ind]
        else:
            p_local[2] = margins[ind]

        return p_local

    def get_hand_pose_full(self,
                           tar_fin_poss: List[List[float]],
                           w_local_init:  List[float],
                           hand_init: List[List[float]] = None)\
            -> List[List[float]]:

        assert self.use_split_region

        hand_local_poss = (np.array(NOMINAL_POSE) + np.array(w_local_init)).tolist()

        if hand_init is None:
            all_poss = [w_local_init] + hand_local_poss.copy()
        else:
            all_poss = hand_init        # TODO: currently 1 specify hand_init just throw away w_local_init

        for i in range(20):
            # TODO: always run IK for 20 iters for now
            # FABRIK iteration

            w_cands = []

            # backward pass for each chain
            for fin_ind in range(self.num_fingertips):
                chain = CHAINS[fin_ind].copy()

                if tar_fin_poss[fin_ind]:
                    all_poss[chain[-1]] = tar_fin_poss[fin_ind].copy()
                # else, no target, i.e. current finger pose is the same as target, do not need to move

                for j in range(len(chain) - 2, -1, -1):
                    # j in n-2, ..., 0 (n == 5)
                    diff = np.array(all_poss[chain[j+1]]) - np.array(all_poss[chain[j]])
                    lam = NOMINAL_LENS[fin_ind * 4 + j] / np.linalg.norm(diff)

                    res = list(np.array(all_poss[chain[j+1]]) - lam * diff)
                    if j != 0:
                        all_poss[chain[j]] = res
                    else:
                        # should not overwrite chain[0] directly
                        w_cands.append(res)

            all_poss[0] = list(np.array(w_cands).mean(axis=0))

            # forward pass for each chain
            for fin_ind in range(self.num_fingertips):
                chain = CHAINS[fin_ind].copy()

                for j in range(0, len(chain) - 1):
                    # j in 0, ..., n-2 (n == 5)
                    diff = np.array(all_poss[chain[j+1]]) - np.array(all_poss[chain[j]])
                    lam = NOMINAL_LENS[fin_ind * 4 + j] / np.linalg.norm(diff)

                    res = list(np.array(all_poss[chain[j]]) + lam * diff)
                    all_poss[chain[j+1]] = res

        return all_poss

    def interp_hand_poss(self, all_hand_poss_0, all_hand_poss_1, percent):
        # The optimization output is some way points, hence the continuous hand pose is simply interpolated
        all_hand_poss_draw = np.array(all_hand_poss_0) * (1 - percent) + np.array(all_hand_poss_1) * percent
        all_hand_poss_draw = all_hand_poss_draw.tolist()

        return all_hand_poss_draw

    def step(self, a: List[float], a_next: List[float] = None, train=True):

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

        # TODO: during step, small block might slide away
        # Compansate for thickness of the object
        def get_small_block_location_local(this_face: int, this_pos_local: List[float]) -> List[float]:
            # size = [0.02, 0.02, 0.01] if i == 0 else [0.01, 0.01, 0.01]       # thumb larger
            this_pos_local = np.array(this_pos_local)
            if this_face == 0:
                return list(this_pos_local + np.array([0., 0., 0.01]))
            elif this_face == 1:
                return list(this_pos_local + np.array([0.01, 0., 0.]))
            else:
                assert this_face == 2
                return list(this_pos_local + np.array([0., 0., -0.01]))

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

                # Eric: suspected a bug
                if np.isclose(pos_vec[2], 0.06, 1e-5): # On the top
                    this_face = 0
                elif 0.05001 > pos_vec[2] > -0.05001: # On the side
                    this_face = 1
                else: # On the bottom
                    assert np.isclose(pos_vec[2], -0.06, 1e-6)
                    this_face = 2
            else: # Currently we always use split region
                if self.use_split_region:
                    # 4 fingers assumed
                    if fin_ind == 0:
                        # thumb, which have diffrent action mapping
                        loc_x = (sub_a[-3] + 1) * 0.5 * 0.15 + 0.05  # [-1, 1] ->[0, 2]-> [0, 0.15]
                        loc_z = 0.05        # always on top hard coded assumption
                        loc_y = sub_a[-2] * 0.2  # [-1, 1] -> [-0.1, 0.1]
                        this_face = 0
                    else: # Different finger have different position mapping, which may be inconsistent between ik
                        if sub_a[-3] < 0: # On the edge
                            loc_x = 0.2
                            loc_z = -sub_a[-3] / 10.0 - 0.05         # [-1, 0] -> [-0.05, 0.05] # Sensitive!
                            this_face = 1
                        else:
                            loc_x = (-sub_a[-3] + 1.0) * 0.15 + 0.05       # [0, 1] -> [0.2, 0]
                            loc_z = -0.05                            # On the other side of the object compared with thumb
                            this_face = 2

                        # Different finger have different y location mapping
                        if fin_ind == 1:
                            loc_y = sub_a[-2] / 30.0 - 2.0 / 30.0 # TODO: Why 30.0 since there are 3 fingers
                        elif fin_ind == 2:
                            loc_y = sub_a[-2] / 30.0 # TODO: Why 30.0 since there are 3 fingers
                        else:
                            assert fin_ind == 3
                            loc_y = sub_a[-2] / 30.0 + 2.0 / 30.0
                else:
                    if sub_a[-3] < -0.2:
                        loc_x = (sub_a[-3] + 1.0) / 4       # [-1, -0.2] -> [0, 0.2]
                        loc_z = 0.05
                        this_face = 0
                    elif sub_a[-3] < 0.2:
                        loc_x = 0.2
                        loc_z = -sub_a[-3] / 4.0            # [-0.2, 0.2] -> [0.05, -0.05]
                        this_face = 1
                    else:
                        loc_x = (-sub_a[-3] + 1.0) / 4       # [0.2, 1.0] -> [0.2, 0]
                        loc_z = -0.05
                        this_face = 2
                    loc_y = sub_a[-2] / 10.0                     # [-1, 1] -> [-0.1, 0.1]

                pos_vec = [loc_x, loc_y, loc_z]
                pos_vec = get_small_block_location_local(this_face, pos_vec.copy())

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
                if this_face == 0:
                    v = [v_t1, v_t2, -v_normal]
                elif this_face == 1:
                    v = [-v_normal, v_t1, v_t2]
                else:
                    assert this_face == 2
                    v = [-v_t1, -v_t2, v_normal]

                pos_force_vec += v

            pos_force_vec += pos_vec
            pos_force_vec += [1.0] if sub_a[-1] > 0 else [0.0]      # last bit on / off (non-colliding finger)
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

            r = - np.linalg.norm(vel_1) * 20 - len(cps_1) * 250 # As slow as possible, as less contact point as possible.
            r += - 300 * np.linalg.norm([pos_1[0], pos_1[1], pos_1[2] - 0.3]) # COM should be 0.3 meter off the ground. While x and y coordinate remain the same.
            r += 150 * rot_metric # hope the object can be rotate to a given z orientation
            return r

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

        a = np.tanh(a)      # [-1, 1]
        if self.opt_time:
            a_time = a[-1]      # [-1, 1]
            self.control_skip = int((a_time + 1.0) * 100.0 + 25.0)

        self.cur_fins, self.interped_fs = transform_a_no_opt_time(a, self.last_fins)

        pre_simulate(self.cur_fins)

        ave_r = 0.0
        wr_pos, fin_poss, cost_remaining = self.get_hand_pose(self.cur_fins, self.last_fins)

        if not train:
            object_poses = []
            tip_poses = []
        for t in range(self.control_skip):
            # apply vel target.....

            # if self.render and self.timer % 4 == 0:
            #     self._p.removeAllUserDebugItems()

            pos, quat = rb.get_link_com_xyz_orn(self._p, self.o_id, -1)
            
            if not train:
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
                if self.all_hand_poss is not None:
                    prev_key_frame = self.timer // 25
                    percent = (self.timer % 25) / 25
                    next_key_frame = prev_key_frame + 1
                    if next_key_frame == 3 * self.n_steps:
                        next_key_frame = 3 * self.n_step - 1

                    all_hand_poss_draw = self.interp_hand_poss(self.all_hand_poss[prev_key_frame],
                                                               self.all_hand_poss[next_key_frame],
                                                               percent)
                    self.draw_hand(all_hand_poss_draw)
                self.renderer.render()
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
        ave_r -= cost_remaining * 10000.0     # 0.01 * 10000 ~ 100
        ave_r += final_r / 300.0 * 2.0

        fin_pos = []
        for fin_ind in range(self.num_fingertips):
            if self.cur_fins[fin_ind][1]:
                fin_pos.append(self.cur_fins[fin_ind][0])
            else:
                fin_pos.append(None)
        self.all_active_fins.append(fin_pos)

        # only run once, used for later runs
        if self.render and self.all_hand_poss is None and self.c_step_timer == self.n_steps and self.solve_hand:
            _, self.all_hand_poss = self.get_hand_pose_full_with_optimization()

        self.last_fins = self.cur_fins.copy()
        
        if train==False:
            return obs, ave_r, False, {"finger_pos":tip_poses,"object_pose":object_poses}
        else:
            return obs, ave_r, False, {}

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

        return obs

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
