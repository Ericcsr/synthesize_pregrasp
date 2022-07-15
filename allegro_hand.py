import torch
import functools
import numpy as np
import open3d as o3d
import utils.rigidBodySento as rb
from multiprocessing import Pool
import pytorch_kinematics as pk
import pybullet as p
from model.rigid_body_model import AllegroHandPlantDrake
from model.param import POINT_NUM, SCALE, AllAllegroHandFingers, AllegroHandFinger, NameToFingerIndex, NameToArmFingerIndex
import model.param as model_param
import copy
import helper
from pydrake.math import RigidTransform
from pydrake.common import eigen_geometry
from pydrake.solvers.snopt import SnoptSolver
from pydrake.solvers.osqp import OsqpSolver
from pydrake.multibody.tree import FrameIndex

PI = np.pi
EULER_UPPER_BOUND = torch.tensor([2*PI, 2*PI, 2*PI])
EULER_LOWER_BOUND = torch.tensor([-2*PI, -2*PI, -2*PI])
INIT_BASE_POS = [0.0, 0.0, 0.0475]
DRAKE_OFFSET = [0.04444854, -0.02782113, 0.13250625]
INIT_BASE_ORN_EULER = [0, 0, -PI]


class AllegroHand:
    num_fingers = 4
    def __init__(self, init_pose=np.zeros(16), iters = 20, lr=[0.01,0.01],tuneBase=False, init_base=np.zeros(6)):
        self.tuneBase = tuneBase
        self.hand_chain = pk.build_chain_from_urdf(open("allegro_hand_description/urdf/allegro_hand_description_right.urdf").read())
        self.iters = iters
        self.finger_tips = {"ifinger":"link_3.0_tip",
                            "mfinger":"link_7.0_tip",
                            "rfinger":"link_11.0_tip",
                            "thumb":"link_15.0_tip"}
        self.lr = lr
        self.weights = {"ifinger":1.0,"mfinger":1.0,"rfinger":1.0, "thumb":1.0}
        # Should be Pos + Euler angle
        self.init_pose = torch.from_numpy(init_pose).float()
        if self.tuneBase:
            self.init_base = torch.from_numpy(init_base).float()
        self.reset()

    def reset(self):  
        self.tunable_pose = self.init_pose.clone().requires_grad_(True)
        if self.tuneBase:
            self.tunable_base = self.init_base.clone().requires_grad_(True)
        params = [{"params": self.tunable_pose,"lr":self.lr[0]}]
        if self.tuneBase:
            params.append({"params":self.tunable_base,"lr":self.lr[1]})
        self.optimizer = torch.optim.Adam(params)

    def regressFingertipPos(self, finger_tip_target, weights=None):
        """
        Finger tip target should be a dict
        """
        if weights != None:
            self.weights = weights
        optimal_state = self.tunable_pose.detach().numpy()
        if self.tuneBase:
            optimal_base = self.tunable_base.detach().numpy()
        best_loss = 1000000
        for i in range(self.iters):
            self.optimizer.zero_grad()
            result = self.hand_chain.forward_kinematics(self.tunable_pose)
            loss=0
            for key in self.finger_tips.keys():
                pos = result[self.finger_tips[key]].get_matrix()[:, :3, 3]
                if self.tuneBase:
                    pos = self.map_to_world(pos,self.tunable_base)
                loss += (pos - torch.from_numpy(finger_tip_target[key])).norm() * self.weights[key]
            loss.backward()
            if loss < best_loss:
                best_loss = float(loss)
                optimal_state = self.tunable_pose.detach().numpy()
                if self.tuneBase:
                    optimal_base = self.tunable_base.detach().numpy()
            self.optimizer.step()
            # Clamp with in bounded range
            if self.tuneBase:
                with torch.no_grad():
                    self.tunable_base[3:] = torch.max(
                        torch.min(self.tunable_base[3:],EULER_UPPER_BOUND),
                        EULER_LOWER_BOUND)
        if self.tuneBase:
            return optimal_state, optimal_base
        else:
            return optimal_state
    
    def map_to_world(self, local_pos, base_pose):
        base_euler = base_pose[3:]
        base_pos = base_pose[:3]
        R = pk.frame.tf.euler_angles_to_matrix(base_euler,convention="XYZ")
        pos_prime = pk.quaternion_apply(pk.matrix_to_quaternion(R.T), local_pos)
        return pos_prime + base_pos

    @staticmethod
    def getBulletJointState(joint_states):
        pjoint_states = np.zeros(20)
        for i in range(AllegroHand.num_fingers):
            pjoint_states[i*5:i*5+4] = joint_states[i*4:i*4+4]
        return pjoint_states

    @staticmethod
    def getBulletBaseState(base_state):
        pos = (base_state[:3]+np.array(INIT_BASE_POS)).tolist()
        orn = base_state[3:]
        orn -= np.array(INIT_BASE_ORN_EULER)
        orn = pk.frame.tf.matrix_to_quaternion(
            pk.frame.tf.euler_angles_to_matrix(torch.from_numpy(orn),"XYZ")).tolist()
        return pos, orn 

# Using Albert's code with same API
# This is not accompanied with a bullet agent
class AllegroHandDrake:
    def __init__(self,
                 object_collidable=True, 
                 useFixedBase=False, 
                 robot_path = model_param.allegro_hand_urdf_path,
                 baseOffset = model_param.allegro_hand_offset,
                 all_fingers = model_param.AllAllegroHandFingers,
                 base_pose = None, # Using euler angle vec 6
                 regularize=False):
        # Parameter for Drake plant
        self.object_collidable = object_collidable
        self.useFixedBase = useFixedBase
        self.robot_path = robot_path
        self.all_fingers = all_fingers
        self.base_pose = base_pose
        self.baseOffset = baseOffset

        self.obj_geo = {"x":0.2*SCALE,
                        "y":0.2*SCALE,
                        "z":0.05*SCALE}
        
        self.obj_id = rb.create_primitive_shape(p, 1.0, p.GEOM_BOX, 
                                      (0.2 * model_param.SCALE,0.2 * model_param.SCALE, 0.05 * model_param.SCALE),
                                      color = (0.6, 0, 0, 0.8), collidable=True)
        self.hand_id = p.loadURDF("model/resources/allegro_hand_description/urdf/allegro_arm.urdf", useFixedBase=1)
        self.setObjectPose(np.array([0,0,0.05 * SCALE]), np.array([0,0,0,1]))
        self.ik_attempts = 5
        self.solver = SnoptSolver()
        self.regularize=regularize
        self.first_solve = True
        self.joint_idx = list(range(p.getNumJoints(self.hand_id)))
        self.joint_init_pose = np.zeros(len(self.joint_idx))

    def setObjectPose(self, pos, orn):
        self.obj_pose = pos
        self.obj_orn = np.array([orn[3], orn[0], orn[1], orn[2]])#np.asarray(orn) 
        drake_orn = np.array([orn[3], orn[0], orn[1], orn[2]]) # Convert to drake
        self.object_world_pose = RigidTransform(quaternion=eigen_geometry.Quaternion(drake_orn), 
                                        p=self.obj_pose)
        if isinstance(self.base_pose, np.ndarray):
            base_orn = pk.matrix_to_quaternion(pk.euler_angles_to_matrix(torch.from_numpy(self.base_pose[3:]), "XYZ")).numpy()
            pybullet_base_orn = helper.convert_quat_for_bullet(base_orn)
            base_world_pose = RigidTransform(quaternion=eigen_geometry.Quaternion(base_orn), p=self.base_pose[:3])
            p.resetBasePositionAndOrientation(self.hand_id, self.base_pose[:3], pybullet_base_orn)
        else:
            base_world_pose = None
        self.hand_plant = AllegroHandPlantDrake(object_world_pose=self.object_world_pose, 
                                                meshcat_open_brower=False, 
                                                num_finger_tips=4,
                                                object_collidable=self.object_collidable,
                                                robot_path=self.robot_path,
                                                useFixedBase=self.useFixedBase,
                                                base_pose=base_world_pose,
                                                baseOffset=self.baseOffset,
                                                all_fingers=self.all_fingers)
        
        self.all_normals = self.createPointCloudNormals()
        return self.all_normals[:,:3]

    def get_new_plant_with_obj_pose(self, pos, orn):
        drake_orn = np.array([orn[3], orn[0], orn[1], orn[2]]) # Convert to drake
        obj_pose =  RigidTransform(quaternion=eigen_geometry.Quaternion(drake_orn), 
                                        p=pos)
        new_plant = AllegroHandPlantDrake(object_world_pose=self.object_world_pose, 
                                                meshcat_open_brower=False, 
                                                num_finger_tips=4,
                                                object_collidable=self.object_collidable,
                                                robot_path=self.robot_path,
                                                useFixedBase=self.useFixedBase,
                                                baseOffset=self.baseOffset,
                                                all_fingers=self.all_fingers)
        _, new_plant_context = new_plant.create_context()
        return new_plant.plant, new_plant_context

    def getBulletJointState(self,q,unused_finger={}):
        return self.hand_plant.get_bullet_hand_config_from_drake_q(q,unused_finger)[2]

    def getBulletBaseState(self, q):
        base_pos, base_orn,_ = self.hand_plant.get_bullet_hand_config_from_drake_q(q)
        return base_pos, base_orn

    def createPointCloudNormals(self):
        # TODO: (eric) the coordinate convention maybe different between drake, pybullet and open3d
        mesh_box = o3d.geometry.TriangleMesh.create_box(
            self.obj_geo['x']*2,
            self.obj_geo['y']*2,
            self.obj_geo['z']*2)
        mesh_box.compute_triangle_normals()
        mesh_box.compute_vertex_normals()
        point_box = mesh_box.sample_points_uniformly(number_of_points=POINT_NUM)
        offset = np.array([0.1, 0.1, 0.025])
        point_box.translate(-offset)
        matrix = pk.quaternion_to_matrix(torch.from_numpy(self.obj_orn))
        point_box.rotate(matrix.numpy(), center=(0,0,0))
        point_box.translate(self.obj_pose)
        self.kd_tree = o3d.geometry.KDTreeFlann(point_box)
        self.desired_positions = None
        pos = np.asarray(point_box.points)
        normals = np.asarray(point_box.normals)
        all_normals = np.hstack([pos, normals])
        return all_normals

    # Need to test the function of this
    def getClosestNormal(self,target_pose):
        indices = list(self.kd_tree.search_knn_vector_3d(target_pose,3)[1])
        normal = self.all_normals[indices].mean(0)
        normal[:3] = target_pose
        return normal 

    def augment_finger(self, finger_tip_target,weights):
        augment_target = {}
        for key in finger_tip_target.keys():
            if weights[key]==1:
                augment_target[key] = self.getClosestNormal(finger_tip_target[key])
        return augment_target

    # Each worker can search a subspace of solution space not only one solution
    def regressFingerTipPosWithRandomSearch(self, finger_tip_target, weights, has_normal=True, n_process=40, prev_q=None, interp_mode=False):
        # Based on geometry of the compute the norm at contact point
        contact_points = {}
        finger_tip_target = self.augment_finger(finger_tip_target, weights)
        name_to_idx = NameToArmFingerIndex if self.all_fingers == model_param.AllAllegroArmFingers else NameToFingerIndex
        for key in finger_tip_target.keys():
            contact_points[name_to_idx[key]] = finger_tip_target[key]
        #print(contact_points)
        # Number of tip target maybe less than 4
        collision_distance = 1e-4 #1e-3
        
        # self.desired_positions = desired_positions
        if isinstance(prev_q, np.ndarray):
            q_init = prev_q
        else:
            q_init = np.random.random(self.hand_plant.getNumDofs()) - 0.5
            print("Use Random Initial guess")

        # Load parameters for different sub processes
        args_list = []
        for i in range(n_process):
            _q_init = q_init + (np.random.random(q_init.shape)-0.5)/0.5*0.2
            _robot_path = self.robot_path
            _object_world_pose = self.object_world_pose
            _baseOffset = self.baseOffset
            _all_fingers = self.all_fingers
            _collision_distance = collision_distance
            _finger_tip_target = finger_tip_target
            _ik_attempts = self.ik_attempts
            _contact_points = contact_points
            _has_normals = has_normal
            # Regularize the solution
            if isinstance(prev_q, np.ndarray) and self.regularize:
                _prev_q = prev_q.copy()
                _interp_mode = interp_mode
                args_list.append((_q_init, _robot_path, _object_world_pose, _baseOffset,_all_fingers,
                              _collision_distance, _finger_tip_target,_ik_attempts, _contact_points, _has_normals, _prev_q, _interp_mode))
            # Not regularize solution
            else:
                args_list.append((_q_init, _robot_path, _object_world_pose, _baseOffset,_all_fingers,
                              _collision_distance, _finger_tip_target,_ik_attempts, _contact_points, _has_normals))
        with Pool(n_process) as proc:
            results = proc.starmap(_parallel_solve, args_list)
        
        # Select best solution
        best_q = None
        best_norm = 1000000
        for result in results:
            if not (result[1] and result[2]):
                continue
            elif np.linalg.norm(result[0]-q_init) < best_norm:
                best_q = result[0]
                best_norm = np.linalg.norm(result[0]-q_init)
        
        desired_positions = results[0][3]
        if not isinstance(best_q, np.ndarray):
            q_sol = result[0]
            flag = False
        else:
            q_sol = best_q
            flag = True
        # return q_sol, no_collision , match_finger_tips
        unused_fingers = set(self.all_fingers)
        for finger in contact_points.keys():
            unused_fingers.remove(finger)
        # Handle the delta angle of unused fingers
        # for finger in unused_fingers:

        # May be even no need to specity unused fingers
        joint_state = self.getBulletJointState(q_sol)
        base_state = self.getBulletBaseState(q_sol)
        self.first_solve = False
        return q_sol, joint_state, base_state, desired_positions, flag

    # Assume object pose in pybullet format
    def map_to_new_object_frame(self,current_obj_pose, next_obj_pose, finger_tip_target):
        r = getRelativePose(finger_tip_target[:3],current_obj_pose[:3],current_obj_pose[3:])
        new_pos = getWorldPose(r, next_obj_pose[:3], next_obj_pose[3:])
        rel_normal = getRelativeNormal(finger_tip_target[3:], current_obj_pose[3:])
        new_normal = getWorldNormal(rel_normal, next_obj_pose[3:])
        print(new_pos)
        return np.hstack([new_pos, new_normal])
        
    def createTipsVisual(self,radius=[0.04, 0.02, 0.02, 0.02]):
        """
        Finger order:
        Thumb, index finger, middle finger, ring finger
        """
        self.tip_id = []
        colors = [(0.9, 0.9, 0.9, 0.7),
                  (0.9, 0.0, 0.0, 0.7),
                  (0.0, 0.9, 0.0, 0.7),
                  (0.0, 0.0, 0.9, 0.7)]
        for i in range(4):
            id = rb.create_primitive_shape(p, 0.1, p.GEOM_SPHERE,
                                           (radius[i],),
                                           color = colors[i],
                                           collidable=False,
                                           init_xyz = (100,100,100))
            self.tip_id.append(id)

    def draw_desired_positions(self, desired_positions ,weights):
        """
        Should be called after creation of visual tips as well as solve ik
        """
        default_orn = [0,0,0,1]
        obsolete_pos = [100,100,100]
        order = copy.deepcopy(self.all_fingers)
        name_to_idx = NameToArmFingerIndex if self.all_fingers == model_param.AllAllegroArmFingers else NameToFingerIndex
        avaliable_tip_index = [0,1,2,3]
        unavaliable_tip_index = []
        unused_order = []
        for i,key in enumerate(weights.keys()):
            if weights[key]==0:
                order.remove(name_to_idx[key])
                unused_order.append(name_to_idx[key])
                avaliable_tip_index.remove(i)
                unavaliable_tip_index.append(i)
        
        for i, ord in zip(avaliable_tip_index, order):
            p.resetBasePositionAndOrientation(self.tip_id[i], desired_positions[ord], default_orn)
        
        for i in unavaliable_tip_index:
            p.resetBasePositionAndOrientation(self.tip_id[i], obsolete_pos, default_orn)

    def reset_finger_tips(self):
        obsolete_pos = [100,100,100]
        default_orn = [0,0,0,1]
        for i in self.tip_id:
            p.resetBasePositionAndOrientation(i, obsolete_pos, default_orn)

    def draw_point_cloud(self, pc):
        p.addUserDebugPoints(np.random.random(size=pc.shape), np.ones((len(pc), 3)))

    def setAction(self, action):
        p.setJointMotorControlArray(self.hand_id, 
                                    self.joint_idx, 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=action)

    def reset(self):
        for i in range(len(self.joint_idx)):
            p.resetJointState(self.hand_id, self.joint_idx[i], self.joint_init_pose[i])

def target_norm_cost(q,q_target):
    return 10 * (q-q_target).dot(q-q_target)

def _parallel_solve(q_init, 
                    robot_path, 
                    obj_world_pose, 
                    baseOffset, 
                    all_fingers,
                    collision_distance, 
                    finger_tip_target, 
                    ik_attempts,
                    contact_points,
                    has_normal=True,
                    prev_q=None,
                    interp_mode=False,
                    object_collidable=True,
                    useFixedBase=True):
    hand_plant = AllegroHandPlantDrake(object_world_pose=obj_world_pose, 
                                                meshcat_open_brower=False, 
                                                num_finger_tips=4,
                                                object_collidable=object_collidable,
                                                robot_path=robot_path,
                                                useFixedBase=useFixedBase,
                                                baseOffset=baseOffset,
                                                all_fingers=all_fingers,
                                                interp_mode=interp_mode)
    ik, constrains_on_finger, collision_constr, desired_positions = hand_plant.construct_ik_given_fingertip_normals(
            hand_plant.create_context()[1],
            contact_points,
            padding=model_param.object_padding,
            collision_distance= collision_distance, # 1e-3
            allowed_deviation=np.ones(3) * 0.00001,
            has_normals=has_normal)
    name_to_idx = NameToArmFingerIndex if all_fingers == model_param.AllAllegroArmFingers else NameToFingerIndex
    solver = SnoptSolver()
    no_collision = True
    match_finger_tips = True
    q_sol = None
    # Add new constraint here, need to make sure all lb are negative and ub are positive, thumb is problematic
    if isinstance(prev_q, np.ndarray): # This constraint have problem but why?? Initialization
        prog = ik.get_mutable_prog()
        lb, ub = hand_plant.get_joint_limits()
        r = (ub - lb) * model_param.ALLOWANCE
        # Using box constraint
        if model_param.USE_SOFT_BOUNDING:
            cost_fn = functools.partial(target_norm_cost, q_target=prev_q)
            prog.AddCost(cost_fn, ik.q())
        else:
            prog.AddBoundingBoxConstraint(prev_q-r, prev_q+r, ik.q())
            
    # Over
    for _ in range(ik_attempts):
        result = solver.Solve(ik.prog(), q_init)
        q_sol = result.GetSolution()
        no_collision = True if collision_distance==None else collision_constr.evaluator().CheckSatisfied(q_sol, tol=3e-2)
        for finger in finger_tip_target.keys(): # TODO: Only on active finger
            #print(constrains_on_finger[NameToFingerIndex[finger]][0].evaluator().Eval(q_sol))
            if not constrains_on_finger[name_to_idx[finger]][0].evaluator().CheckSatisfied(q_sol,tol=1e-4):
                match_finger_tips = False
                break
        if no_collision and match_finger_tips:
            break
        elif no_collision and not match_finger_tips:
            q_init = q_sol + (np.random.random(q_sol.shape)-0.5)/0.5*0.03
        else:
            q_init = q_sol + (np.random.random(q_sol.shape)-0.5)/0.5*0.2
    return q_sol, no_collision, match_finger_tips, desired_positions

def getRelativePose(tip_pose, obj_pose, obj_orn):
    r = tip_pose - obj_pose
    orn = torch.tensor([obj_orn[3], obj_orn[0], obj_orn[1], obj_orn[2]])
    orn_inv = pk.quaternion_invert(orn)
    r = pk.quaternion_apply(orn_inv, torch.from_numpy(r)).numpy()
    return r

def getWorldPose(rel_pos, obj_pose, obj_orn):
    orn = torch.tensor([obj_orn[3], obj_orn[0], obj_orn[1], obj_orn[2]])
    r = pk.quaternion_apply(orn, torch.from_numpy(rel_pos)).numpy()
    return r + obj_pose

def getRelativeNormal(normal, obj_orn):
    orn = torch.tensor([obj_orn[3], obj_orn[0], obj_orn[1], obj_orn[2]])
    orn_inv = pk.quaternion_invert(orn)
    rel_normal = pk.quaternion_apply(orn_inv, torch.from_numpy(normal)).numpy()
    return rel_normal

def getWorldNormal(normal, obj_orn):
    orn = torch.tensor([obj_orn[3], obj_orn[0], obj_orn[1], obj_orn[2]])
    abs_normal = pk.quaternion_apply(orn, torch.from_numpy(normal)).numpy()
    return abs_normal

if __name__ == "__main__":
    def setStates(r_id, states):
        for i in range(len(states)):
            p.resetJointState(r_id, i, states[i])
    # target_finger_tip = {"ifinger":np.array([0.06753495335578918, 0.11384217441082001, 0.028212862089276314]),
    #                       "mfinger":np.array([0.06753495335578918, 0.06748118996620178, 0.03647235780954361]),
    #                       "rfinger":np.array([0.06753495335578918, 0.020606638863682747, 0.0399756096303463]),
    #                       "thumb":np.array([0.09108926355838776, 0.045409176498651505, -0.01953653059899807])}
    # target_finger_tip = {"ifinger":np.array([1.0770504474639893, 1.0506165027618408, 1.0969237089157104]),
    #                       "mfinger":np.array([1.0597103834152222, 1.0532760620117188, 1.0532222986221313]),
    #                       "rfinger":np.array([1.0387502908706665, 1.0583890676498413, 1.011460781097412]),
    #                       "thumb":np.array([1.020847201347351, 1.1046178340911865, 1.058937668800354])}
    target_finger_tip = {"thumb":np.array([-4.27850485e-02, -4.24860492e-02,  1.88064143e-01]) * SCALE,
                         "ifinger":np.array([-4.04909030e-02, -2.00706780e-01,  2.12514341e-01]) * SCALE,
                         "mfinger":np.array([1.16615419e+01,  1.37120544e+02,  1.05254150e+02]) * SCALE,
                         "rfinger":np.array([5.99451810e-02, -1.12616345e-01,  1.93208486e-01]) * SCALE}
    weights = {"ifinger":1.0, # Good
               "mfinger":0.0, # Good
               "rfinger":1.0, # Good
               "thumb":1.0}   # Good Using projection
    object_pos_pybullet = np.array([-0.00834451, -0.08897273,  0.19901382]) * SCALE
    object_orn_pybullet = [-0.30716777, -0.64500374, -0.30996876,  0.62732567]
    drake_hand = AllegroHandDrake(useFixedBase=True,
                                  robot_path=model_param.allegro_arm_urdf_path,
                                  baseOffset=model_param.allegro_arm_offset,
                                  all_fingers = model_param.AllAllegroArmFingers,
                                  object_collidable=True)
    drake_hand.setObjectPose(object_pos_pybullet, object_orn_pybullet)
    #drake_hand.frames_printer()
    #input()
    #print("hand_created")
    #joint_state, base_state,targets,success = drake_hand.regressFingerTipPos(target_finger_tip, weights, has_normal=True)
    joint_state, base_state, targets,success =  drake_hand.regressFingerTipPosWithRandomSearch(target_finger_tip, weights, has_normal=True, n_process=4)
    print("solution obtained:",success) # Solutions are in the format of numpy array
    #print(joint_state)
    #drake_hand.root_printer()
    #input()


    # ref_state = [PI/4,PI/4,PI/4,PI/4, # Index
    #              PI/4,PI/4,PI/4,PI/4, # Middle
    #              PI/4,PI/4,PI/4,PI/4, # Ring
    #              PI/4,PI/4,PI/4,PI/4] # Thumb
    # ref_base_pos = [1.0, 1.0, 1.0]
    # ref_base_orn = [0.7325, 0.4619, 0.1913, 0.4619]
    
    # hand = AllegroHand(np.zeros(16), iters = 500,tuneBase=True,init_base=np.zeros(6))
    # #hand = AllegroHand(np.zeros(16), iters = 100,tuneBase=False)
    # result, base_state = hand.regressFingertipPos(target_finger_tip)
    # #result = hand.regressFingertipPos(target_finger_tip)
    # print(ref_state-result)
    # pjoints_states = AllegroHand.getBulletJointState(result)
    # base_pos, base_orn = AllegroHand.getBulletBaseState(base_state)
    p.connect(p.GUI)
    hand_id = p.loadURDF("model/resources/allegro_hand_description/urdf/allegro_arm.urdf",useFixedBase=1)
    base_pos, base_orn = base_state
    setStates(hand_id, joint_state)
    p.resetBasePositionAndOrientation(hand_id, base_pos, base_orn)
    center = rb.create_primitive_shape(p,1.0,p.GEOM_SPHERE,(0.01,),collidable=False,init_xyz=object_pos_pybullet)
    # while True:
    #     setStates(hand_id, joint_state)
    #     p.resetBasePositionAndOrientation(hand_id, base_pos, base_orn)
    #     print(p.getLinkState(hand_id,4)[4])
    #     print(p.getLinkState(hand_id,9)[4])
    #     print(p.getLinkState(hand_id,14)[4])
    #     print(p.getLinkState(hand_id,19)[4])
    #     input("Press enter to close")
    #     # input("Press enter to close")
    #     setStates(hand_id, AllegroHand.getBulletJointState(ref_state))
    #     p.resetBasePositionAndOrientation(hand_id, ref_base_pos, ref_base_orn) # TODO: (Eric) base rotation have some problem
    #     print("Reference Pose:")
    #     print(p.getLinkState(hand_id,4)[4])
    #     print(p.getLinkState(hand_id,9)[4])
    #     print(p.getLinkState(hand_id,14)[4])
    #     print(p.getLinkState(hand_id,19)[4])
    #     input("Press enter to close")
    obj_id = rb.create_primitive_shape(p, 1.0, p.GEOM_BOX,(0.2*SCALE, 0.2*SCALE,0.05*SCALE),
                                       color=(0.6, 0, 0, 0.8),collidable=True,
                                       init_xyz=object_pos_pybullet,
                                       init_quat=object_orn_pybullet)
    drake_hand.createTipsVisual()
    drake_hand.draw_desired_positions(targets,weights)
    drake_hand.draw_point_cloud()
    input()


            
