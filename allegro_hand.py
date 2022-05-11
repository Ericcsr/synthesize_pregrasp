import torch
import numpy as np
import open3d as o3d
import rigidBodySento as rb
import pytorch_kinematics as pk
import pybullet as p
from model.rigid_body_model import AllegroHandPlantDrake
from model.param import POINT_NUM, SCALE, AllAllegroHandFingers, AllegroHandFinger, NameToFingerIndex
import model.param as model_param
from pydrake.math import RigidTransform
from pydrake.common import eigen_geometry
from pydrake.solvers.snopt import SnoptSolver
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
class AllegroHandDrake:
    def __init__(self, object_collidable=True):
        self.object_collidable = object_collidable
        self.obj_geo = {"x":0.2*SCALE,
                        "y":0.2*SCALE,
                        "z":0.05*SCALE}
        self.setObjectPose(np.array([0,0,0.05 * SCALE]), np.array([0,0,0,1]))
        self.ik_attempts = 5
        self.solver = SnoptSolver()

    def setObjectPose(self, pos, orn):
        self.obj_pose = pos
        self.obj_orn = np.array([orn[3], orn[0], orn[1], orn[2]])#np.asarray(orn) 
        drake_orn = np.array([orn[3], orn[0], orn[1], orn[2]]) # Convert to drake
        obj_world_pose = RigidTransform(quaternion=eigen_geometry.Quaternion(drake_orn), 
                                        p=self.obj_pose)
        self.hand_plant = AllegroHandPlantDrake(object_world_pose=obj_world_pose, 
                                                meshcat_open_brower=True, 
                                                num_viz_spheres=4,
                                                object_collidable=self.object_collidable)
        self.all_normals = self.createPointCloudNormals()

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

    def frames_printer(self):
        N = self.hand_plant.plant.num_frames()
        for i in range(N):
            index = FrameIndex(i)
            print(self.hand_plant.plant.get_frame(index))

    def root_printer(self):
        q = np.zeros(51)
        q[0] = 1
        base_pos, base_orn, joint_state = self.hand_plant.convert_q_to_hand_configuration(q)
        print(base_pos)
        print(base_orn)
        output_port = self.hand_plant.plant.get_body_poses_output_port()
        state_output_port = self.hand_plant.plant.get_state_output_port()
        diagram_context, plant_context = self.hand_plant.create_context()
        vector = output_port.Eval(plant_context)
        state = state_output_port.Eval(plant_context)
        print(vector[2])
        print(vector[3])
        #print(state)
        self.hand_plant.plant.SetPositions(plant_context, q)
        self.hand_plant.diagram.Publish(diagram_context)
        
    def regressFingerTipPos(self, finger_tip_target, weights, has_normal=False):
        # Based on geometry of the compute the norm at contact point
        contact_points = {}
        finger_tip_target = self.augment_finger(finger_tip_target, weights)
        for key in finger_tip_target.keys():
            contact_points[NameToFingerIndex[key]] = finger_tip_target[key]
        #print(contact_points)
        # Number of tip target maybe less than 4
        collision_distance = 1e-4 #1e-3
        ik, constrains_on_finger, collision_constr, desired_positions = self.hand_plant.construct_ik_given_fingertip_normals(
            self.hand_plant.create_context()[1],
            contact_points,
            padding=model_param.object_padding,
            collision_distance= collision_distance, # 1e-3
            allowed_deviation=np.ones(3) * 0.00001,
            has_normals=has_normal)
        self.desired_positions = desired_positions
        q_init = np.random.random(len(ik.q())) - 0.5
        match_finger_tips = False
        no_collision = False
        for _ in range(self.ik_attempts):
            result = self.solver.Solve(ik.prog(), q_init)
            q_sol = result.GetSolution(ik.q())
            no_collision = True if collision_distance==None else collision_constr.evaluator().CheckSatisfied(q_sol, tol=3e-2)
            match_finger_tips = True
            for finger in finger_tip_target.keys(): # TODO: Only on active finger
                #print(constrains_on_finger[NameToFingerIndex[finger]][0].evaluator().Eval(q_sol))
                if not constrains_on_finger[NameToFingerIndex[finger]][0].evaluator().CheckSatisfied(q_sol,tol=1e-4):
                    match_finger_tips = False
                    break
            if no_collision and match_finger_tips:
                break
            elif no_collision and not match_finger_tips:
                q_init = q_sol + (np.random.random(q_sol.shape)-0.5)/0.5*0.03
            else:
                q_init = q_sol + (np.random.random(q_sol.shape)-0.5)/0.5*0.2
        
        # return q_sol, no_collision , match_finger_tips
        unused_fingers = set(AllAllegroHandFingers)
        for finger in contact_points.keys():
            unused_fingers.remove(finger)
        joint_state = self.getBulletJointState(q_sol, unused_fingers)
        base_state = self.getBulletBaseState(q_sol)
        return joint_state, base_state, match_finger_tips and no_collision

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

    def draw_desired_positions(self, weights):
        """
        Should be called after creation of visual tips as well as solve ik
        """
        order = [AllegroHandFinger.THUMB,
                 AllegroHandFinger.INDEX, 
                 AllegroHandFinger.MIDDLE, 
                 AllegroHandFinger.RING]
        for key in weights.keys():
            if weights[key]==0:
                order.remove(NameToFingerIndex[key])
        orn = [0,0,0,1]
        print(self.desired_positions)
        for i in range(len(order)):
            p.resetBasePositionAndOrientation(self.tip_id[i], self.desired_positions[order[i]], orn)

    def draw_point_cloud(self):
        p.addUserDebugPoints(self.all_normals[:,:3], np.ones((len(self.all_normals), 3)))



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
    drake_hand = AllegroHandDrake(object_collidable=True)
    drake_hand.setObjectPose(object_pos_pybullet, object_orn_pybullet)
    #print("hand_created")
    joint_state, base_state,success = drake_hand.regressFingerTipPos(target_finger_tip, weights, has_normal=True)
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
    hand_id = p.loadURDF("model/resources/allegro_hand_description/urdf/allegro_hand_description_right.urdf",useFixedBase=1)
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
    drake_hand.draw_desired_positions(weights)
    drake_hand.draw_point_cloud()
    input()


            
