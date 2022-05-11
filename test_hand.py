import inspect
import math
import os
import time

import numpy as np
import pybullet as p
import pytorch_kinematics as pk
import torch
from pybullet_utils import bullet_client

PI = np.pi

def setStates(r_id, states):
    for i in range(len(states)):
        p.resetJointState(r_id, i, states[i])

p.connect(p.GUI)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
hand = p.loadURDF(os.path.join(currentdir, "model/resources/allegro_hand_description/urdf/allegro_hand_description_right.urdf"), useFixedBase=1)
print(p.getNumBodies(hand))
setStates(hand,[0] * 20)
orn = torch.tensor([0.0, 0, 0])
pos = torch.tensor([0, 0, 0.145])
q = pk.frame.tf.matrix_to_quaternion(pk.frame.tf.euler_angles_to_matrix(orn,"XYZ"))
print(p.getBasePositionAndOrientation(hand))
p.resetBasePositionAndOrientation(hand,pos.tolist(),[0,0,0,1])
# For this hand the finger tip's com position is identical to link position.
print(p.getLinkState(hand,0)[0])
print(p.getLinkState(hand,1)[0])
print(p.getLinkState(hand,2)[0])
# print(p.getLinkState(hand,14)[4])
# print(p.getLinkState(hand,19)[4])
for i in range (10000):
    p.stepSimulation()
    # Finger tip
    #p.resetJointState(hand, 4, targetValue=2*math.sin(0.01 * i))
    time.sleep(1./240.)
    # Need to place hand on the object's control point as close as possible.

