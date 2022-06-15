import numpy as np
from argparse import ArgumentParser
import model.param as model_param
import torch
import pytorch_kinematics as pk

# Relative to the object
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

parser = ArgumentParser()
parser.add_argument("--exp_name", type=str, default="")
args = parser.parse_args()

tip_pose_data = np.load(f"data/tip_data/{args.exp_name}.npy")
tip_pose_output = tip_pose_data.copy()
obj_pose_data = np.load(f"data/object_poses/{args.exp_name}.npy")

steps = tip_pose_data.shape[0]//model_param.CONTROL_SKIP

# This should be done before solve for key points
for i in range(1, steps):
    for j in range(4):
        if tip_pose_data[(i-1)*50,j].sum() < 100: # In contact previously
            rel_pose = getRelativePose(tip_pose_data[i*50-1], obj_pose_data[i*50-1,:3], obj_pose_data[i*50-1,3:])
            world_pose = getWorldPose(rel_pose, obj_pose_data[i*50, :3], obj_pose_data[i*50, 3:])
            tip_pose_output[i*50,j] = world_pose

np.save(f"data/tip_data/{args.exp_name}_modified.npy", tip_pose_output)
