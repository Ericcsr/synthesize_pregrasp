import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--drake_obj_name",type=str, default="003_cracker_box")
args = parser.parse_args()

object_name = args.drake_obj_name
filelists = os.listdir("../kin_feasible/finger_tips_pose/")

for file in filelists:
    if file.startswith(f"{object_name}_pose_"):
        file_idx = int(file[-5])
        dyn_feasible_grasps = np.load(f"../dyn_feasible/{object_name}_textured_256/{file_idx}.npy")[:,:,:3]
        kin_feasible_grasps = np.load(f"../kin_feasible/finger_tips_pose/{object_name}_pose_{file_idx}.npy")
        print(f"Total: dyn_feasible: {len(dyn_feasible_grasps)} kin_feasible: {len(kin_feasible_grasps)}")
        # Find common regions, id are reflected in dyn_feasible
        common_id = []
        for i in range(len(kin_feasible_grasps)):
            idx, = np.where((dyn_feasible_grasps == kin_feasible_grasps[i]).all(axis=(1,2)))
            common_id.append(idx[0])
        common_id = np.array(common_id)
        print("Length of Common ID:", len(common_id))
        common_mask = np.zeros(len(dyn_feasible_grasps)).astype(np.bool_)
        common_mask[common_id] = True
        uncommon_mask = ~common_mask
        # Kinematically infeasible
        kin_infeasible_grasps = dyn_feasible_grasps[uncommon_mask,:,:3]
        sample_idx = np.random.choice(len(kin_infeasible_grasps),len(kin_feasible_grasps))
        kin_infeasible_grasps = kin_infeasible_grasps[sample_idx]
        print(kin_infeasible_grasps.shape, kin_feasible_grasps.shape)
        np.save(f"../kin_infeasible/{object_name}_pose_{file_idx}.npy", kin_infeasible_grasps)