import data.data_generation_config as dgc

import numpy as np
import os

if __name__=='__main__':
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",
                                dgc.kin_feasible_configs_path)
    print("Reading files in "+path)
    all_names = {}
    all_ans = None
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            # Exclude poses
            name_split = name.split(".")
            if name_split[-2].endswith("poses") or name_split[-1] != "npy" or \
                "-" not in name_split[-2]:
                continue
            drake_obj_and_pose_idx = "_".join(name.split("_")[:-1])
            file_path = os.path.join(root, name)
            ans = np.load(file_path, allow_pickle=True)
            if len(ans)==0:
                continue
            if drake_obj_and_pose_idx not in all_names:
                all_names[drake_obj_and_pose_idx] = ans
            else:
                all_names[drake_obj_and_pose_idx] = np.vstack([all_names[drake_obj_and_pose_idx], ans])
    for name in list(all_names.keys()):
        print(f"Consolidated {all_names[name].shape[0]} grasps for "+ name)
        # Safeguard to prevent overwriting
        write_file_name = os.path.join(path, name)
        assert not os.path.exists(write_file_name)
        np.save(write_file_name, all_names[name], allow_pickle=True)