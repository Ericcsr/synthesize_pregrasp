import data.data_generation_config as dgc
import model.param as model_param

import numpy as np
import os

"""
Simple script to create smaller versions of the grasp configuration dataset for faster loading
"""
 
samples = 100
seed = 0
rs = np.random.RandomState(seed)

objects = model_param.drake_ycb_objects.keys()
for obj_name in objects:
     print('Processing object ', obj_name)
     grasp_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 dgc.kin_feasible_configs_path, obj_name+'.npy')
     data_original = np.load(grasp_data_path, allow_pickle=True)
     random_indices = rs.choice(np.arange(len(data_original)), size=samples)
     data_truncated = data_original[random_indices]
     grasp_data_truncated_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../data',
                 dgc.kin_feasible_configs_path, obj_name+f'_{samples}.npy')
     np.save(grasp_data_truncated_path, data_truncated, allow_pickle=True)