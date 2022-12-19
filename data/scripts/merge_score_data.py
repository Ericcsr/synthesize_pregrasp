import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dir", type=str, required=True, default=None)

args = parser.parse_args()

prefix = f"../score_function_data/{args.dir}"
dl = os.listdir(prefix)
print(dl)
scores = []
conditions = []
point_cloud_labels = []
point_clouds = []

for f in dl:
    d = np.load(f"{prefix}/{f}")
    scores.append(d["scores"])
    conditions.append(d["conditions"])
    point_cloud_labels.append(d["point_cloud_labels"])
    point_clouds.append(d["point_clouds"])

scores = np.hstack(scores)
point_cloud_labels = np.hstack(point_cloud_labels)
conditions = np.vstack(conditions)
point_clouds = np.vstack(point_clouds)

np.savez(f"../score_function_data/{args.dir}_score_data.npz", scores=scores,point_cloud_labels=point_cloud_labels,conditions=conditions, point_clouds=point_clouds)
