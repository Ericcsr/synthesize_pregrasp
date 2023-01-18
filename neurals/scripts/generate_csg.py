import os
import copy
import inspect
import open3d as o3d
import numpy as np
import torch
from argparse import ArgumentParser
from utils.contact_state_graph import ContactStateGraph
from utils.small_block_region import SmallBlockRegionDummy
from utils.mesh_region import MeshRegion
from neurals.network import LargeScoreFunction
from envs.scales import SCALES
from envs.bounding_boxes import BOUNDING_BOXES
from utils.helper import convert_q_bullet_to_matrix

NUM_SAMPLES = 10
K = 2

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

parser = ArgumentParser()
parser.add_argument("--latent_dim", type=int, default = 20)
parser.add_argument("--has_distance_field", action="store_true", default=False)
parser.add_argument("--model_name", type=str, default="with_df")
parser.add_argument("--env", type=str, required=True)
args = parser.parse_args()

# Should not generate CSG until very end
if args.env in ["plate","pen","waterbottle"]:
    mesh = o3d.io.read_triangle_mesh(os.path.join(currentdir, f"../../envs/assets/{args.env}_cvx_simple.obj"))
else:
    scale = SCALES[args.env]
    mesh = o3d.geometry.TriangleMesh.create_box(0.4*scale[0],0.4*scale[1],0.1*scale[2])
    mesh.translate([-0.5*0.4*scale[0], -0.5*0.4*scale[0], -0.5*0.1*scale[2]])
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()
pcd = mesh.sample_points_poisson_disk(1024, use_triangle_normal=True)
contact_regions = MeshRegion(mesh)
cropped_pcd = o3d.io.read_point_cloud(os.path.join(currentdir, f"../../envs/assets/init_pcds/{args.env}.ply"))# Should be saved from environment and distance value 




# Create distance field based on environment
df = np.asarray(pcd.points)[:,2] + 0.1 # 10 cm above the table top
cropped_df = np.load(os.path.join(currentdir, f"../../envs/assets/init_dfs/{args.env}.npy"))
aabbs = BOUNDING_BOXES[args.env]
init_pose = np.load(os.path.join(currentdir, f"../../envs/assets/init_pose/{args.env}.npy"))
rot_mat = convert_q_bullet_to_matrix(init_pose[3:])
# remove occluded region, those region should not be initial states
valid_id = []
sample_points = []

def deocclude(aabbs,pcd):
    all_points = []
    for bb in aabbs:
        cropped_pcd = copy.deepcopy(pcd).crop(bb)
        all_points.append(np.asarray(cropped_pcd.points))
    all_points = np.vstack(all_points)
    deoccluded_pcd = o3d.geometry.PointCloud()
    deoccluded_pcd.points = o3d.utility.Vector3dVector(all_points)
    return deoccluded_pcd

for r_id in contact_regions.regions:
    points = contact_regions.sample_points(region_id=r_id, n_samples=NUM_SAMPLES)
    _pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _pcd.rotate(rot_mat,center=[0,0,0])
    _pcd.translate(init_pose[:3])
    _pcd = deocclude(aabbs, pcd)
    if len(_pcd.points) > NUM_SAMPLES/2:
        valid_id.append(r_id)
    sample_points.append(points)


# construct all nodes from remaining regions
# Node should be all ordered pairs doesn't include two same region
# should use should make 
all_nodes = []
all_node_ids = []
id = 0
for i in contact_regions.regions:
    for j in contact_regions.regions:
        if i !=j:
            all_nodes.append((i,j)) # i,j are regional id
            all_node_ids.append(id)
            id += 1

init_nodes = []
init_ids = []
for i,node in enumerate(all_nodes):
    if node[0] in valid_id and node[1] in valid_id:
        init_nodes.append(node)
        init_ids.append(all_node_ids[i])

# evaluate score of nodes
# CPU should be fine
score_function = LargeScoreFunction(num_fingers=2, latent_dim=args.latent_dim, has_distance_field=args.has_distance_field)
score_function.load_state_dict(torch.load(f"{currentdir}/../pretrained_score_function/only_score_model_{args.model_name}/2980.pth"))
score_function.eval()

points_th = torch.from_numpy(np.asarray(pcd.points)).view(1,-1,3).float()
cropped_points_th = torch.from_numpy(np.asarray(cropped_pcd.points)).view(1,-1,3).float()

df_th = torch.from_numpy(df).view(1,-1,1).float()
cropped_df_th = torch.from_numpy(cropped_df).view(1,-1,1).float()

# First lets only use simple 2 point contacts
def compute_score(node_id,mode="init"):
    node = all_nodes[node_id]
    i_sample_points = sample_points[node[0]]
    j_sample_points = sample_points[node[1]]
    score = 0
    for t in range(len(j_sample_points)):
        extra_cond = np.hstack([i_sample_points[t],j_sample_points[t]])
        extra_cond_th = torch.from_numpy(extra_cond).view(1,len(extra_cond)).float()
        if mode == "init":
            score += score_function.pred_score(cropped_points_th, extra_cond_th, cropped_df_th)[0]
        else:
            score += score_function.pred_score(points_th, extra_cond_th, df_th)[0]
    return float(score) / len(i_sample_points)

init_nodes_scores = []
for init_id in init_ids:
    score = compute_score(init_id, mode="init")
    init_nodes_scores.append(score)

all_nodes_scores = []
for node_id in all_node_ids:
    score = compute_score(node_id, mode="all")
    all_nodes_scores.append(score)
print("Nodal score has complete")

# Construct path and path wise scores
paths = []
path_scores= []

def find_path(cum_path, cum_score, k):
    if k == 0:
        paths.append(copy.deepcopy(cum_path))
        path_scores.append(cum_score)
    else:
        for node_id in all_node_ids:
            node = all_nodes[node_id]
            if node[0] == cum_path[-1][0] or node[1] == cum_path[-1][1]:
                branch_path = copy.deepcopy(cum_path)
                branch_path.append(node)
                find_path(branch_path, cum_score+all_nodes_scores[node_id],k-1)

for i in range(len(init_ids)): # Here ID is nodal id
    path_score = init_nodes_scores[i]
    local_path = [init_nodes[i]]
    find_path(local_path, path_score, K-1)

# Rank all path scores
order = np.argsort(path_scores)

sorted_path = []
for o in order:
    sorted_path.append(paths[o])

print("Top paths")
csg_nodes = set()
for p in sorted_path[:10]:
    for node in p:
        csg_nodes.add(node)
csg_nodes = list(csg_nodes)
csg_nodes_np = np.asarray(list(csg_nodes))
csg_paths_ids = []
for p in sorted_path[:10]:
    csg_path_id = []
    for node in p:
        csg_path_id.append(csg_nodes.index(node))
    csg_paths_ids.append(csg_path_id)

csg_paths_ids = np.asarray(csg_paths_ids)
print(csg_nodes_np)
print(csg_paths_ids)
np.save(os.path.join(currentdir, f"../../data/contact_states/{args.env}_env/csg.npy"), csg_nodes_np)
np.save(os.path.join(currentdir, f"../../data/contact_states/{args.env}_env/paths_id.npy"), csg_paths_ids)
print("csg saved")
