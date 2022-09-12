from envs.small_block_contact_graph_env import LaptopBulletEnv
from envs.bookshelf_graph_env import BookShelfBulletEnv
from envs.wall_box_graph_env import WallBoxBulletEnv
from envs.table_box_graph_env import TableBoxBulletEnv

import open3d as o3d
import numpy as np
import random
import imageio
import json
from argparse import ArgumentParser

# TODO: Check whether there are some problem induced by changing the number of fingers
def parse_fin_data(fin_data):
    post_data = np.zeros((len(fin_data), 4, 4))
    for i in range(len(fin_data)):
        for j in range(4):
            post_data[i,j,:3] = fin_data[i][j][0]
            post_data[i,j,3] = fin_data[i][j][1]
    return post_data

envs_dict = {
    "bookshelf":BookShelfBulletEnv,
    "laptop":LaptopBulletEnv,
    "wallbox":WallBoxBulletEnv,
    "tablebox":TableBoxBulletEnv
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="u_opt_0-10_tp4")
    parser.add_argument("--env", type=str, default="laptop")
    parser.add_argument("--init_data", type=str, default="")
    parser.add_argument("--playback",action="store_true", default=False)
    parser.add_argument("--showImage", action="store_true", default=False)
    parser.add_argument("--camera_conf", type=str, default="camera_1")
    args = parser.parse_args()

    uopt = np.load(f"data/traj/{args.exp_name}.npy")
    # Directly read max force information from experiment name.
    max_force = float(args.exp_name.split("_")[-1])
    print("Using Max Force:", max_force)

    steps = uopt.shape[0]

    if args.init_data != "":
        last_fin = np.load(f"data/fin_data/{args.init_data}_fin_data.npy")[-1]
        init_obj_pose = np.load(f"data/object_poses/{args.init_data}_object_poses.npy")[-1]
    else:
        last_fin = None
        init_obj_pose = None
    # Load camera configuration
    with open(f"data/camera_params/{args.camera_conf}.json", 'r') as json_file:
        data=json_file.read()
    camera_config = json.loads(data)

    # Define paths
    path = np.array([5,5,0])

    env = envs_dict[args.env]
    world = env(render=True, 
                active_finger_tips=[0,1], 
                num_interp_f=7, 
                last_fins=last_fin,
                init_obj_pose=init_obj_pose,
                train=False,
                steps = steps,
                path=path,
                showImage=args.showImage,
                max_forces = max_force)
    world.renderer.read_config(camera_config)

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    world.seed(seed)

    for i in range(1):
        world.reset()
        J = 0
        finger_poses = []
        object_poses = []
        last_fins = []
        images = []
        pcd = None
        for j in range(steps):
            state, c, done, info = world.step(uopt[j, :])
            finger_poses += info["finger_pos"]
            object_poses += info["object_pose"]
            images += info['images']
            pcd = info["pcd"]
            last_fins.append(info["last_fins"])
            c = -c
            J += c
        if not args.playback:
            np.save(f"data/tip_data/{args.exp_name}_tip_poses.npy", finger_poses)
            np.save(f"data/object_poses/{args.exp_name}_object_poses.npy", object_poses)
            # fin_data = parse_fin_data(last_fins)
            # np.save(f"data/fin_data/{args.exp_name}_fin_data.npy", fin_data)
            if not (pcd is None):
                o3d.io.write_point_cloud(f"data/output_pcds/{args.exp_name}_pcd.ply", pcd)
            with imageio.get_writer(f"data/video/{args.exp_name}_test.gif",mode="I") as writer:
                for i in range(len(images)):
                    writer.append_data(images[i])
