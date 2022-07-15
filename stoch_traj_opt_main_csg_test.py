from envs.small_block_contact_graph_env import LaptopBulletEnv
from envs.bookshelf_env import BookShelfBulletEnv

from stoch_traj_opt import StochTrajOptimizer
import numpy as np
import random
import imageio
from argparse import ArgumentParser

def parse_fin_data(fin_data):
    post_data = np.zeros((len(fin_data), 4, 4))
    for i in range(len(fin_data)):
        for j in range(4):
            post_data[i,j,:3] = fin_data[i][j][0]
            post_data[i,j,3] = fin_data[i][j][1]
    return post_data

envs_dict = {
    "bookshelf":BookShelfBulletEnv,
    "laptop":LaptopBulletEnv
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="u_opt_0-10_tp4")
    parser.add_argument("--env", type=str, default="laptop")
    parser.add_argument("--init_data", type=str, default="")
    parser.add_argument("--playback",action="store_true", default=False)
    args = parser.parse_args()

    uopt = np.load(f"data/traj/{args.exp_name}.npy")

    steps = uopt.shape[0]

    if args.init_data != "":
        last_fin = np.load(f"data/fin_data/{args.init_data}_fin_data.npy")[-1]
        init_obj_pose = np.load(f"data/object_poses/{args.init_data}_object_poses.npy")[-1]
    else:
        last_fin = None
        init_obj_pose = None

    # Define paths
    path = np.array([2,0,0])

    env = envs_dict[args.env]
    world = env(render=True, 
                num_fingertips=4, 
                num_interp_f=7, 
                last_fins=last_fin,
                init_obj_pose=init_obj_pose,
                train=False,
                steps = steps,
                path=path)

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
        for j in range(steps):
            state, c, done, info = world.step(uopt[j, :])
            finger_poses += info["finger_pos"]
            object_poses += info["object_pose"]
            images += info['images']
            last_fins.append(info["last_fins"])
            c = -c
            J += c
        if not args.playback:
            np.save(f"data/tip_data/{args.exp_name}_tip_poses.npy", finger_poses)
            np.save(f"data/object_poses/{args.exp_name}_object_poses.npy", object_poses)
            fin_data = parse_fin_data(last_fins)
            np.save(f"data/fin_data/{args.exp_name}_fin_data.npy", fin_data)
            print(len(images))
            print(images[-1])
            with imageio.get_writer(f"data/video/{args.exp_name}_test.gif",mode="I") as writer:
                for i in range(len(images)):
                    writer.append_data(images[i])
