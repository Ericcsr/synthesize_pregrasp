import stable_baselines3 as sb3
import torch
import numpy as np
from small_block_contact_env import SmallBlockContactBulletEnv
from argparse import ArgumentParser
import random


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="u_opt_0-10_tp4")
    parser.add_argument("--init_data", type=str, default="")
    parser.add_argument("--iters",type=int, default=100)
    parser.add_argument("--steps",type=int, default=3)
    args = parser.parse_args()

    if args.init_data != "":
        last_fin = np.load(f"data/fin_data/{args.init_data}_fin_data.npy")[-1]
        init_obj_pose = np.load(f"data/object_poses/{args.init_data}_object_poses.npy")[-1]
    else:
        last_fin = None
        init_obj_pose = None

    
    env = SmallBlockContactBulletEnv(render=False, 
                                     num_fingertips=4, 
                                     num_interp_f=7,
                                     last_fins=last_fin,
                                     init_obj_pose=init_obj_pose,
                                     train=True,
                                     opt_time=False,
                                     steps=args.steps)

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    model = sb3.TD3("MlpPolicy", 
                    env, 
                    verbose=1, 
                    device=torch.device("cpu"),
                    learning_starts=1000,
                    target_policy_noise=0.5)
    model.learn(total_timesteps=3 * args.iters)

    # Since we are optimizing a trajectory instead of policy we may need to replay the policy and record the actions sequence.

    obs = env.reset()
    actions = np.zeros((args.steps, 96))
    for i in range(args.steps):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        actions[i,:] = action

    np.save(f"data/traj/{args.exp_name}.npy", actions)