#from my_pybullet_envs.point_contact_env import PointContactBulletEnv
from torch import Argument
from small_block_contact_env import SmallBlockContactBulletEnv
#from my_pybullet_envs.shadow_hand_grasp_env import ShadowHandGraspEnv
from stoch_traj_opt import StochTrajOptimizer
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':
    # init ctrl all 0
    # sess = StochTrajOptimizer(env=PointContactBulletEnv, sigma=0.5, initial_guess=None,
    #                           TimeSteps=4, seed=12353, render=False, Iterations=100, num_fingertips=4, num_interp_f=7)
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="u_opt_0-10_tp4")
    args = parser.parse_args()

    optimizer = StochTrajOptimizer(env=SmallBlockContactBulletEnv, sigma=0.4, initial_guess=None,
                              TimeSteps=5, seed=12367134, render=False, Iterations=500, num_fingertips=4, num_interp_f=7,
                              Num_processes=40, Traj_per_process=15, opt_time=False)

    # sess = StochTrajOptimizer(env=ShadowHandGraspEnv, sigma=0.6, initial_guess=None,
    #                           TimeSteps=30, seed=123573, render=False, Iterations=200,
    #                           Num_processes=12)

    uopt, Jopt = optimizer.optimize()
    print(uopt.shape)
    # save uopt?
    np.save(f'data/traj/{args.exp_name}.npy',uopt)

    # uopt = np.load('u_opt_optt_010_tp.npy')

    # replay optimal trajectory?
    optimizer.render_traj(uopt)