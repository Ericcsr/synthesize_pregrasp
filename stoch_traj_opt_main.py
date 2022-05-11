#from my_pybullet_envs.point_contact_env import PointContactBulletEnv
from small_block_contact_env import SmallBlockContactBulletEnv
#from my_pybullet_envs.shadow_hand_grasp_env import ShadowHandGraspEnv
from stoch_traj_opt import StochTrajOptimizer
import numpy as np

if __name__ == '__main__':
    # init ctrl all 0
    # sess = StochTrajOptimizer(env=PointContactBulletEnv, sigma=0.5, initial_guess=None,
    #                           TimeSteps=4, seed=12353, render=False, Iterations=100, num_fingertips=4, num_interp_f=7)

    optimizer = StochTrajOptimizer(env=SmallBlockContactBulletEnv, sigma=0.4, initial_guess=None,
                              TimeSteps=5, seed=12367134, render=False, Iterations=200, num_fingertips=4, num_interp_f=7,
                              Num_processes=12, opt_time=False)

    # sess = StochTrajOptimizer(env=ShadowHandGraspEnv, sigma=0.6, initial_guess=None,
    #                           TimeSteps=30, seed=123573, render=False, Iterations=200,
    #                           Num_processes=12)

    uopt, Jopt = optimizer.optimize()
    print(uopt.shape)
    # save uopt?
    np.save('u_opt_0-10_tp4.npy',uopt)

    # uopt = np.load('u_opt_optt_010_tp.npy')

    # replay optimal trajectory?
    optimizer.render_traj(uopt)