#from my_pybullet_envs.point_contact_env import PointContactBulletEnv
from torch import Argument
from small_block_contact_env import SmallBlockContactBulletEnv
#from my_pybullet_envs.shadow_hand_grasp_env import ShadowHandGraspEnv
from stoch_traj_opt import StochTrajOptimizer
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="u_opt_0-10_tp4")
    parser.add_argument("--render", action="store_true",default=False)
    parser.add_argument("--iters",type=int, default=100)
    args = parser.parse_args()

    max_forces_list = [50] #[10,20,30,40,50,100]
    sigma_list = [0.8] #[0.1, 0.2, 0.4, 0.8]

    log_text_list = []
    f = open(f"data/log/{args.exp_name}.txt", 'w')
    try:
        for max_forces in max_forces_list:
            for sigma in sigma_list:
                optimizer = StochTrajOptimizer(env=SmallBlockContactBulletEnv, sigma=sigma, initial_guess=None,
                                        TimeSteps=5, seed=12367134, render=False, Iterations=args.iters, num_fingertips=4, num_interp_f=7,
                                        Num_processes=64, Traj_per_process=10, opt_time=False, max_forces=max_forces, verbose=1)

                uopt = None
                Jopt_log = []
                r_log = []
                Jopt_total = 0
                Jopt = -np.inf
                for i in range(3):
                    _uopt, _Jopt,_Jopt_log, _r_log = optimizer.optimize()
                    Jopt_total += _Jopt/3
                    if _Jopt > Jopt:
                        Jopt = _Jopt
                        Jopt_log = _Jopt_log
                        r_log = _r_log
                        uopt = _uopt
                # Save running rewards as well as optimal reward of each iterations
                np.save(f"data/rewards/optimal_{args.exp_name}_{max_forces}_{sigma}.npy", Jopt_log)
                np.save(f"data/rewards/run_{args.exp_name}_{max_forces}_{sigma}.npy", r_log)
                np.save(f'data/traj/{args.exp_name}_{max_forces}_{sigma}.npy',uopt)
                log_text = f"Max force: {max_forces} Sigma: {sigma} mean rewards: {Jopt_total}, best rewards:{Jopt}\n"
                print(log_text)
                f.writelines(log_text)
                # replay optimal trajectory
                if args.render:
                    optimizer.render_traj(uopt)
    except:
        print("An Exception occured")
    finally:
        f.close()


