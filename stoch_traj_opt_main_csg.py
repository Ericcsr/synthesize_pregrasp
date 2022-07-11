#from my_pybullet_envs.point_contact_env import PointContactBulletEnv
from envs.small_block_contact_graph_env import LaptopBulletEnv
from envs.bookshelf_env import BookShelfBulletEnv
#from my_pybullet_envs.shadow_hand_grasp_env import ShadowHandGraspEnv
from stoch_traj_opt import StochTrajOptimizer
import numpy as np
import traceback
from argparse import ArgumentParser

envs_dict = {
    "bookshelf":BookShelfBulletEnv,
    "laptop":LaptopBulletEnv
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="u_opt_0-10_tp4")
    parser.add_argument("--env", type=str, default="laptop")
    parser.add_argument("--render", action="store_true",default=False)
    parser.add_argument("--iters",type=int, default=100)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--init_data",type=str, default="")
    args = parser.parse_args()

    if args.init_data != "":
        fin_data = np.load(f"data/fin_data/{args.init_data}_fin_data.npy")[-1]
        init_obj_pose = np.load(f"data/object_poses/{args.init_data}_object_poses.npy")[-1]
    else:
        fin_data = None
        init_obj_pose = None

    sigma_list = [0.8] #[0.1, 0.2, 0.4, 0.8]

    # Create a reference environment
    ref_env = envs_dict[args.env](steps=args.steps, render=False, init_obj_pose=init_obj_pose)
    #paths, weight = ref_env.csg.getPathFromState(2, args.steps-1)
    paths = np.array([[2,0,0],[2,1,1]])
    weight = np.array([0.5, 0.5])

    #idx = np.argsort(weight)[:-11:-1]
    # weight = weight[idx]
    # paths = paths[idx]
    # print(paths)
    # print(weight)
    # exit()

    weight /= weight.sum()

    total_resources = 128
    resource_each_paths = total_resources * weight
    print(resource_each_paths)

    log_text_list = []
    f = open(f"data/log/{args.exp_name}.txt", 'w')
    try:
        for i, path in enumerate(paths):
            optimizer = StochTrajOptimizer(env=envs_dict[args.env], sigma=0.8, initial_guess=None,
                                    TimeSteps=args.steps, seed=12367134, render=False, Iterations=args.iters, num_fingertips=4, num_interp_f=7,
                                    Num_processes=int(resource_each_paths[i]), Traj_per_process=10, opt_time=False, verbose=1, 
                                    last_fins=fin_data, init_obj_pose=init_obj_pose, steps=args.steps, path=path)

            uopt = None
            Jopt_log = []
            r_log = []
            Jopt_total = 0
            Jopt = -np.inf
            for j in range(args.runs):
                _uopt, _Jopt, _Jopt_log, _r_log = optimizer.optimize()
                Jopt_total += _Jopt/args.runs
                if _Jopt > Jopt:
                    Jopt = _Jopt
                    Jopt_log = _Jopt_log
                    r_log = _r_log
                    uopt = _uopt
            # Save running rewards as well as optimal reward of each iterations
            np.save(f"data/rewards/optimal_{args.exp_name}_{i}.npy", Jopt_log)
            np.save(f"data/rewards/run_{args.exp_name}_{i}.npy", r_log)
            np.save(f'data/traj/{args.exp_name}_{i}.npy',uopt)
            log_text = f"Sigma: {i} mean rewards: {Jopt_total}, best rewards:{Jopt}\n"
            print(log_text)
            f.writelines(log_text)
            # replay optimal trajectory
            if args.render:
                optimizer.render_traj(uopt)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        f.close()


