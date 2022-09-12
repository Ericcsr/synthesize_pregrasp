#from my_pybullet_envs.point_contact_env import PointContactBulletEnv
from envs.small_block_contact_graph_env import LaptopBulletEnv
from envs.bookshelf_graph_env import BookShelfBulletEnv
from envs.wall_box_graph_env import WallBoxBulletEnv
from envs.table_box_graph_env import TableBoxBulletEnv
import model.param as model_param
from stoch_traj_opt import StochTrajOptimizer
import numpy as np
import traceback
from neurals.test_options import TestOptions
#from utils.path_filter import filter_paths

envs_dict = {
    "bookshelf":BookShelfBulletEnv,
    "laptop":LaptopBulletEnv,
    "tablebox":TableBoxBulletEnv,
    "wallbox":WallBoxBulletEnv
}

if __name__ == '__main__':
    original_parser = TestOptions()
    parser = original_parser.parser
    parser.add_argument("--exp_name", type=str, default="u_opt_0-10_tp4")
    parser.add_argument("--env", type=str, default="laptop")
    parser.add_argument("--render", action="store_true",default=False)
    parser.add_argument("--iters",type=int, default=20)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max_force", type=float, default=-1)
    parser.add_argument("--mode", type=str, default="only_score")
    parser.add_argument("--name_score", type=str, default=None, required=True)
    parser.add_argument("--name_epoch", type=str, default=None, required=True)
    parser.add_argument("--has_distance_field", action="store_true", default=False)
    args = original_parser.parse()

    
    fin_data = None
    init_obj_pose = None
    sigma_list = [2.0] #[0.1, 0.2, 0.4, 0.8]
    max_force = model_param.MAX_FORCE if args.max_force == -1 else args.max_force

    # Create a reference environment
    # ref_env = envs_dict[args.env](steps=args.steps, render=False, init_obj_pose=init_obj_pose)
    # paths_raw, distances = ref_env.csg.getPathFromState(2, args.steps)

    # paths = filter_paths(paths_raw, ref_env.csg, ref_env.contact_region)
    # Need to check final state dynamical feasibility here
    
    paths = np.array([[5,5,0]])
    weight = np.array([0.5])

    log_text_list = []
    f = open(f"data/log/{args.exp_name}.txt", 'w')
    try:
        opt_dict=args.__dict__
        opt_dict["force_skip_load"] = True
        for i, path in enumerate(paths): # Search for all the paths.
            optimizer = StochTrajOptimizer(env=envs_dict[args.env], sigma=0.8, initial_guess=None,
                                    TimeSteps=args.steps, seed=12367134, render=False, Iterations=args.iters, active_finger_tips=[0,1], num_interp_f=7,
                                    Num_processes=3, Traj_per_process=130, opt_time=False, verbose=1,mode=args.mode,
                                    last_fins=fin_data, init_obj_pose=init_obj_pose, steps=args.steps, path=path,
                                    sc_path=f"neurals/pretrained_score_function/{args.name_score}/{args.name_epoch}.pth",
                                    dex_path=f"checkpoints/{args.name}/latest_net.pth", opt_dict=opt_dict, 
                                    has_distance_field=args.has_distance_field, max_forces = max_force)

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
            np.save(f"data/rewards/optimal_{args.exp_name}_{max_force}.npy", Jopt_log)
            np.save(f"data/rewards/run_{args.exp_name}_{max_force}.npy", r_log)
            np.save(f'data/traj/{args.exp_name}_{max_force}.npy',uopt)
            log_text = f"Max force: {max_force} mean rewards: {Jopt_total}, best rewards:{Jopt}\n"
            print(log_text)
            f.writelines(log_text)
            # replay optimal trajectory
            if args.render:
                optimizer.render_traj(uopt)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        f.close()


