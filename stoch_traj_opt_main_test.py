#from my_pybullet_envs.point_contact_env import PointContactBulletEnv
from small_block_contact_env import SmallBlockContactBulletEnv
#from my_pybullet_envs.shadow_hand_grasp_env import ShadowHandGraspEnv
#from a2c_ppo_acktr.algo import StochTrajOptimizer
from stoch_traj_opt import StochTrajOptimizer
import numpy as np
import random

if __name__ == '__main__':
    # init ctrl all 0
    # sess = StochTrajOptimizer(env=PointContactBulletEnv, sigma=0.5, initial_guess=None,
    #                           TimeSteps=4, seed=12353, render=False, Iterations=100, num_fingertips=4, num_interp_f=7)

    # sess = StochTrajOptimizer(env=SmallBlockContactBulletEnv, sigma=0.4, initial_guess=None,
    #                           TimeSteps=5, seed=12357, render=False, Iterations=200, num_fingertips=4, num_interp_f=7,
    #                           Num_processes=1)

    # sess = StochTrajOptimizer(env=ShadowHandGraspEnv, sigma=0.6, initial_guess=None,
    #                           TimeSteps=30, seed=123572, render=False, Iterations=200,
    #                           Num_processes=12)

    uopt = np.load('u_opt_0-10_tp4.npy')

    steps = uopt.shape[0]

    env = SmallBlockContactBulletEnv
    world = env(render=True, num_fingertips=4, num_interp_f=7)

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    world.seed(seed)

    for i in range(1):
        world.reset()
        J = 0
        finger_poses = []
        object_poses = []
        for j in range(steps):
            # if j < steps - 1:
            #     state, c, done, _ = world.step(uopt[j, :], uopt[j + 1, :])      # interpolate temporal hand pose
            # else:
            #     state, c, done, _ = world.step(uopt[j, :], None)
            state, c, done, pose = world.step(uopt[j, :], None, train=False)
            finger_poses.append(pose["finger_pos"])
            object_poses += pose["object_pose"]
            print(len(pose["object_pose"]))
            c = -c
            J += c
        np.save("tip_poses.npy", finger_poses)
        np.save("object_poses.npy", object_poses)
