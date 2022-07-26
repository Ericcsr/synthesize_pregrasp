import open3d as o3d
import argparse
import multiprocessing as mp
import numpy as np
import itertools
import open3d as o3d
import os
import cvxpy as cp
from tqdm import tqdm

MIN_NORMAL_FORCE = {
    3:1500.0,
    4:1000.0}
MIN_DISTANCE_BETWEEN_FINGER = 0.03 # Should differ task from task
DEFAULT_FRICTION_COEFF = 0.4

def get_min_external_wrench_qp_matrices(p_WF, C_WF, mu=DEFAULT_FRICTION_COEFF, 
                                        psd_offset=1e-7, n_fingers=3):
    C_123_T = np.vstack([C_WF[i,:,:].T for i in range(n_fingers)]) # 12x3
    Q = C_123_T @ (C_123_T.T)
    p_WF_cross_C = np.hstack([np.cross(np.atleast_2d(p_WF[i,:]),
                                                C_WF[i,:,:].T).T for i in range(n_fingers)])
    Q += p_WF_cross_C.T @ (p_WF_cross_C)
    P = np.zeros((1,3 * n_fingers))
    Gf = np.zeros((4 * n_fingers,3 * n_fingers))
    Ga = np.zeros((n_fingers, 3 * n_fingers))
    for i in range(n_fingers):
        Gf[4*i:4*(i+1),3*i:3*(i+1)] = np.array([[mu, -1., 0.],
                                            [mu, 0., -1.],
                                            [mu, 1., 0.],
                                            [mu, 0., 1.]]) # 16x12
        Ga[i, 3*i] = 1.
    G = np.vstack([Gf, Ga])
    h = np.hstack([np.zeros(4 * n_fingers),-np.ones(n_fingers)]) * MIN_NORMAL_FORCE[n_fingers]
    A = np.zeros((1,3 * n_fingers))
    b = np.zeros(1)
    Q += np.eye(Q.shape[0])*psd_offset
    return Q, P, G, h, A, b

def construct_and_solve_wrench_closure_qp(p_WF, C_WF, mu=DEFAULT_FRICTION_COEFF,num_contact_points=3):
    Q, P, G, h, A, b = get_min_external_wrench_qp_matrices(p_WF, C_WF, mu, n_fingers=num_contact_points)
    try:
        z_cp = cp.Variable(Q.shape[0])
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(z_cp, Q)),
                        [G @ z_cp <= h])
        prob.solve()
        z_hat = z_cp.value
        obj = prob.value
    except cp.SolverError:
        return None, np.inf
    return z_hat, obj 

# Should be able to use outside of this file in the main pipeline
def check_dyn_feasible(contact_points, contact_normals, tol=1e-4):
    n_contacts = len(contact_points)
    normals = np.copy(np.hstack([contact_points, contact_normals]))

    # Check whether there are two point that are too close
    for test_i in list(itertools.combinations(range(n_contacts),2)):
        i1, i2 = test_i
        p1 = normals[i1, :3]
        p2 = normals[i2, :3]
        if np.linalg.norm(p1-p2)<MIN_DISTANCE_BETWEEN_FINGER:
            return None

    p_WF = np.hstack([contact_points]).copy()
    C_WF = np.zeros((n_contacts,3,3))

    for finger_i in range(n_contacts):
        C_WF[finger_i,:,0] = contact_normals[finger_i, :].copy()
        n = C_WF[finger_i,:,0]
        t0 = np.zeros(3)
        t0[0] = n[1]
        t0[1] = -n[0]
        t1 = np.cross(n, t0)
        C_WF[finger_i,:,1] = t0.copy()
        C_WF[finger_i,:,2] = t1.copy()
    try:
        _, obj= construct_and_solve_wrench_closure_qp(
        p_WF.copy(), C_WF.copy(), mu= DEFAULT_FRICTION_COEFF, num_contact_points=n_contacts)
    except Exception as e:
        print(e)
        return None
    if obj<=tol:
        return normals
    return None

def find_dynamically_feasible_contacts(point_cloud, tol=1e-4, num_procs=10, n_contacts=3):
    '''

    :param point_cloud:
    :return: n*3*6 np array of feasible points
    '''
    surface_points = np.asarray(point_cloud.points).copy()
    assert point_cloud.has_normals()
    # print(point_cloud)
    surface_normals = np.asarray(point_cloud.normals).copy()
    cloud_point_count = len(point_cloud.points)
    # Hack
    # https://stackoverflow.com/questions/52265120/python-multiprocessing-pool-attributeerror
    global check_if_feasible

    def check_if_feasible(indices):
        indices = np.asarray(indices)
        normals = np.copy(np.hstack([surface_points[indices, :],
                                          surface_normals[indices, :]]))
        # if any pair of contact points is too close, discard it
        for test_i in list(itertools.combinations(range(n_contacts),2)):
            i1, i2 = test_i
            p1 = normals[i1, :3]
            p2 = normals[i2, :3]
            if np.linalg.norm(p1-p2)<MIN_DISTANCE_BETWEEN_FINGER:
                return None

        # p_WF is the first of the 3 normals
        p_WF = np.hstack([surface_points[indices, :]]).copy() # TODO: adapt to 4 fingers, trivial
        C_WF = np.zeros((n_contacts,3,3)) # TODO: adapt to 4 fingers, trivial
        # C_WF is arranged as 3x(3x3) orthonormal basis Cᵢ = [ᵂn̂ᵢ,ᵂt̂ᵢ₀,ᵂt̂ᵢ₁]
        # Construct an aribitrary orthonormal basis
        for finger_i, idx in enumerate(indices):
            C_WF[finger_i,:,0] = surface_normals[idx, :].copy()
            n = C_WF[finger_i,:,0]
            t0 = np.zeros(3)
            t0[0] = n[1]
            t0[1] = -n[0]
            t1 = np.cross(n, t0)
            C_WF[finger_i,:,1] = t0.copy()
            C_WF[finger_i,:,2] = t1.copy()
        try:
            _, obj= construct_and_solve_wrench_closure_qp(
            p_WF.copy(), C_WF.copy(), mu= DEFAULT_FRICTION_COEFF, num_contact_points=n_contacts)
        except Exception as e:
            print(e)
            return None
        if obj<=tol:
            return normals
        return None
    print(f'Solving with {num_procs} threads')
    # with mp.Pool(num_procs) as p:
    #     tmp_ans = p.map(check_if_feasible,
    #                 [comb for comb in itertools.combinations(range(cloud_point_count), n_contacts)])
    pool = mp.Pool(processes=num_procs)
    tmp_ans = []
    arg_list = list(itertools.combinations(range(cloud_point_count), n_contacts))
    for result in tqdm(pool.imap(check_if_feasible,arg_list, n_contacts), total=len(arg_list)):
        tmp_ans.append(result)
    dyn_feasible = []
    dyn_infeasible = []
    for a in tmp_ans:
        if a is not None:
            dyn_feasible.append(a)
        else:
            dyn_infeasible.append(a)
    return np.array(dyn_feasible), np.array(dyn_infeasible)

if __name__ == "__main__":
    import neurals.data_generation_config as dgc
    from scipy.spatial.transform import Rotation as scipy_rot
    parser = argparse.ArgumentParser(description="Find dynamically feasible grasps by solving IK")
    parser.add_argument('--drake_obj_name', type=str,
                    default='003_cracker_box')
    parser.add_argument('--start_pose_idx', type=int, default=0)
    parser.add_argument('--end_pose_idx', type=float, default=np.inf)
    parser.add_argument('--num_process',type=int, default=mp.cpu_count())
    parser.add_argument('--num_contacts', type=int, default=3)
    args = parser.parse_args()
    assert(args.num_contacts in MIN_NORMAL_FORCE)
    feasible_output_path = '/'.join([os.path.dirname(os.path.abspath(__file__)),"..",
                            dgc.dyn_feasible_points_cropped_path])
    infeasible_output_path = '/'.join([os.path.dirname(os.path.abspath(__file__)),"..",
                            dgc.dyn_infeasible_points_cropped_path])

    pointcloud_path = '/'.join([os.path.dirname(os.path.abspath(__file__)),"..",
                            "/crop_point_cloud_simple"])
    filelists = os.listdir(f"{pointcloud_path}/{args.drake_obj_name}")
    print(f"File names are: {filelists}")
    pcds = []
    for file in filelists:
        pcd = o3d.io.read_point_cloud(f"{pointcloud_path}/{args.drake_obj_name}/{file}")
        pcds.append(pcd)
    
    # Point clouds are stored with mesh_sdf_xyz_rpy considered
    print('drake_object_name',args.drake_obj_name)
    # Load the point cloud
    # Create directory for output
    feasible_output_dir = os.path.join(feasible_output_path, args.drake_obj_name)
    infeasible_output_dir = os.path.join(infeasible_output_path, args.drake_obj_name)
    if not os.path.exists(feasible_output_dir):
        os.makedirs(feasible_output_dir)
    if not os.path.exists(infeasible_output_dir):
        os.makedirs(infeasible_output_dir)
    # For each rest pose, we iterate through all triples to find those that are dynamically feasible
    for idx, pcd in enumerate(pcds):
        if idx < args.start_pose_idx or idx >=args.end_pose_idx:
            continue
        feasible_output_path_name = os.path.join(feasible_output_dir, str(idx))
        infeasible_output_path_name = os.path.join(infeasible_output_dir, str(idx))
        # Fresh read of point cloud
        print('Storing to '+ feasible_output_path_name, "and", infeasible_output_path_name)
        # Transform the point cloud
        dyn_feasible, dyn_infeasible = find_dynamically_feasible_contacts(pcd, num_procs=args.num_process, n_contacts=args.num_contacts)
        np.save(feasible_output_path_name, dyn_feasible)
        np.save(infeasible_output_path_name, dyn_infeasible)
