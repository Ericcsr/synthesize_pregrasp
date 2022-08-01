import open3d as o3d # FIXME: this breaks the code if it's not at the top
import argparse
import multiprocessing as mp
import numpy as np
import random
import itertools
import open3d as o3d
import os
import cvxpy as cp
from tqdm import tqdm

MIN_NORMAL_FORCE = {
    3:1.0,
    4:1000.0}
MIN_DISTANCE_BETWEEN_FINGER = 0.03 # Should differ task from task
DEFAULT_FRICTION_COEFF = 20.0

def get_screw_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def is_psd(x):
    return (np.linalg.eigvals(x) >= 0).all()

def get_min_external_wrench_qp_matrices(p_WF, C_WF, mu=DEFAULT_FRICTION_COEFF, 
                                        psd_offset=1e-4, n_fingers=3):
    # For force closure
    C_123_T = np.vstack([C_WF[i,:,:].T for i in range(n_fingers)]) # 12x3
    Q1 = C_123_T @ (C_123_T.T)
    #assert(is_psd(Q1))
    # For torque closure
    p_WF_cross_C = np.vstack([(get_screw_symmetric(p_WF[i,:])@
                                                C_WF[i,:,:]).T for i in range(n_fingers)])
    Q2 = p_WF_cross_C @ (p_WF_cross_C.T)
    #assert(is_psd(Q2))
    Q = Q1 + Q2
    #assert(is_psd(Q))
    #print("Q:",np.linalg.eigvals(Q))
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
    if n_fingers in MIN_NORMAL_FORCE.keys():
        min_normal_force = MIN_NORMAL_FORCE[n_fingers]
    else:
        min_normal_force = 10
    h = np.hstack([np.zeros(4 * n_fingers),-np.ones(n_fingers)]) * min_normal_force
    A = np.zeros((1,3 * n_fingers))
    b = np.zeros(1)
    #Q += np.eye(Q.shape[0])*psd_offset
    return Q, P, G, h, A, b

def construct_and_solve_wrench_closure_qp(p_WF, C_WF, mu=DEFAULT_FRICTION_COEFF,num_contact_points=3):
    Q, P, G, h, A, b = get_min_external_wrench_qp_matrices(p_WF, C_WF, mu, n_fingers=num_contact_points)
    try:
        z_cp = cp.Variable(Q.shape[0])
        z_cp.value = np.random.random(size=Q.shape[0]) * 1000
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(z_cp, Q)),
                        [G @ z_cp <= h])
        prob.solve()
        z_hat = z_cp.value
        obj = prob.value
        #print("Value:",0.5 * z_hat.T @ Q @ z_hat)
        #print(z_hat)
    except cp.SolverError:
        return None, np.inf
    return z_hat, obj 

# Should be able to use outside of this file in the main pipeline
def check_dyn_feasible(contact_points, contact_normals, tol=1e-4):
    np.random.seed(os.getpid())
    random.seed(os.getpid())
    mask = contact_points[:,0] < 50
    normals = np.copy(np.hstack([contact_points[mask], contact_normals[mask]]))

    # Check whether there are two point that are too close
    n_contacts = len(normals)
    combs = itertools.combinations(range(n_contacts), 3)
    for comb in combs:
        comb = np.asarray(comb)
        normal = normals[comb]
        p_WF = normal[:,:3].copy()
        C_WF = np.zeros((n_contacts,3,3))
        for finger_i in range(len(normal)):
            C_WF[finger_i,:,0] = normal[finger_i, 3:].copy()/np.linalg.norm(normal[finger_i, 3:])
            n = C_WF[finger_i,:,0]
            t0 = np.zeros(3)
            t0[0] = n[1]
            t0[1] = -n[0]
            t1 = np.cross(n, t0)
            C_WF[finger_i,:,1] = t0.copy()
            C_WF[finger_i,:,2] = t1.copy()
    
        _, obj= construct_and_solve_wrench_closure_qp(
        p_WF.copy(), C_WF.copy(), mu= DEFAULT_FRICTION_COEFF, num_contact_points=3)
        #print("Objective:", obj)
        if obj<=tol:
            return normal
    return None


def check_dyn_feasible_parallel(contact_points, contact_normals, tol=1e-4, num_process=10):
    args_lists = [[contact_points, contact_normals, tol]] * num_process
    with mp.Pool(num_process) as proc:
        results = proc.starmap(check_dyn_feasible, args_lists)

    for result in results:
        if not (result is None):
            return result

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
    ans = []
    for a in tmp_ans:
        if a is not None:
            ans.append(a)
    return np.array(ans)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find dynamically feasible grasps by solving IK")
    parser.add_argument('--num_process',type=int, default=mp.cpu_count())
    parser.add_argument('--num_contacts', type=int, default=3)
    args = parser.parse_args()
    assert(args.num_contacts in MIN_NORMAL_FORCE)
    # Create pointcloud
    mesh_box = o3d.geometry.TriangleMesh.create_box(0.4, 0.4, 0.1)
    mesh_box.translate(np.array([-0.2, -0.2, -0.05]))
    mesh_box.compute_triangle_normals()
    mesh_box.compute_vertex_normals()
    pcd_box = mesh_box.sample_points_poisson_disk(256,use_triangle_normal=True)
    ans = find_dynamically_feasible_contacts(pcd_box, num_procs=args.num_process, n_contacts=args.num_contacts)
    print("There are: ",len(ans),"dyn_feasible points")

    