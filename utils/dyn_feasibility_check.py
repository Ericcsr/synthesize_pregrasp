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
import pytorch_kinematics as pk
import torch
from itertools import combinations
MIN_NORMAL_FORCE = {
    3:1.0,
    4:1000.0}
MIN_DISTANCE_BETWEEN_FINGER = 0.03 # Should differ task from task
DEFAULT_FRICTION_COEFF = 2

def get_screw_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def is_psd(x):
    return (np.linalg.eigvals(x) >= 0).all()

def get_min_external_wrench_qp_matrices(p_WF, C_WF, mu=DEFAULT_FRICTION_COEFF, 
                                        psd_offset=1e-8, n_fingers=3):
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
def check_dyn_feasible(contact_points, contact_normals, tol=2e-3):
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
        try:
            _, obj= construct_and_solve_wrench_closure_qp(
            p_WF.copy(), C_WF.copy(), mu= DEFAULT_FRICTION_COEFF, num_contact_points=3)
        except:
            return None
        if obj < 0:
            #print("Warning: QP is unbounded, result is inaccurate!!")
            return None
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

def project2norm(v, n):
    return v.dot(n)/(np.linalg.norm(n)**2)*n

def check_in_triangle(vertices, point):
    res0 = np.cross(vertices[1] - vertices[0], point-vertices[0])
    res1 = np.cross(vertices[2] - vertices[1], point-vertices[1])
    res2 = np.cross(vertices[0] - vertices[2], point-vertices[2])
    c1 = res0.dot(res1)
    c2 = res0.dot(res2)
    if (c1>0 and c2>0) or (c1<0 and c1<0):
        return True
    else:
        return False

def check_sign_change(vectors, vertices):
    center = (vertices[0] + vertices[1] + vertices[2])/3
    prev_c = None
    for i in range(len(vertices)):
        r = vertices[i] - center
        c1 = np.cross(r, vectors[2*i])
        c2 = np.cross(r, vectors[2*i+1])
        if prev_c is None:
            prev_c = c1
        if prev_c.dot(c1) < 0 or c1.dot(c2) < 0:
            return True
        else:
            prev_c = c2
    return False       

def simple_check_dyn_feasible_3p(contact_points, contact_normals):
    v1 = contact_points[1] - contact_points[0]
    v2 = contact_points[2] - contact_points[0]
    n = np.cross(v1,v2)
    if np.linalg.norm(n) < 1e-4: # 2cm2
        return False
    n /= np.linalg.norm(n) # Normalized normal vector
    # Project every norm on to plane, assume vectors in contact_normals are normal vector
    pj_n0 = project2norm(contact_normals[0], n)
    pj_n1 = project2norm(contact_normals[1], n)
    pj_n2 = project2norm(contact_normals[2], n)

    # Project on plane
    pj_p0 = contact_normals[0] - pj_n0
    pj_p1 = contact_normals[1] - pj_n1
    pj_p2 = contact_normals[2] - pj_n2

    if np.linalg.norm(pj_n0) / np.linalg.norm(pj_p0) > DEFAULT_FRICTION_COEFF:
        return False
    elif np.linalg.norm(pj_n1) / np.linalg.norm(pj_p1) > DEFAULT_FRICTION_COEFF:
        return False
    elif np.linalg.norm(pj_n2) / np.linalg.norm(pj_p2) > DEFAULT_FRICTION_COEFF:
        return False

    temp0 = np.linalg.norm(contact_normals[0]) / np.linalg.norm(pj_p0) * np.linalg.norm(pj_n0)
    temp1 = np.linalg.norm(contact_normals[1]) / np.linalg.norm(pj_p1) * np.linalg.norm(pj_n1)
    temp2 = np.linalg.norm(contact_normals[2]) / np.linalg.norm(pj_p2) * np.linalg.norm(pj_n2)

    mu = DEFAULT_FRICTION_COEFF
    angle0 = np.arctan(np.sqrt(mu**2-temp0**2) / np.sqrt(temp0**2+np.linalg.norm(contact_normals[0])**2))
    angle1 = np.arctan(np.sqrt(mu**2-temp1**2) / np.sqrt(temp1**2+np.linalg.norm(contact_normals[1])**2))
    angle2 = np.arctan(np.sqrt(mu**2-temp2**2) / np.sqrt(temp2**2+np.linalg.norm(contact_normals[2])**2))

    # Rotate via axis angle
    pj_p0 = pj_p0 / np.linalg.norm(pj_p0)
    pj_p1 = pj_p1 / np.linalg.norm(pj_p1)
    pj_p2 = pj_p2 / np.linalg.norm(pj_p2)

    cone00 = pk.quaternion_apply(pk.axis_angle_to_quaternion(torch.from_numpy(n*angle0)),torch.from_numpy(pj_p0)).numpy()
    cone01 = pk.quaternion_apply(pk.axis_angle_to_quaternion(torch.from_numpy(-n*angle0)),torch.from_numpy(pj_p0)).numpy()
    cone10 = pk.quaternion_apply(pk.axis_angle_to_quaternion(torch.from_numpy(n*angle1)),torch.from_numpy(pj_p1)).numpy()
    cone11 = pk.quaternion_apply(pk.axis_angle_to_quaternion(torch.from_numpy(-n*angle1)),torch.from_numpy(pj_p1)).numpy()
    cone20 = pk.quaternion_apply(pk.axis_angle_to_quaternion(torch.from_numpy(n*angle2)),torch.from_numpy(pj_p2)).numpy()
    cone21 = pk.quaternion_apply(pk.axis_angle_to_quaternion(torch.from_numpy(-n*angle2)),torch.from_numpy(pj_p2)).numpy()
    origin = np.zeros(3)
    vectors = [cone00,cone01,cone10,cone11,cone20,cone21]
    # Criteria for force closure
    if check_in_triangle([cone00,cone10,cone20],origin):
        pass
    elif check_in_triangle([cone00,cone11,cone20],origin):
        pass
    elif check_in_triangle([cone00,cone10,cone21],origin):
        pass
    elif check_in_triangle([cone00,cone11,cone21],origin):
        pass
    elif check_in_triangle([cone01,cone10,cone20],origin):
        pass
    elif check_in_triangle([cone01,cone11,cone20],origin):
        pass
    elif check_in_triangle([cone01,cone10,cone21],origin):
        pass
    elif check_in_triangle([cone01,cone11,cone21],origin):
        pass
    else:
        return False

    # Criteria for torque closure
    
    if check_sign_change(vectors, contact_points):
        return True
    else:
        return False
    
def simple_check_dyn_feasible(contact_points, contact_normals):
    if len(contact_points)==3:
        return simple_check_dyn_feasible_3p(contact_points, contact_normals)
    else:
        comb = combinations(list(range(len(contact_points))),3)
        for c in list(comb):
            c = list(c)
            if simple_check_dyn_feasible_3p(contact_points[c],contact_normals[c]):
                return True
        return False
    
    

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

    