import enum

import geomdl
import math
import numpy as np
import pybullet as p
import scipy.optimize
from sklearn.decomposition import PCA
import torch

from scipy.spatial.transform import Rotation as R

def fibonacci_sphere(samples=1):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.asarray(points)


def construct_T_from_position_quaterion(translation, quaternion):
    """
    Given translation t and rotation q, construct
    T = [R(q), t; 0, 1]
    :param translation:
    :param quaternion:
    :return:
    """
    T = np.zeros([4, 4])
    T[:3, -1] = translation
    T[:3, :3] = np.array(p.getMatrixFromQuaternion(quaternion)).reshape([3, 3])
    T[-1, -1] = 1
    return T


def construct_T_from_position_matrix(translation, matrix):
    if type(translation) == torch.Tensor:
        T = torch.zeros([4, 4]).type_as(translation)
        T[:3, -1] = translation
        T[:3, :3] = matrix
        T[-1, -1] = torch.tensor(1).type_as(translation)
        T = torch.clone(T)
    else:
        T = np.zeros([4, 4])
        T[:3, -1] = translation
        T[:3, :3] = matrix
        T[-1, -1] = 1
    return T


def compute_joint_T(joint_angle, joint_axis):
    """ This function is used to compute the transformation matrix for a
    rotation operation. You are supposed to make use of the self._theta
    parameter and the sin and cos functions which have already been imported.
    You will have 3 cases depending on which axis the rotation is about, which
    you can obtain from either self.axis or self.axis_index.

    Returns:
      transform: The 4x4 transformation matrix.
    """
    assert(len(joint_axis)==3)
    nonzero_axis = np.nonzero(joint_axis)[0]
    if len(nonzero_axis) == 0:
        # Fixed joint
        if type(joint_angle) == torch.Tensor:
            return torch.eye(4)
        else:
            return np.eye(4)
    assert len(nonzero_axis) == 1
    # Flip the joint angle if joint axis is negative
    if type(joint_angle) == torch.Tensor:
        joint_angle = torch.clone(joint_angle) * joint_axis[nonzero_axis[0]]
        cth, sth = torch.cos(joint_angle), torch.sin(joint_angle)
        tensor_0 = torch.zeros_like(joint_angle)
        tensor_1 = torch.ones_like(joint_angle)
        if nonzero_axis[0] == 0:
            R = torch.stack([
                torch.stack([tensor_1, tensor_0, tensor_0]),
                torch.stack([tensor_0, cth, -sth]),
                torch.stack([tensor_0, sth, cth])]).reshape(3, 3)
        elif nonzero_axis[0] == 1:
            R = torch.stack([
                torch.stack([cth, tensor_0, sth]),
                torch.stack([tensor_0, tensor_1, tensor_0]),
                torch.stack([-sth, tensor_0, cth])]).reshape(3, 3)
        elif nonzero_axis[0] == 2:
            R = torch.stack([
                torch.stack([cth, -sth, tensor_0]),
                torch.stack([sth, cth, tensor_0]),
                torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)
        else:
            raise AssertionError
        return torch.block_diag(R, tensor_1).type_as(joint_angle)
    else:
        joint_angle = np.copy(joint_angle) * joint_axis[nonzero_axis[0]]
        cth, sth = np.cos(joint_angle), np.sin(joint_angle)
        H = np.zeros((4, 4))
        H[3, 3] = 1.
        # H[:3, :3] = R.from_euler(self.axis, self._theta).as_matrix()
        if nonzero_axis[0] == 0:
            # axis is x
            H[1, 1] = cth
            H[2, 2] = cth
            H[1, 2] = -sth
            H[2, 1] = sth
            H[0, 0] = 1.
        elif nonzero_axis[0] == 1:
            # axis is y
            H[0, 0] = cth
            H[2, 2] = cth
            H[2, 0] = -sth
            H[0, 2] = sth
            H[1, 1] = 1.
        elif nonzero_axis[0] == 2:
            # axis is z
            H[0, 0] = cth
            H[1, 1] = cth
            H[0, 1] = -sth
            H[1, 0] = sth
            H[2, 2] = 1.
        else:
            raise AssertionError
    return H


def get_basis_from_3_normals(normals):
    '''
    Given three 6D normals vectors with positions of shape 3x(3+3), return a
    matrix with shape 3x4x3. The 4 dim1s are p, n, t1, t2, where p is the position
    component in the input, [n,t1,t2] is an orthonormal basis in 3D space.
    n is the normals component in the input, t1 is the basis that is parallel to
    the plane defined by the 3 p's, n is the basis perpendicular to the 3 p's
    :param normals:
    :return:
    '''
    assert normals.shape == (3, 6)
    if type(normals) == torch.Tensor:
        device = normals.device
        t1 = torch.vstack([normals[:, 4],
                           -normals[:, 3],
                           torch.zeros(3).to(device)]).transpose(0, 1).type_as(normals)
        for i in range(3):
            if torch.allclose(t1[i, :], torch.zeros(3).type_as(t1)):
                t1[i, 0] += 1

        t2 = torch.cross(normals[:, 3:], t1, dim=1)
        t1 = (t1.transpose(0, 1)/torch.linalg.norm(t1, dim=1)).transpose(0, 1)
        t2 = (t2.transpose(0, 1)/torch.linalg.norm(t2, dim=1)).transpose(0, 1)

        ans = torch.stack([normals[:, :3].reshape(3, 1, 3),
                           normals[:, 3:].reshape(3, 1, 3),
                           t1.reshape(3, 1, 3),
                           t2.reshape(3, 1, 3)
                           ], dim=1).reshape([3, 4, 3])
        return ans
    else:
        raise NotImplementedError


def get_cross_matrix(p):
    if isinstance(p, torch.Tensor):
        return torch.tensor([[0., -p[2], p[1]],
                             [p[2], 0., -p[0]],
                             [-p[1], p[0], 0.]], requires_grad=p.requires_grad)
    else:
        return np.array([[0., -p[2], p[1]],
                         [p[2], 0., -p[0]],
                         [-p[1], p[0], 0.]])

def reformulate_to_soft_QP(Q_0, p_0, G_0, h_0, A_0, b_0,
                           inequality_constraint_weight,
                           equality_constraint_weight
                           ):
    '''
    Transforms a QP of the form
    min zᵀQ₀z + p₀ᵀz s.t. A₀z=b₀, G₀z≤h₀
    into a "soft" QP where Az=b is penalized with equality_constraint_weight
    and Gz≤h is penalized with inequality_constraint_weight:
    min z̃ᵀQ̃z̃ + p̃ᵀz̃ s.t. z̃ ≥ 0
    Dropping the constant terms,
    the constraint A₀z=b₀ is transformed to quadratic penalty
    min (A₀z−b₀)ᵀ(A₀z-b₀) → min zᵀA₀ᵀA₀z−2b₀ᵀA₀z
    and the constraint Gz≤h is transformed to quadratic penalty with slack
    variables s
    min (G₀z-h₀+s)ᵀ(G₀z-h₀+s) → min zᵀG₀ᵀG₀z+2sᵀG₀z+sᵀs−2h₀ᵀG₀z−2h₀ᵀs
    Rewriting the optimization by defining
    z̃ = [z; s], len(s) = len(h₀)
    The original costs become
    Q̃₀ = block_diag([Q₀, 0ₛ])
    p̃₀ᵀ = [p₀ᵀ, 0]
    The equality constraint cost weights are
    Q̃ₑ = block_diag([A₀ᵀA₀, 0ₛ])*equality_constraint_weight
    p̃ₑᵀ = [−2b₀ᵀA₀, 0]*equality_constraint_weight
    The inequality constraint cost weights are
    Q̃ᵢₑ = [G₀ᵀG₀, G₀; G₀ᵀ, Iₛ]*inequality_constraint_weight
    p̃ᵢₑᵀ = [-2h₀ᵀG₀, -2h₀ᵀ]*inequality_constraint_weight
    The total optimization becomes
    Q̃=Q̃₀+Q̃ₑ+Q̃ᵢₑ
    p̃ᵀ=p̃₀ᵀ+p̃ₑᵀ+p̃ᵢₑᵀ
    Ã, b̃ = Variable(torch.Tensor())
    G̃ = -I
    h̃ = 0
    :return: The matrices defining the soft QP Q̃, p̃, G̃, h̃, Ã, b̃
    '''
    raise DeprecationWarning
    slack_count = len(h_0)
    zeros_vec_slack_count = torch.zeros(slack_count).type_as(Q_0)
    zeros_mat_slack_count = torch.zeros(
        [slack_count, slack_count]).type_as(Q_0)
    eye_slack_count = torch.eye(slack_count).type_as(Q_0)
    Q_0_tilde = torch.block_diag(Q_0, zeros_mat_slack_count)
    p_0_tilde = torch.hstack([p_0, zeros_vec_slack_count])
    Q_e_tilde = torch.block_diag(A_0.transpose(0, 1)@A_0,
                                 zeros_mat_slack_count) *\
        equality_constraint_weight
    p_e_tilde = torch.hstack([-2.*b_0@A_0,
                              zeros_vec_slack_count])*equality_constraint_weight

    Q_ie_tilde = torch.vstack(
        [torch.hstack([G_0.transpose(0, 1)@G_0, G_0.transpose(0, 1)]),
         torch.hstack([G_0, eye_slack_count])]) *\
        inequality_constraint_weight
    p_ie_tilde = torch.hstack([-2.*h_0@G_0, -2.*h_0]) * \
        inequality_constraint_weight
    Q_tilde = (Q_0_tilde+Q_e_tilde+Q_ie_tilde).type_as(Q_0)
    p_tilde = (p_0_tilde+p_e_tilde+p_ie_tilde).type_as(Q_0)
    G_tilde = -torch.eye(Q_tilde.shape[0]).type_as(Q_0)
    h_tilde = torch.zeros(Q_tilde.shape[0]).type_as(Q_0)
    A_tilde = torch.autograd.Variable(torch.Tensor()).type_as(Q_0)
    b_tilde = torch.autograd.Variable(torch.Tensor()).type_as(Q_0)
    return Q_tilde, p_tilde, G_tilde, h_tilde, A_tilde, b_tilde


def reformulate_eq_to_soft_QP(Q_0, p_0, G_0, h_0, A_0, b_0,
                              equality_constraint_weight
                              ):
    '''
    Transforms a QP of the form
    min zᵀQ₀z + p₀ᵀz s.t. A₀z=b₀, G₀z≤h₀
    into a "soft" QP where Az=b is penalized with equality_constraint_weight
    but Gz≤h is still enforced.
    min zᵀQ̃z + p̃ᵀz s.t. z ≥ 0, G₀z≤h₀
    Dropping the constant terms
    the constraint A₀z=b₀ is transformed to quadratic penalty
    min (A₀z−b₀)ᵀ(A₀z-b₀) → min zᵀA₀ᵀA₀z−2b₀ᵀA₀z
    The equality constraint cost weights are
    Q̃ₑ = block_diag([A₀ᵀA₀])*equality_constraint_weight
    p̃ₑᵀ = [−2b₀ᵀA₀]*equality_constraint_weight
    The total optimization becomes
    Q̃=Q₀+Q̃ₑ
    p̃ᵀ=p₀ᵀ+p̃ₑᵀ
    Ã, b̃ = Variable(torch.Tensor())
    G̃ = [G₀; -I]
    h̃ = [h₀; 0]
    :return: The matrices defining the soft QP Q̃, p̃, G̃, h̃, Ã, b̃
    '''
    equality_constraint_weight = equality_constraint_weight.to(A_0.device)
    # Synced all devices to match Q_0
    p_0, G_0, h_0, A_0, b_0, equality_constraint_weight = p_0.to(Q_0.device), G_0.to(Q_0.device), h_0.to(Q_0.device), A_0.to(Q_0.device), b_0.to(Q_0.device), equality_constraint_weight.to(Q_0.device)
    Q_e_tilde = A_0.transpose(0, 1)@A_0*equality_constraint_weight
    p_e_tilde = -torch.tensor(2.,device=Q_0.device)*b_0@A_0*equality_constraint_weight
    Q_tilde = (Q_0+Q_e_tilde).type_as(Q_0)
    p_tilde = (p_0+p_e_tilde).type_as(Q_0)
    G_tilde = torch.vstack(
        [G_0, -torch.eye(Q_0.shape[0]).type_as(Q_0)]).type_as(Q_0)
    h_tilde = torch.hstack(
        [h_0, torch.zeros(Q_0.shape[0]).type_as(Q_0)]).type_as(Q_0)
    A_tilde = torch.autograd.Variable(torch.Tensor()).type_as(Q_0)
    b_tilde = torch.autograd.Variable(torch.Tensor()).type_as(Q_0)
    return Q_tilde, p_tilde, G_tilde, h_tilde, A_tilde, b_tilde


def polynomial_function_deg3(data, a, b, c, d, e, f, g, h, i, j):
    """
    z = f(x,y) = ax³+by³+cx²y+dxy²+ex²+fy²+gxy+hx+iy+j
    :param data: nx2 data
    :return: nx1
    """
    x = data[:,0]
    y = data[:,1]
    return a*x**3+b*y**3+c*x**2*y+d*x*y**2+e*x**2+f*y**2+g*x*y+h*x+i*y+j

def polynomial_function_deg3_grad(data, a, b, c, d, e, f, g, h, i, j):
    """
    ∂f/∂x = 3ax²+2cxy+dy²+2ex+gy+h
    ∂f/∂y = 3by²+cx²+2dxy+2fy+gx+i
    :param data:
    :return: 2x1 numpy array [∂f/∂x, ∂f/∂y]ᵀ
    """
    x = data[:,0]
    y = data[:,1]
    return np.array([[3*a*x**2+2*c*x*y+d*y**2+2*e*x+g*y+h],
                     [3*b*y**2+c*x**2+2*d*x*y+2*f*y+g*x+i]])

def polynomial_function_deg1(data, a, b, c):
    """
    z = ax+by+c
    :param data:
    :return:
    """
    x = data[:,0]
    y = data[:,1]
    return a*x+b*y+c

def polynomial_function_deg1_grad(data, a, b, c):
    return np.array([[a], [b]])

class SurfaceFitFunction(enum.Enum):
    POLYNOMIAL_DEG3 = (polynomial_function_deg3, polynomial_function_deg3_grad)
    POLYNOMIAL_DEG1 = (polynomial_function_deg1, polynomial_function_deg1_grad)

def get_fitted_fn(fn_type, popt):
    fn, fn_grad = fn_type.value
    def fitted_f(x):
        return fn(x, *popt)
    def fitted_f_grad(x):
        return fn_grad(x, *popt)
    return fitted_f, fitted_f_grad

def compute_principal_components(points):
    """

    :param points:
    :return: 3-tuple
    ndarray of shape (n_components, n_features)
    ndarray of shape (n_features,)
    """
    pca = PCA(n_components=3)
    transformed_points = pca.fit_transform(points)
    return pca.components_, pca.mean_, transformed_points


def fit_surface_to_points(points,
                          fit_function=SurfaceFitFunction.POLYNOMIAL_DEG1,
                          **kwargs):
    """
    Given nx3 points, find the rotation matrix that provide.
    :param points:
    :param fit_function:
    :param kwargs:
    :return: (R, the best rotation matrix that fits the
    """
    assert points.shape[1] == 3
    return scipy.optimize.curve_fit(fit_function.value[0], points[:, :2], points[:, 2])


def transform_and_fit_surface_to_points(
        p_W,
        fit_function=SurfaceFitFunction.POLYNOMIAL_DEG3,
        **kwargs):
    """

    :param p_W:
    :param fit_function:
    :param kwargs:
    :return:
    (R, t, fitted_points, fitted_fn, transformed_fitted_fn)
    Given a point x, y, the predicted function is
    """
    # First do PCA
    components, mean, p_S = compute_principal_components(p_W)
    # Fit to the transformed points
    popt, pcov = fit_surface_to_points(p_S,
                          fit_function,**kwargs)
    # Reconstruct the function
    fitted_fn_S, fitted_fn_grad_S = get_fitted_fn(fit_function, popt)
    R_WS = components.T
    t_WS = mean
    return R_WS, t_WS, p_S, fitted_fn_S, fitted_fn_grad_S

def transform_and_fit_consistent_normal_surface_to_points(
        all_p_W,
        fit_function=SurfaceFitFunction.POLYNOMIAL_DEG3,
        **kwargs):
    """

    :param points: 3D array of shape [3, n, 3]
    :param fit_function:
    :param kwargs:
    :return:
    (R, t, fitted_points, fitted_fn, transformed_fitted_fn)
    Given a point x, y, the predicted function is
    """
    assert len(all_p_W) == 3
    for i in range(len(all_p_W)):
        assert all_p_W[i].shape[1] == 3
    all_components = np.zeros([3, 3, 3])
    all_means = np.zeros([3, 3])
    for idx, points in enumerate(all_p_W):
        # First do PCA
        components, mean, _ = compute_principal_components(points)
        all_components[idx, :, :] = components
        all_means[idx,:] = mean
    all_pavg_W = np.average(all_means, axis=0)
    ans = []
    for idx, points in enumerate(all_p_W):
        # Flip the signs of the basis if necessary
        sign = int(np.dot(all_means[idx,:]-all_pavg_W, all_components[idx, 2, :]) < 0)
        all_components[idx] *= (-1)**sign
        # Make sure the normals are consistent
        # Fit to the transformed points
        # Reconstruct the function
        R_WS = all_components[idx].T
        t_WS = all_means[idx,:]
        p_S = (R_WS.T @ (all_p_W[idx]-t_WS).T).T
        popt, pcov = fit_surface_to_points(p_S,
                              fit_function, **kwargs)
        fitted_fn, fitted_fn_grad = get_fitted_fn(fit_function, popt)
        ans.append((R_WS, t_WS, p_S, fitted_fn, fitted_fn_grad))
    return ans

def compute_nF_S(p_S, fitted_fn_grad):
    """
    Compute the fitted surface normal of a point p_S in the PCA basis space
    See https://en.wikipedia.org/wiki/Normal_(geometry)#Calculating_a_surface_normal for math
    :param p_S:
    :param fitted_fn_grad:
    :return: nF_S
    """
    grad_p = np.squeeze(fitted_fn_grad(p_S[:2,:].reshape(1,-1)))
    nF_S = np.array([-grad_p[0], -grad_p[1], 1.])
    return nF_S / np.linalg.norm(nF_S)


def compute_n_W(p_W, R_WS, t_WS, fitted_fn_grad):
    p_S = np.matmul(R_WS.T, p_W.reshape(3,1) - t_WS.reshape(3,1))
    n_S = compute_nF_S(p_S, fitted_fn_grad)
    return R_WS @ n_S