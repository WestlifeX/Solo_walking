from numpy import zeros, dot, array, eye, tile
from second_order import plot_utils
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt

def compute_CoM_backoffs_magnitude(N, sigma_w, A_k, beta, CoM_constraint):
    eta_N   = zeros((N))
    #eta_N[0] = CoM_constraint
    cov_x_i = array([[0.0, 0.0], [0.0, 0.0]])
    cov_w   = array([[sigma_w[0]**2, 0.0], [0.0, sigma_w[1]**2]])
    for i in range(N):

        # propagate state covariance
        cov_x_i_plus_one = dot(dot(A_k, cov_x_i), A_k.T) + cov_w
        cov_x_i = cov_x_i_plus_one

        # compute inverse CDF
        scale = sqrt(cov_x_i_plus_one[0,0])
        eta_N[i] = norm.ppf(beta, scale=scale)
    #print eta_N
    return eta_N

def add_CoM_chance_constraints(N, y_hat_k, P_ps, P_pu, sigma_w, A_k, beta,
                                                                CoM_constraint):
    # pre-allocate memory
    A = zeros((N, 2*N))
    b = zeros(N)
    eta_N = compute_CoM_backoffs_magnitude(N, sigma_w, A_k, beta,CoM_constraint)
    A[0:N, N:2*N] = -P_pu
    b = -tile(CoM_constraint, N) + eta_N + dot(P_ps, y_hat_k)
    return A, b

def add_CoM_chance_constraints_box(N, y_hat_k, P_ps, P_pu, sigma_w, A_k, beta,
                                                                com_constraint):
    # pre-allocate memory
    A = zeros((2*N, 2*N))
    b = zeros(2*N)
    eta_N = compute_CoM_backoffs_magnitude(N, sigma_w, A_k, beta, com_constraint)
    A[0:N, N:2*N]   = -P_pu
    A[N:2*N, N:2*N] = P_pu
    b[0:N] = -tile(com_constraint, N) + eta_N + dot(P_ps, y_hat_k)
    b[N:2*N] = -tile(com_constraint, N) + eta_N - dot(P_ps, y_hat_k)
    return A, b

def add_CoM_chance_constraints_lfoot(no_constraints, N, y_hat_k, P_ps, P_pu,
                                            sigma_w, A_k, beta, com_constraint):
    # pre-allocate memory
    A = zeros((no_constraints, 2*N))
    b = zeros(no_constraints)

    eta_N = compute_CoM_backoffs_magnitude(N, sigma_w, A_k, beta, com_constraint)
    A[0:no_constraints, N:2*N] = -P_pu[N-no_constraints:N, :]
    b[0:no_constraints] = -tile(com_constraint, no_constraints) + \
           eta_N[N-no_constraints:N] + dot(P_ps[N-no_constraints:N, :], y_hat_k)
    return A, b

def add_CoM_chance_constraints_rfoot(no_constraints, N,  y_hat_k, P_ps, P_pu,
                                            sigma_w, A_k, beta, com_constraint):
    # pre-allocate memory
    A = zeros((no_constraints, 2*N))
    b = zeros(no_constraints)

    eta_N = compute_CoM_backoffs_magnitude(N, sigma_w, A_k, beta, com_constraint)
    A[0:no_constraints, N:2*N] = P_pu[N-no_constraints:N, :]
    b[0:no_constraints] = -tile(com_constraint, no_constraints) + \
           eta_N[N-no_constraints:N] - dot(P_ps[N-no_constraints:N, :], y_hat_k)
    return A, b

def compute_CoP_backoff_magnitude(N, sigma_w, K, A_k, beta):
    backoff_magnitude_N   = zeros((N))
    cov_w   = array([[sigma_w[0]**2, 0.0], [0.0, sigma_w[1]**2]])
    cov_x_i = array([[0.0, 0.0], [0.0, 0.0]])

    # compute inverse CDF
    for i in range(N):
        # propagate state covariance
        cov_x_i_plus_one = dot(dot(A_k, cov_x_i), A_k.T) + cov_w
        cov_x_i = cov_x_i_plus_one
        k_cov_x_i_plus_one = dot(dot(K, cov_x_i_plus_one), K.T)
        scale = sqrt(k_cov_x_i_plus_one)
        inv_CDF_i = norm.ppf(beta, scale=scale)
        backoff_magnitude_N[i] = inv_CDF_i
    #print backoff_magnitude_N
    return backoff_magnitude_N

def add_CoP_chance_constraints(N, foot_length, foot_width, Z_ref_k, K, A_k,
                                                                 sigma_w, beta):
    # pre-allocate memory
    A = zeros((4*N, 2*N))
    b = zeros((4*N))

    # x-direction
    A[0:N  , 0:N]   = eye(N)
    A[N:2*N, 0:N]   = -eye(N)
    cov_x_i = array([[0.0, 0.0], [0.0, 0.0]])

    # y-direction
    A[2*N:3*N, N:2*N] = eye(N)
    A[3*N:4*N, N:2*N] = -eye(N)

    # back-offs (y-direction only)
    uy_backoff_bN = compute_CoP_backoff_magnitude(N, sigma_w, K, A_k, beta)

    foot_length_N = tile(foot_length,(N))
    foot_width_N  = tile(foot_width,(N))

    b[0:N]     =  Z_ref_k[:,0] - (0.5*foot_length_N)
    b[N:2*N]   = -Z_ref_k[:,0] - (0.5*foot_length_N)
    b[2*N:3*N] =  Z_ref_k[:,1] - (0.5*foot_width_N) + uy_backoff_bN
    b[3*N:4*N] = -Z_ref_k[:,1] - (0.5*foot_width_N) + uy_backoff_bN
    return A, b
