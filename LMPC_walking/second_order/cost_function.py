#    LMPC_walking is a python software implementation of some of the linear MPC
#    algorithms based presented in:
#    https://groups.csail.mit.edu/robotics-center/public_papers/Wieber15.pdf
#
#    Copyright (C) 2019 @ahmad gazar

#    LMPC_walking is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    LMPC_walking is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# headers:
# -------
import numpy as np
from numpy import eye, zeros, dot, tile, reshape, nditer
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Description:
# -----------
# this function compute the canonical quadratic objective term Q (hessian)
# and the linear objective term p.T (gradient) of this cost function:
# min zx_k, zy_k
#    beta/2||x-x_r||^2 + gamma/2||x_dot-x_dot_r||^2 + alpha ||z_x-zx_r||^2
#  + beta/2||y-y_r||^2 + gamma/2||y_dot-y_dot_r||^2 + alpha ||z_y-zy_r||^2

# Parameters:
# ----------
# alpha          : CoP error squared cost weight            (scalar)
# beta           : CoM position error squared cost weight   (scalar)
# gamma          : CoM velocity error squared cost weight   (scalar)
# N              : preceding horizon                        (scalar)
# P_ps, P_pu     : CoM position recursive dynamics matrices (Nx2 numpy.array,
#                  x^_k+1 = P_ps x^_k + P_pu z_k             NXN numpy.array)
# P_vs , P_vu    : CoM velocity recursive dynamics matrix   (Nx2 numpy.array,
#                  x^dot_k+1 = P_vs x^dot_k + P_vu z_k       NXN numpy.array)
# x_hat_k        : [x^_k, x^dot_k].T current CoM state in x (2, numpy.array)
# y_hat_k        : [y^_k, y^dot_k].T current CoM state in y (2, numpy.array)
# Z_ref_k        : [z_ref_x_k  , z_ref_y_k  ]   CoP reference trajectory
#                   .       ,    .                          (Nx2 numpy.array)
#                   .       ,    .
#                  [z_ref_x_k+N, z_ref_y_k+N]

# Returns:
# -------
# Q     : Hessian  (2Nx2N numpy.array)
# p_k   : Gradient (2N,  numpy.array)

def compute_objective_terms(alpha, beta, gamma, step_duration, no_steps_per_T,
                            N, stride_length, stride_width, P_ps, P_pu, P_vs,
                            P_vu, x_hat_k, y_hat_k, Z_ref_k):

    # pre-allocate memory
    Q    = zeros((2*N, 2*N))
    p_k  = zeros((2*N))
    Q_prime = zeros((N,N))

    Q_prime = alpha*eye(N) + (beta*dot(P_pu.T, P_pu)) + (gamma*dot(P_vu.T, P_vu))
    Q[0:N, 0:N]     = Q_prime  # x-direction
    Q[N:2*N, N:2*N] = Q_prime  # y-direction

    x_r_N = zeros((N))
    y_r_N = zeros((N))
    x_dotr_N = zeros((N))
    y_dotr_N = zeros((N))

    x_r_N    = tile(stride_length, N)
    y_r_N    = tile(0.05, N)
    x_dotr_N = tile(stride_length/step_duration, N)
    y_dotr_N = tile(stride_width/step_duration, N)

    p_k[0:N] = gamma*(dot(P_vu.T, (dot(P_vs, x_hat_k)))- dot(P_vu.T, x_dotr_N))\
               + beta*(dot(P_pu.T, (dot(P_ps, x_hat_k)))- dot(P_pu.T, x_r_N)) \
               - alpha*Z_ref_k[:,0]

    p_k[N:2*N] = gamma*(dot(P_vu.T, (dot(P_vs, y_hat_k)))- dot(P_vu.T, y_dotr_N))\
                 + beta *(dot(P_pu.T, (dot(P_ps, y_hat_k)))- dot(P_pu.T, y_r_N))\
                 - alpha*Z_ref_k[:,1]
    return Q, p_k

# I've added This function
# same function as before to work with quadrupeds
def compute_objective_terms_quad(alpha, beta, gamma, step_duration, no_steps_per_T,
                            N, stride_length, stride_width, P_ps, P_pu, P_vs,
                            P_vu, x_hat_k, y_hat_k, Z_ref_k):

    # pre-allocate memory
    Q    = zeros((2*N, 2*N))
    p_k  = zeros((2*N))
    Q_prime = zeros((N,N))

    Q_prime = alpha*eye(N) + (beta*dot(P_pu.T, P_pu)) + (gamma*dot(P_vu.T, P_vu))
    Q[0:N, 0:N]     = Q_prime  # x-direction
    Q[N:2 * N, N:2 * N] = 1e-18*eye(N)

    x_r_N = zeros((N))
    y_r_N = zeros((N))
    x_dotr_N = zeros((N))
    y_dotr_N = zeros((N))

    x_r_N    = tile(stride_length, N)
    x_dotr_N = tile(stride_length/step_duration, N)

    p_k[0:N] = gamma*(dot(P_vu.T, (dot(P_vs, x_hat_k)))- dot(P_vu.T, x_dotr_N))\
               + beta*(dot(P_pu.T, (dot(P_ps, x_hat_k)))- dot(P_pu.T, x_r_N)) \
               - alpha*Z_ref_k[:,0]
    p_k[N : 2*N] = 0*np.ones(N)

    return Q, p_k

def compute_objective_terms_box(alpha, beta, gamma, step_duration, no_steps_per_T,
                         N, stride_length, stride_width, P_ps, P_pu, P_vs, P_vu,
                             x_hat_k, y_hat_k, Z_ref_k, com_constraint_desired):

    # pre-allocate memory
    Q    = zeros((2*N, 2*N))
    p_k  = zeros((2*N))
    Q_prime = zeros((N,N))

    Q_prime = alpha*eye(N) + (beta*dot(P_pu.T, P_pu)) + (gamma*dot(P_vu.T, P_vu))
    Q[0:N, 0:N]     = Q_prime  # x-direction
    Q[N:2*N, N:2*N] = Q_prime  # y-direction

    x_r_N = zeros((N))
    y_r_N = zeros((N))
    x_dotr_N = zeros((N))
    y_dotr_N = zeros((N))

    x_r_N    = tile(stride_length, N)
    y_r_N    = com_constraint_desired
    x_dotr_N = tile(stride_length/step_duration, N)
    y_dotr_N = y_r_N/step_duration

    p_k[0:N] = gamma*(dot(P_vu.T, (dot(P_vs, x_hat_k)))- dot(P_vu.T, x_dotr_N))\
               + beta*(dot(P_pu.T, (dot(P_ps, x_hat_k)))- dot(P_pu.T, x_r_N)) \
               - alpha*Z_ref_k[:,0]

    p_k[N:2*N] = gamma*(dot(P_vu.T, (dot(P_vs, y_hat_k)))- dot(P_vu.T, y_dotr_N))\
                 + beta *(dot(P_pu.T, (dot(P_ps, y_hat_k)))- dot(P_pu.T, y_r_N))\
                 - alpha*Z_ref_k[:,1]
    return Q, p_k

#TODO
def compute_objective_terms_with_automatic_footstep_adjustment(alpha, beta,
                        gamma, step_duration, N, stride_length, stride_width,
                        P_ps, P_pu, P_vs, P_vu, x_hat_k, y_hat_k, U_k_plus_one,
                        U_c_k_plus_one, X_fc_k, Y_fc_k, m, i):

    # pre-allocate memory
    Q    = zeros((2*(N+m), 2*(N+m)))
    p_k  = zeros((2*(N+m)))
    Q_prime = zeros((N+m, N+m))

    Q_prime[0:N  , 0:N]   = alpha*eye(N) + (beta*dot(P_pu.T, P_pu)) \
                          + (gamma*dot(P_vu.T, P_vu))
    Q_prime[0:N  , N:N+m] = -alpha*U_k_plus_one
    Q_prime[N:N+m, 0:N]   = -alpha*U_k_plus_one.T
    Q_prime[N:N+m, N:N+m] = alpha*dot(U_k_plus_one.T, U_k_plus_one)

    Q[0:N+m, 0:N+m]             = Q_prime  # x-direction
    Q[N+m:2*(N+m), N+m:2*(N+m)] = Q_prime  # y-direction

    # CoM position and velocity reference trajectories
    x_r_N    = tile(stride_length, N)
    y_r_N    = tile(0.05, N)
    x_dotr_N = tile(stride_length/step_duration, N)
    y_dotr_N = tile(stride_width/step_duration, N)

    # x-direction
    p_k[0:N] = gamma*(dot(P_vu.T, (dot(P_vs, x_hat_k)))- dot(P_vu.T, x_dotr_N))\
             +  beta*(dot(P_pu.T, (dot(P_ps, x_hat_k)))- dot(P_pu.T, x_r_N)) \
             - alpha*dot(U_c_k_plus_one, X_fc_k)
    p_k[N:N+m] = alpha*dot(U_k_plus_one.T, X_fc_k*U_c_k_plus_one)

    # y-direction
    p_k[N+m:(2*N)+m] = gamma*(dot(P_vu.T,(dot(P_vs, y_hat_k))) \
                            - dot(P_vu.T, y_dotr_N))\
                     +  beta*(dot(P_pu.T, (dot(P_ps, y_hat_k))) \
                            - dot(P_pu.T, y_r_N)) \
                     - alpha*dot(U_c_k_plus_one, Y_fc_k)
    p_k[(2*N)+m:2*(N+m)] = alpha*dot(U_k_plus_one.T, Y_fc_k*U_c_k_plus_one)
    return Q, p_k

# ------------------------------------------------------------------------------
# unit test: A.K.A red pill or blue pill
# ------------------------------------------------------------------------------
if __name__=='__main__':
    import numpy.random as random
    import motion_model
    print(' visualize your matrices like a Neo ! '.center(60,'*'))
    delta_t = 0.1
    h       = 0.80
    g       = 9.81
    alpha   = 1
    gamma   = 1
    beta    = 0
    N       = 16
    stride_length = 0.21
    stride_width  = 2*random.rand(1)-0.5
    step_time       = 0.8
    no_steps_per_T  = int(round(step_time/delta_t))
    Z_ref_k = random.rand(N,2)-0.5
    x_hat_k = random.rand(2)-0.5
    y_hat_k = random.rand(2)-0.5
    U_c_k_plus_one = zeros(N)
    U_c_k_plus_one[0:N//2] = 1.0
    U_k_plus_one = zeros((N,2))
    U_k_plus_one[0:N//2,0] = 1.0
    U_k_plus_one[N//2:N,1] = 1.0
    X_fc_k = 0.0
    Y_fc_k = -0.10
    m = 2
    [P_ps, P_vs, P_pu, P_vu] = motion_model.compute_recursive_matrices(delta_t,
                                                                       g, h, N)
    Q, p = compute_objective_terms(alpha, beta, gamma, step_time, no_steps_per_T,
                                   N, stride_length, stride_width, P_ps, P_pu,
                                   P_vs, P_vu, x_hat_k, y_hat_k, Z_ref_k)
    #Q, p = compute_objective_terms_with_automatic_footstep_adjustment(alpha,beta,
    #    gamma, step_time, N, stride_length, stride_width, P_ps, P_pu, P_vs,P_vu,
    #    x_hat_k, y_hat_k, U_k_plus_one, U_c_k_plus_one, X_fc_k, Y_fc_k, m)
    p    = reshape(p, (p.size,1))
# ------------------------------------------------------------------------------
# visualize your Hessian and gradient like a Neo:
# ------------------------------------------------------------------------------
    with nditer(Q, op_flags=['readwrite']) as it:
        for x in it:
            if x[...] != 0:
                x[...] = 1
    plt.figure(1)
    plt.suptitle('Structure of hessian matrix Q')
    plt.imshow(Q, cmap='Greys', extent=[0,Q.shape[1],Q.shape[0],0],
    interpolation = 'nearest')
    #with np.nditer(p, op_flags=['readwrite']) as it:
    #    for x in it:
    #        if x[...] != 0:
    #            x[...] = 1
    plt.figure(2)
    plt.imshow(p, cmap='Greys',  extent=[0,p.shape[1],p.shape[0],0],
    interpolation = 'nearest', aspect=0.25)
    plt.suptitle('Structure of gradient vector P')
    plt.show()
