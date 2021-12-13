#    LMPC_walking is a python software implementation of some of the linear MPC
#    algorithms based presented in:
#    https://groups.csail.mit.edu/robotics-center/public_papers/Wieber15.pdf
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
from numpy import zeros, ones, eye, dot, tile, reshape, nditer, array, sqrt
import matplotlib.pyplot as plt
from PIL import Image

# Desctiption:
# -----------
# this function assembles A and b encapsulating the ZMP inequality constraints
# in the Quadratic Program of this form (quadprog solver format):
#        minimize_u
#            (1/2) * u.T * Q * u - p.T * u
#        subject to
#            A.T * u >= b
# The foot shape is assumed to be rectangular

# Parameters:
# ----------
# N           : preceding horizon length                (scalar)
# foot_length : length of the foot along the x-axis     (scalar)
# foot_width  : length of the foot along the y-axis     (scalar)
# Z_ref_k     : [z_ref_x_k  , z_ref_y_k  ]   CoP reference trajectory
#                   .       ,    .                      (Nx2 numpy.array)
#                   .       ,    .
#               [z_ref_x_k+N, z_ref_y_k+N]
# x_hat_k     : [x_k, x_dot_k].T current CoM state in x (2, numpy.array)
# y_hat_k     : [y_k, y_dot_k].T current CoM state in y (2, numpy.array)

# Returns:
# -------
# A  : (4Nx2N numpy.array)
#      matrix defining defining the linear terms in the CoP control inputs in
#      x and y directions taking this form:
#      Z_x_k <= Zx_ref_k + foot_length/2
#      Z_x_k >= Zx_ref_k - foot_length/2
#      Z_y_k <= Zy_ref_k + foot_width/2
#      Z_y_k >= Zy_ref_k - foot_width/2
# b  : (4N, numpy.array)
#      vector defining the remaining terms

def add_ZMP_constraints(N, foot_length, foot_width, Z_ref_k, x_hat_k, y_hat_k):

    # pre-allocate memory
    A = zeros((4*N, 2*N))
    b = zeros((4*N))

    # x-direction
    A[0:N  , 0:N]   = eye(N)
    A[N:2*N, 0:N]   = -eye(N)

    # y-direction
    A[2*N:3*N, N:2*N] = eye(N)
    A[3*N:4*N, N:2*N] = -eye(N)

    foot_length_N = tile(foot_length,(N))
    foot_width_N  = tile(foot_width,(N))

    b[0:N]     =  Z_ref_k[:,0] - (0.5*foot_length_N)
    b[N:2*N]   = -Z_ref_k[:,0] - (0.5*foot_length_N)
    b[2*N:3*N] =  Z_ref_k[:,1] - (0.5*foot_width_N)
    b[3*N:4*N] = -Z_ref_k[:,1] - (0.5*foot_width_N)
    print(A)
    print(b)
    return A,b

def add_ZMP_constraints_quad(N, Foot_steps, D, l):

    # pre-allocate memory
    A = zeros((3*N, 2*N))
    b = zeros((3*N))

    # x-direction equality
    A[0:N, 0:N] = eye(N)
    A[0:N, N:2 * N] = -eye(N)* D [:,0]

    # alpha inequalities
    A[N:2 * N, N:2 * N] = eye(N)
    A[2*N:3 * N, N:2 * N] = -eye(N)

    b[0:N] = Foot_steps[:,0]
    b[N:2 * N] = 0 * ones(N)
    b[2*N : 3*N] = -l * ones(N)


    return A,b

def add_ZMP_constraints_with_automatic_footstep_adjustment(N, foot_length,
                    foot_width, x_hat_k, y_hat_k, U_k_plus_one, U_c_k_plus_one,
                    X_fc_k, Y_fc_k, m):

    # pre-allocate memory
    A = zeros((4*N, 2*(N+m)))
    b = zeros((4*N))

    # x-direction
    A[0:N  , 0:N]   = eye(N)
    A[0:N  , N:N+m] = -U_k_plus_one
    A[N:2*N, 0:N]   = -eye(N)
    A[N:2*N, N:N+m] = U_k_plus_one

    # y-direction
    A[2*N:3*N, N+m:(2*N)+m]     = eye(N)
    A[2*N:3*N, (2*N)+m:2*(N+m)] = -U_k_plus_one
    A[3*N:4*N, N+m:(2*N)+m]     = -eye(N)
    A[3*N:4*N, (2*N)+m:2*(N+m)] = U_k_plus_one

    foot_length_N = tile(foot_length,(N))
    foot_width_N  = tile(foot_width,(N))

    b[0:N]     =  X_fc_k*U_c_k_plus_one - (0.5*foot_length_N)
    b[N:2*N]   = -X_fc_k*U_c_k_plus_one - (0.5*foot_length_N)
    b[2*N:3*N] =  Y_fc_k*U_c_k_plus_one - (0.5*foot_width_N)
    b[3*N:4*N] = -Y_fc_k*U_c_k_plus_one - (0.5*foot_width_N)

    return A, b

# Desctiption:
# -----------
# this function assembles A_eq matrix and b_eq vector encapsulating the CoM
# equality terminal constraints at the end of the preceding horizon

# Parameters:
# ----------
# N              : preceding horizon length                 (scalar)
# terminal_index : array index of the terminal constraint   (scalar)
# x_hat_k        : [x^_k, x^dot_k].T current CoM state in x (2, numpy.array)
# y_hat_k        : [y^_k, y^dot_k].T current CoM state in y (2, numpy.array)
# x_terminal     : [x_t, x_dot_t].T terminal CoM state in x (2, numpy.array)
# y_terminal     : [y_t, y_dot_t].T terminal CoM state in y (2, numpy.array)
# P_ps, P_pu     : CoM position recursive dynamics matrices (Nx2 numpy.array,
#                  x^_k+1 = P_ps x^_k + P_pu z_k            (NXN numpy.array)
# P_vs, P_vu     : CoM velocity recursive dynamics matrix   (Nx2 numpy.array,
#                  x^dot_k+1 = P_vs x^dot_k + P_vu z_k      (NXN numpy.array)

# Returns:
# -------
# A_eq  : (4x2N numpy.array)
#         matrix defining the linear terms in the CoM state. However, since the
#         MPC problem decision variables are only the CoP control inputs
#         then the CoM equality terminal constraints can be formulated in terms
#         of the CoP control inputs in x and y as follows:
#         P_ps x^_k + P_pu[terminal_index,:] z_x_k = x_terminal[0]
#         P_vs x^_k + P_vu[terminal_index,:] z_x_k = x_terminal[0]

# b_eq  : (4, numpy.array)
#         vector defining the remaining terms

def add_terminal_constraints(N, terminal_index, x_hat_k, y_hat_k, x_terminal,
                             y_terminal, P_ps, P_vs, P_pu, P_vu):

    # pre-allocate memory
    A_eq = zeros((4, 2*N))
    b_eq = zeros((4))

    P_ps_current = P_ps[terminal_index,:]
    P_vs_current = P_vs[terminal_index,:]

    P_pu_current = P_pu[terminal_index,:]
    P_vu_current = P_vu[terminal_index,:]

    # x-direction
    A_eq[0, 0:N]    = P_pu_current
    A_eq[1, 0:N]    = P_vu_current

    # y-direction
    A_eq[2, N:2*N]  = P_pu_current
    A_eq[3, N:2*N]  = P_vu_current

    b_eq[0] = x_terminal[0] - dot(P_ps_current, x_hat_k)
    b_eq[1] = x_terminal[1] - dot(P_vs_current, x_hat_k)
    b_eq[2] = y_terminal[0] - dot(P_ps_current, y_hat_k)
    b_eq[3] = y_terminal[1] - dot(P_vs_current, y_hat_k)

    return A_eq, b_eq


def add_terminal_constraints_quad(N, terminal_index, x_hat_k, y_hat_k, x_terminal,
                             y_terminal, P_ps, P_vs, P_pu, P_vu, foot_steps, D):

    # pre-allocate memory
    A_eq = zeros((4, 2*N))
    b_eq = zeros((4))

    P_ps_current = P_ps[terminal_index,:]
    P_vs_current = P_vs[terminal_index,:]

    P_pu_current = P_pu[terminal_index,:]
    P_vu_current = P_vu[terminal_index,:]

    # x-direction
    A_eq[0, 0:N]    = P_pu_current
    A_eq[1, 0:N]    = P_vu_current

    # y-direction --> alpha
    A_eq[2, N:2 * N] = P_pu_current * D[:, 1]
    A_eq[3, N:2 * N] = P_vu_current * D[:, 1]


    b_eq[0] = x_terminal[0] - dot(P_ps_current, x_hat_k)
    b_eq[1] = x_terminal[1] - dot(P_vs_current, x_hat_k)

    b_eq[2] = y_terminal[0] - dot(P_ps_current, y_hat_k) - np.dot(P_pu_current, foot_steps[:, 1])
    b_eq[3] = y_terminal[1] - dot(P_vs_current, y_hat_k) - np.dot(P_vu_current, foot_steps[:, 1])

    return A_eq, b_eq



# Desctiption:
# -----------
# this function assembles A matrix and b vector encapsulating the capturability
# terminal constraints at the end of the preceding horizon
# \xi_x = x + (1/w)*x^dot \in P_x(t)
# \xi_y = y + (1/w)*y^dot \in P_y(t)

# Parameters:
# ----------
# N              : preceding horizon length                 (scalar)
# g              : norm of the gravity acceleration vector  (scalar)
# h              : fixed height of the CoM assuming walking on a flat terrain
# x_hat_k        : [x^_k, x^dot_k].T current CoM state in x (2, numpy.array)
# y_hat_k        : [y^_k, y^dot_k].T current CoM state in y (2, numpy.array)
# Z_ref_k        : [z_ref_x_k  , z_ref_y_k  ]    CoP reference trajectory
#                   .       ,    .                          (Nx2 numpy.array)
#                   .       ,    .
#                  [z_ref_x_k+N, z_ref_y_k+N]
# P_ps, P_pu     : CoM position recursive dynamics matrices (Nx2 numpy.array,
#                  x^_k+1 = P_ps x^_k + P_pu z_k            (NXN numpy.array)
# P_vs, P_vu     : CoM velocity recursive dynamics matrix   (Nx2 numpy.array,
#                  x^dot_k+1 = P_vs x^dot_k + P_vu z_k      (NXN numpy.array)
# foot_length    : length of the foot along the x-axis      (scalar)
# foot_width     : length of the foot along the y-axis      (scalar)

# Returns:
# -------
# A : (4x2N numpy.array)
#      matrix defining the linear terms in the CoM state. However, since the
#      MPC problem decision variables are only the CoP control inputs
#      then the capturability terminal constraints can be formulated in terms
#      of the CoP control inputs in x and y as follows:
#      P_pu[N-1,:] + (1/w)*P_vu[N-1,:] Zx_k <= Zx_ref_k[N-1, 0] - foot_length/2
#                                             -(P_ps[N-1] + (1/w)*P_vs[N-1,:])
#      P_pu[N-1,:] + (1/w)*P_vu[N-1,:] Zx_k <=-Zx_ref_k[N-1, 0] - foot_length/2
#                                             +(P_ps[N-1] + (1/w)*P_vs[N-1,:])
#      P_pu[N-1,:] + (1/w)*P_vu[N-1,:] Zy_k <= Zy_ref_k[N-1, 0] - foot_width/2
#                                             -(P_ps[N-1] + (1/w)*P_vs[N-1,:])
#      P_pu[N-1,:] + (1/w)*P_vu[N-1,:] Zy_k <=-Zy_ref_k[N-1, 0] - foot_width/2
#                                             +(P_ps[N-1] + (1/w)*P_vs[N-1,:])

# b : (4, numpy.array)
#     vector defining the remaining terms

def add_capturability_terminal_constraints(N, g, h, x_hat_k, y_hat_k, Z_ref_k,
                                           P_ps, P_vs, P_pu, P_vu, foot_width,
                                           foot_length):
    # pre-allocate memory
    A = zeros((4, 2*N))
    b = zeros((4))

    w = sqrt(g/h)

    P_ps_terminal = P_ps[N-1,:]
    P_vs_terminal = P_vs[N-1,:]

    P_pu_terminal = P_pu[N-1,:]
    P_vu_terminal = P_vu[N-1,:]

    # x-direction
    A[0, 0:N] =  P_pu_terminal + (1/w)*P_vu_terminal
    A[1, 0:N] = -P_pu_terminal - (1/w)*P_vu_terminal
    b[0] =  Z_ref_k[N-1, 0] - 0.5*foot_length - \
            dot([P_ps_terminal+(1/w)*P_vs_terminal], x_hat_k)
    b[1] = -Z_ref_k[N-1, 0] - 0.5*foot_length + \
            dot([P_ps_terminal+(1/w)*P_vs_terminal], x_hat_k)

    # y_direction
    A[2, N:2*N] =  P_pu_terminal + (1/w)*P_vu_terminal
    A[3, N:2*N] = -P_pu_terminal - (1/w)*P_vu_terminal
    b[2] =  Z_ref_k[N-1, 1] - 0.5*foot_width - \
            dot([P_ps_terminal+(1/w)*P_vs_terminal], y_hat_k)
    b[3] = -Z_ref_k[N-1, 1] - 0.5*foot_width + \
            dot([P_ps_terminal+(1/w)*P_vs_terminal], y_hat_k)
    return A, b

# Desctiption:
# -----------
# this function assembles A matrix and b vector encapsulating the com position
# lateral constraint in the y-direction

# Parameters:
# ----------
# N              : preceding horizon length                 (scalar)
# y_hat_k        : [y^_k, y^dot_k].T current CoM state in y (2, numpy.array)
# P_ps, P_pu     : CoM position recursive dynamics matrices (Nx2 numpy.array,
#                  x^_k+1 = P_ps x^_k + P_pu z_k            (NXN numpy.array)
# com_constraint : com position lateral constraint          (scalar)
#                  in the y-direction

# Returns:
# -------
# A : (Nx2N numpy.array)
#      matrix defining the linear terms in the CoM state. However, since the
#      MPC problem decision variables are only the CoP control inputs
#      then the CoM equality terminal constraints can be formulated in terms
#      of the CoP control inputs in y as follows:
#      -P_pu zy_k >= -com_constraint - P_ps y^_k

# b : (N, numpy.array)
#     vector defining the remaining terms

def add_CoM_constraints(N, y_hat_k, P_ps, P_pu, com_constraint):

    # pre-allocate memory
    A = zeros((N, 2*N))
    b = zeros(N)

    A[0:N, N:2*N] = -P_pu
    b[0:N] = -tile(com_constraint, N) + dot(P_ps, y_hat_k)

    return A, b

def add_CoM_constraints_box(N, y_hat_k, P_ps, P_pu, com_constraint):

    # pre-allocate memory
    A = zeros((2*N, 2*N))
    b = zeros(2*N)

    A[0:N, N:2*N] = -P_pu
    A[N:2*N, N:2*N] = P_pu
    b[0:N] = -tile(com_constraint, N) + dot(P_ps, y_hat_k)
    b[N:2*N] = -tile(com_constraint, N) - dot(P_ps, y_hat_k)
    return A, b

def add_CoM_constraints_rfoot(no_constraints, N, y_hat_k, P_ps, P_pu,
                                                                com_constraint):
    # pre-allocate memory
    A = zeros((no_constraints, 2*N))
    b = zeros(no_constraints)

    A[0:no_constraints, N:2*N] = P_pu[N-no_constraints, :]
    b[0:no_constraints] = -tile(com_constraint, no_constraints) - \
                           dot(P_ps[N-no_constraints, :], y_hat_k)
    return A, b

def add_CoM_constraints_lfoot(no_constraints, N, y_hat_k, P_ps, P_pu,
                                                                com_constraint):
    # pre-allocate memory
    A = zeros((no_constraints, 2*N))
    b = zeros(no_constraints)

    A[0:no_constraints, N:2*N] = -P_pu[N-no_constraints, :]
    b[0:no_constraints] = -tile(com_constraint, no_constraints) + \
                           dot(P_ps[N-no_constraints, :], y_hat_k)
    return A, b

#TODO
def add_CoM_constraints_with_automatic_footstep_adjustment(N, x_hat_k, y_hat_k,
                        P_ps, P_vs, P_pu, P_vu, CoM_constraint, m):
    # pre-allocate memory
    A = zeros((N, 2*(N+m)))
    b = zeros(N)
    A[0:N, N+m:(2*N)+m] = -P_pu
    b[0:N] = -tile(CoM_constraint, N) + dot(P_ps, y_hat_k)
    return A, b
# ------------------------------------------------------------------------------
# unit test: A.K.A red pill or blue pill
# ------------------------------------------------------------------------------
if __name__=='__main__':
    import numpy.random as random
    print(' visualize your matrices like a Neo ! '.center(60,'*'))
    n       = 2
    m       = 1
    N       = 16
    delta_t = 0.1
    h       = 0.8
    g       = 9.81
    x_hat_k = random.rand(2)- 0.5
    y_hat_k = random.rand(2)-0.5
    Z_ref   = random.rand(N,n) - 0.5
    foot_length = 0.20
    foot_width  = 0.10
    U_c_k_plus_one = zeros(N)
    U_c_k_plus_one[0:N//2] = 1.0
    U_k_plus_one = zeros((N,2))
    U_k_plus_one[N//2:N,0] = 1.0
    U_k_plus_one[0:N//2,1] = 1.0
    X_fc_k = 0.0
    Y_fc_k = -0.10
    m = 2

# ------------------------------------------------------------------------------
# visualize your ZMP inequality matrices like a Neo:
# ------------------------------------------------------------------------------

    A_ineq, b_ineq = add_ZMP_constraints(N, foot_length, foot_width, Z_ref
                                         , x_hat_k, y_hat_k)

    #A_ineq, b_ineq = add_ZMP_constraints_with_automatic_footstep_adjustment(N,
    #                    foot_length, foot_width, x_hat_k, y_hat_k, U_k_plus_one,
    #                    U_c_k_plus_one, X_fc_k, Y_fc_k, m)

    b_ineq         = reshape(b_ineq, (b_ineq.size,1))
    with nditer(A_ineq, op_flags=['readwrite']) as it:
        for x in it:
            if x[...] != 0:
                x[...] = 1
    plt.figure()
    plt.grid()
    plt.suptitle('Structure of A_ineq matrix')
    plt.imshow(A_ineq, cmap='Greys', extent =[0,A_ineq.shape[1],
                A_ineq.shape[0],0], interpolation = 'nearest')
    plt.figure()
    plt.grid()
    plt.suptitle('Structure of b_ineq vector')
    plt.imshow(b_ineq, cmap='Greys', interpolation = 'nearest',
    extent=[0,b_ineq.shape[1],b_ineq.shape[0],0], aspect=0.25)
    plt.show()
