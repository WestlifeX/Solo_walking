from numpy import array, reshape, random, eye, tile, zeros, tile, nditer, ones
from numpy import arange, dot, vstack
from second_order.rmpc.mrpi.polyhedron import polyhedron, plot_polygon_list
from second_order.rmpc.mrpi.mRPI_set import compute_mRPI
from second_order.reference_trajectories import manual_foot_placement
from second_order.reference_trajectories import create_CoP_trajectory
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import sqrt, cosh, sinh
import matplotlib.pyplot as plt
from PIL import Image

def compute_CoP_backoff(Omega, K):
    # compute CoP back-off (K * \Omega):
    KOmega = OMega.affineMap(Omega, K)
    # compute H-rep
    KOmega.compute_Hrep()
    return KOmega

def compute_CoP_backoff_dead_beat(k_dead_beat, GW):
    KGW = GW.affineMap(k_dead_beat)
    KGW.compute_Hrep()
    # convert from cdd to standard affine inequalities formate Ax<=b
    for j in range(KGW.A.shape[0]):
        if KGW.A[j] < 0.0:
            KGW.b[j] = KGW.b[j]/-KGW.A[j]
            KGW.A[j] = -1.0
        else:
            KGW.b[j] = KGW.b[j]/KGW.A[j]
            KGW.A[j] = 1.0
    return KGW

def add_CoP_robust_constraints(N, foot_length, foot_width, Z_ref_k, KGW):

    # pre-allocate memory
    A = zeros((4*N, 2*N))
    b = zeros((4*N))

    # x-direction
    A[0:N  , 0:N]   = eye(N)
    A[N:2*N, 0:N]   = -eye(N)

    # y-direction
    A[2*N:3*N, N:2*N] = eye(N)
    A[3*N:4*N, N:2*N] = -eye(N)

    # back-offs (y-direction only)
    uy_backoff_bN = tile(KGW.b[0],N)

    foot_length_N = tile(foot_length,(N))
    foot_width_N  = tile(foot_width,(N))

    b[0:N]     =  Z_ref_k[:,0] - (0.5*foot_length_N)
    b[N:2*N]   = -Z_ref_k[:,0] - (0.5*foot_length_N)
    b[2*N:3*N] =  Z_ref_k[:,1] - (0.5*foot_width_N) + uy_backoff_bN
    b[3*N:4*N] = -Z_ref_k[:,1] - (0.5*foot_width_N) + uy_backoff_bN
    return A, b
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
# TODO: UNIT TEST
def add_capturability_robust_terminal_constraints(N, g, h, x_hat_k, y_hat_k,
                                           Z_ref_k, P_ps, P_vs, P_pu, P_vu,
                                           foot_width, foot_length, KGW):
    # pre-allocate memory
    A = zeros((4, 2*N))
    b = zeros((4))

    w = sqrt(g/h)

    P_ps_terminal = P_ps[N-1,:]
    P_vs_terminal = P_vs[N-1,:]

    P_pu_terminal = P_pu[N-1,:]
    P_vu_terminal = P_vu[N-1,:]

    # back-offs (y-direction only)
    uy_backoff_b = KGW.b[0]

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
    b[2] =  Z_ref_k[N-1, 1] - 0.5*foot_width + uy_backoff_b - \
            dot([P_ps_terminal+(1/w)*P_vs_terminal], y_hat_k)
    b[3] = -Z_ref_k[N-1, 1] - 0.5*foot_width + uy_backoff_b + \
            dot([P_ps_terminal+(1/w)*P_vs_terminal], y_hat_k)
    return A, b

def add_CoM_robust_constraints(N, x_hat_k, y_hat_k, P_ps, P_vs, P_pu, P_vu,
                               CoM_back_off):
    # pre-allocate memory
    A = zeros((N, 2*N))
    b = zeros(N)

    A[0:N, N:2*N] = -P_pu
    b[0:N] = -tile(CoM_back_off, N) + dot(P_ps, y_hat_k)
    return A, b

def add_CoM_robust_constraints_box(N, x_hat_k, y_hat_k, P_ps, P_vs, P_pu, P_vu,
                                        com_constraint, com_back_off_magnitude):
    # pre-allocate memory
    A = zeros((2*N, 2*N))
    b = zeros(2*N)

    A[0:N, N:2*N]   = -P_pu
    A[N:2*N, N:2*N] = P_pu
    b[0:N] = -tile(com_constraint, N) + tile(com_back_off_magnitude, N) \
                                                            + dot(P_ps, y_hat_k)
    b[N:2*N] = -tile(com_constraint, N) + tile(com_back_off_magnitude, N) \
                                                            - dot(P_ps, y_hat_k)
    return A, b

def add_CoM_robust_constraints_rfoot(no_constraints, N, y_hat_k, P_ps, P_pu,
                                        com_constraint, com_back_off_magnitude):
    # pre-allocate memory
    A = zeros((no_constraints, 2*N))
    b = zeros(no_constraints)

    A[0:no_constraints, N:2*N] = P_pu[N-no_constraints:N, :]
    b[0:no_constraints] = -tile(com_constraint, no_constraints) + \
                           tile(com_back_off_magnitude, no_constraints) - \
                           dot(P_ps[N-no_constraints:N, :], y_hat_k)
    return A, b

def add_CoM_robust_constraints_lfoot(no_constraints, N, y_hat_k, P_ps, P_pu,
                                        com_constraint, com_back_off_magnitude):
    # pre-allocate memory
    A = zeros((no_constraints, 2*N))
    b = zeros(no_constraints)

    A[0:no_constraints, N:2*N] = -P_pu[N-no_constraints:N, :]
    b[0:no_constraints] = -tile(com_constraint, no_constraints) + \
                           tile(com_back_off_magnitude, no_constraints) + \
                           dot(P_ps[N-no_constraints:N, :], y_hat_k)
    return A, b

# TODO: UNIT TEST
def add_projected_x_0_robust_constraint(N, y_hat_k, P_ps, P_pu):
    A = zeros((2, 2*N))
    b = zeros(2)
    A[0, N:2*N] = -P_pu[0,:]
    A[0, N:2*N] = P_pu[0,:]
    b[0] = -0.05 + dot(P_ps[0,:], y_hat_k) - y_hat_k[0]
    b[0] = -0.05 - dot(P_ps[0,:], y_hat_k) + y_hat_k[0]
    return A, b
# ------------------------------------------------------------------------------
# unit test: A.K.A red pill or blue pill
# ------------------------------------------------------------------------------
if __name__=='__main__':
    import numpy.random as random
    from numpy import amax
    print(' visualize your matrices like a Neo ! '.center(60,'*'))

    # Inverted pendulum parameters:
    h     = 0.80
    g     = 9.81
    tau   = 0.005
    omega = 3.5

    # Dynamics:
    A_d = array([[cosh(omega*tau)  , (1/omega)*sinh(omega*tau)],
                    [omega*sinh(omega*tau), cosh(omega*tau)]])

    B_d = array([1 - cosh(omega*tau), -omega*sinh(omega*tau)])
    B_d = B_d.reshape([B_d.shape[0],1])

    N  = 16
    Z_ref_k = random.rand(N,2) - 0.5
    x_hat_k = array([0.0, 0.0])
    y_hat_k = array([-0.09, 0.0])
    P_ps = ones((N,2))
    P_vs = ones((N,2))
    P_pu =  eye(N)
    P_vu =  eye(N)

    # compute CoP reference trajectory:
    # --------------------------------
    delta_t           = 0.1                 # sampling time interval
    step_time         = 0.8                 # time needed for every step
    no_steps_per_T    = int(round(step_time/delta_t))
    no_desired_steps  = 3                   # number of desired walking steps
    desired_walking_time  = no_desired_steps * no_steps_per_T
    foot_step_0       = array([0.0, -0.09]) # initial foot step position in x-y
    foot_length       = 0.20
    foot_width        = 0.14
    step_length       = 0.21                # fixed step length in the xz-plane

    desiredFoot_steps = manual_foot_placement(foot_step_0, step_length,
                                               no_desired_steps)
    desired_Z_ref = create_CoP_trajectory(no_desired_steps, desiredFoot_steps,
                                          desired_walking_time, no_steps_per_T)


    # dead-beat choice of pre-stabilizing gains
    k = 57.6
    k_dead_beat = array([[k, k/omega]])

    # algorithm initialization
    alpha      = 0.0
    logicalVar = 1.0
    epsilon    = 10.0**(-6)    # error threshold

    # 1D-polyhdron disturbance set on the control input
    P_A = array([[1.0], [-1.0]])
    P_b = array([0.05, 0.05])
    P_b = P_b.reshape([P_b.shape[0],1])
    W  = polyhedron(P_A, P_b)
    BW = W.affineMap(B_d)

    # state back-off \Omega
    Omega, Fs_list = compute_mRPI(epsilon, BW, A_d, B_d, k_dead_beat)
    Omega.compute_Hrep()
    CoM_constraint = 0.05
    #print Omega.vertices, '\n'
    max_vertex = amax(Omega.vertices, axis=0)
    CoM_constraint_vector = tile(CoM_constraint, (desired_walking_time+1,1))
    CoM_backoff = CoM_constraint - max_vertex[0]

    #print 'Omega_A = ', Omega.A
    #print 'Omega_b = ', Omega.b
    #Omega.minVrep()
    #print 'Omega  vertices = ', Omega.vertices

    # control back-off KBW (exact)
    KBW = compute_CoP_backoff_dead_beat(k_dead_beat, BW)
    for j in range(KBW.A.shape[0]):
        if KBW.A[j] < 0.0:
            KBW.b[j] = KBW.b[j]/-KBW.A[j]
            KBW.A[j] = -1.0
        else:
            KBW.b[j] = KBW.b[j]/KBW.A[j]
            KBW.A[j] = 1.0
    print('cop_back_off_A = ', KBW.A)
    print('cop_back_off_b = ', KBW.b)

    A,b = add_CoP_robust_constraints(N, foot_length, foot_width, Z_ref_k, KBW)

    A, b = add_CoM_robust_constraints(N, x_hat_k, y_hat_k, P_ps, P_vs, P_pu, P_vu,
                                   CoM_backoff)
    print('A = ', A, '\n')
    print('b = ', b, '\n')

# ------------------------------------------------------------------------------
# visualize your constraints matrices like a Neo:
# ------------------------------------------------------------------------------
    b  = reshape(b, (b.size,1))
    with nditer(A, op_flags=['readwrite']) as it:
        for x in it:
            if x[...] != 0:
                x[...] = 1
    plt.figure()
    plt.grid()
    plt.suptitle('Structure of A_ineq matrix')
    plt.imshow(A, cmap='Greys', extent =[0,A.shape[1],
                A.shape[0],0], interpolation = 'nearest')
    plt.figure()
    plt.grid()
    plt.suptitle('Structure of b_ineq vector')
    plt.imshow(b, cmap='Greys', interpolation = 'nearest',
    extent=[0,b.shape[1],b.shape[0],0], aspect=0.25)
    plt.show()
