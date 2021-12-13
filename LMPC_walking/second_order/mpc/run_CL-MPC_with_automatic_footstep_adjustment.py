# headers:
# -------
from numpy import array, absolute, zeros, dot, tile, arange, append, concatenate, repeat
from second_order.cost_function import compute_objective_terms_with_automatic_footstep_adjustment
from second_order.constraints import add_CoM_constraints_with_automatic_footstep_adjustment
from second_order.constraints import add_ZMP_constraints_with_automatic_footstep_adjustment
from second_order.motion_model import compute_recursive_matrices
from second_order.motion_model import compute_recursive_dynamics_step_adjustment
from second_order.cost_function import compute_objective_terms
from second_order.motion_model import discrete_LIP_dynamics
from second_order.rmpc.mrpi.mRPI_set import compute_mRPI
from second_order import plot_utils
from quadprog import solve_qp
from numpy import random, eye

#TODO
# cost weights in the objective function:
# ---------------------------------------
alpha       = 10**(-1)     # CoP error squared cost weight
beta        = 0*10**(-1)     # CoM position error squared cost weight
gamma       = 10**(-4)     # CoM velocity error squared cost weight

# Inverted pendulum parameters:
# ----------------------------
h           = 0.80
g           = 9.81
foot_length = 0.20
foot_width  = 0.14

# MPC Parameters:
# --------------
tau            = 0.005                          # tracking sampling period
delta_t        = 0.1                            # sampling time interval
step_time      = 0.8                            # time needed for every step
no_steps_per_T = int(round(step_time/delta_t))
tracking_time  = int(delta_t/tau)
N              = 16                             # preceding horizon
m              = 2

# walking parameters:
# ------------------
step_length           = 0.20                              # fixed step length in the xz-plane
no_desired_steps      = 2                                 # number of desired walking steps
no_planned_steps      = 2+no_desired_steps                # planning 2 steps ahead (increase if you want to increase the horizon)
desired_walking_time  = no_desired_steps * no_steps_per_T # number of desired walking intervals
planned_walking_time  = no_planned_steps * no_steps_per_T # number of planned walking intervals

# CoM initial state: [x, xdot, x_ddot].T
#                    [y, ydot, y_ddot].T
# --------------------------------------
x_init = array([0.0, 0.0])
y_init = array([-0.10, 0.0])
step_width = 2*absolute(y_init[0])
foot_step_0   = array([0.0, -0.10])    # initial foot step position in x-y


# pre-allocate memory for saving trajectories
# -------------------------------------------
# CoM trajectories
X_cl   = zeros((tracking_time*(desired_walking_time)+1,2))
Y_cl   = zeros((tracking_time*(desired_walking_time)+1,2))
X_cl[0,:] = x_init
Y_cl[0,:] = y_init
X_plus = zeros((tracking_time,2))
Y_plus = zeros((tracking_time,2))

# CoP trajectories
Z_x_cl = zeros((tracking_time*(desired_walking_time)+1))
Z_y_cl = zeros((tracking_time*(desired_walking_time)+1))
Z_x_cl[0] = foot_step_0[0]
Z_y_cl[0] = foot_step_0[1]
ZX_MPC_plus = zeros((tracking_time,2))
ZY_MPC_plus = zeros((tracking_time,2))

# foot placement trajectories
desired_Z_ref = zeros((desired_walking_time+1, 2))
desired_Z_ref[0, :] = foot_step_0

[P_ps, P_vs, P_pu, P_vu] = compute_recursive_matrices(delta_t, g, h, N)
U_c = repeat(array([1.0, 0.0]), N/2, axis=0)
U   = repeat(array([[0.0, 0.0], [1.0, 0.0]]), N/2, axis=0)

u_c_counter = (N/2)-1
u_counter   = N-1

# Initialization
x_0  = x_init
x_cl = x_init
y_0  = y_init
y_cl = y_init
x_fc = x_init[0]
y_fc = y_init[0]
x_f  = x_init[0]
y_f  = y_init[0]
CoM_constraint = 0.05
# ------------------------------------------------------------------------------
#                            MPC loop (every 0.1 sec)
# ------------------------------------------------------------------------------
for i in range(desired_walking_time):
    [Q, p_k] = compute_objective_terms_with_automatic_footstep_adjustment(alpha,
               beta, gamma, no_steps_per_T, N, step_length, step_width, P_ps,
               P_pu, P_vs, P_vu, x_0, y_0, U, U_c, x_fc, y_fc, m, i)

    # making sure that the hessian is positive definite
    Q = Q + (1e-6*eye(Q.shape[0]))

    [A_zmp, b_zmp]  = add_ZMP_constraints_with_automatic_footstep_adjustment(N,
                      foot_length, foot_width, x_0, y_0, U, U_c, x_fc, y_fc, m)

    # call quadprog solver:
    U_OL = solve_qp(Q, -p_k, A_zmp.T, b_zmp)[0]
    #U_OL = solve_qp(Q, -p_k, A.T, b)[0]

    # first elements of the optimal control trajectory
    vx_MPC = U_OL[0]
    vy_MPC = U_OL[N+m]

    # first optimal foot step locations
    xf_MPC = U_OL[N]
    yf_MPC = U_OL[(2*N)+m]

    # evaluate your recursive dynamics given the current initial state:
    [X_OL, Y_OL] = compute_recursive_dynamics_step_adjustment(P_ps, P_vs, P_pu,
                                                    P_vu, N, x_0, y_0, U_OL, m)

    # first element of the optimal state trajectory
    zx_MPC = X_OL[0,:]
    zy_MPC = Y_OL[0,:]
    # --------------------------------------------------------------------------
    #             simulation loop (every tau = 0.005 sec)
    # --------------------------------------------------------------------------
    for j in range(tracking_time):
        A_d, B_d = discrete_LIP_dynamics(tau, g, h) # simulation dynamics

        # save closed-loop CoP trajectories
        Z_x_cl[(i*tracking_time)+j+1] = vx_MPC
        Z_y_cl[(i*tracking_time)+j+1] = vy_MPC

        # update_closed-loop dynamics
        X_plus[j,:] = dot(A_d, x_cl) + dot(B_d, vx_MPC.squeeze())
        Y_plus[j,:] = dot(A_d, y_cl) + dot(B_d, vy_MPC.squeeze())

        x_cl   = X_plus[j,:]
        y_cl   = Y_plus[j,:]

        # save closed-loop CoM trajectories
        X_cl[(i*tracking_time)+j+1,:] = x_cl
        Y_cl[(i*tracking_time)+j+1,:] = y_cl
    # --------------------------------------------------------------------------
    #           update initial conditions for next MPC iteration
    # --------------------------------------------------------------------------
    x_0  = X_plus[X_plus.shape[0]-1,:]
    y_0  = Y_plus[Y_plus.shape[0]-1,:]
    x_cl = X_plus[X_plus.shape[0]-1,:]
    y_cl = Y_plus[Y_plus.shape[0]-1,:]

    # update foot location on the ground only if step duration is finished
    if i != 0 and i % 8 == 0:
         x_fc = xf_MPC
         y_fc = yf_MPC
    desired_Z_ref[i+1, 0] = x_fc
    desired_Z_ref[i+1, 1] = y_fc

    # update foot placement matrices and their counters
    print U_c
    print U
    print x_fc
    print y_fc, '\n'
    if u_c_counter == 0: #reset arrays and counters
        U_c = repeat(array([1.0, 0.0]), N/2, axis=0)
        U   = repeat(array([[0.0, 0.0], [1.0, 0.0]]), N/2, axis=0)
        u_c_counter = N/2
        u_counter = N
    else:
        U_c[u_c_counter] = 0.0
        U[u_c_counter,0] = 1.0
        U[u_counter, 0]  = 0.0
        U[u_counter, 1]  = 1.0
    u_c_counter = u_c_counter - 1
    u_counter   = u_counter - 1

# ------------------------------------------------------------------------------
#                 visualize closed-loop trajectories
# ------------------------------------------------------------------------------
#print Z_x_cl
walking_time = round((desired_walking_time+1)*delta_t, 2)
total_time   = round((tracking_time*(desired_walking_time)+1)*tau, 3)
reference_time_stamp = arange(0, walking_time, delta_t)
tracking_time_stamp  = arange(0, total_time, tau)


min_admissible_cop = desired_Z_ref - tile([foot_length/2, foot_width/2],
                    (desired_walking_time+1,1))
max_admissible_cop = desired_Z_ref + tile([foot_length/2, foot_width/2],
                    (desired_walking_time+1,1))

max_admissible_CoM = tile(0.05, (desired_walking_time+1,1))

# time vs CoP and CoM in x: 'A.K.A run rabbit run !'
# -------------------------------------------------
plot_utils.plot_x(tracking_time_stamp,(tracking_time*desired_walking_time)+1,
reference_time_stamp, desired_walking_time, min_admissible_cop,
max_admissible_cop, Z_x_cl, X_cl, desired_Z_ref)

# time VS CoP and CoM in y: 'A.K.A what goes up must go down'
# ----------------------------------------------------------
plot_utils.plot_y(tracking_time_stamp,(tracking_time*desired_walking_time)+1,
reference_time_stamp, desired_walking_time, min_admissible_cop,
max_admissible_cop, Z_y_cl, Y_cl, desired_Z_ref, max_admissible_CoM)

# plot CoP, CoM in x Vs Cop, CoM in y:
# -----------------------------------
plot_utils.plot_xy(tracking_time_stamp,(tracking_time*desired_walking_time)+1,
reference_time_stamp, desired_walking_time, foot_length, foot_width,
desired_Z_ref, Z_x_cl, Z_y_cl, X_cl, Y_cl)
