# headers:
# -------
from numpy import array, dot, tile, arange, absolute, zeros, sqrt, append, amax
from second_order.stmpc.chance_constraints import add_CoM_chance_constraints_lfoot
from second_order.stmpc.chance_constraints import add_CoM_chance_constraints_rfoot
from second_order.rmpc.robust_constraints import add_CoM_robust_constraints_lfoot
from second_order.rmpc.robust_constraints import add_CoM_robust_constraints_rfoot
from second_order.stmpc.chance_constraints import add_CoM_chance_constraints_box
from second_order.rmpc.robust_constraints import add_CoM_robust_constraints_box
from second_order.rmpc.robust_constraints import compute_CoP_backoff_dead_beat
from second_order.stmpc.truncated_normal import sample_from_truncated_normal
from second_order.stmpc.chance_constraints import add_CoP_chance_constraints
from second_order.rmpc.robust_constraints import add_CoP_robust_constraints
from second_order.rmpc.mrpi.polyhedron import polyhedron, plot_polygon_list
from second_order.reference_trajectories import manual_foot_placement
from second_order.reference_trajectories import create_CoP_trajectory
from second_order.reference_trajectories import create_CoM_trajectory
from second_order.cost_function import compute_objective_terms_box
from second_order.motion_model import compute_recursive_matrices
from second_order.motion_model import compute_recursive_dynamics
from second_order.constraints import add_CoM_constraints_box
from numpy import exp, abs, mean, std, linspace, concatenate
from second_order.motion_model import discrete_LIP_dynamics
from second_order.constraints import add_ZMP_constraints
from second_order.rmpc.mrpi.mRPI_set import compute_mRPI
from second_order import plot_utils
import matplotlib.pyplot as plt
from quadprog import solve_qp
# cost weights in the objective function:
# ---------------------------------------
alpha       = 10**(-1)     # CoP error squared cost weight
beta        = 10**(-4)     # CoM position error squared cost weight
gamma       = 10**(-4)     # CoM velocity error squared cost weight

# Inverted pendulum parameters:
# ----------------------------
h           = 0.88
g           = 9.81
foot_length = 0.15
foot_width  = 0.10
omega       = sqrt(g/h)
delta_t     = 0.1   # MPC sampling period

# dead-beat choice of LIPM pre-stabilizing gains
# ----------------------------------------------
k = exp(omega*delta_t)/((exp(omega*delta_t))-1.0)
k_dead_beat = array([[k, k/omega]])

# MPC Parameters:
# --------------
step_time             = 0.8                         # step period
N                     = 16                          # preceding horizon
no_steps_per_T        = int(step_time/delta_t)

# walking parameters:1
# ------------------
step_length           = 0.20                              # fixed step length in the xz-plane
no_desired_steps      = 8                                 # number of desired walking steps
no_planned_steps      = 2+no_desired_steps                # planning 2 steps ahead (increase if you want to increase the horizon)
desired_walking_time  = no_desired_steps * no_steps_per_T # number of desired walking intervals
planned_walking_time  = no_planned_steps * no_steps_per_T # number of planned walking intervals

# pre-allocate memory
no_sim = 200
total_traj_cost_mpc = zeros(no_sim)
total_traj_cost_50 = zeros(no_sim)
total_traj_cost_60 = zeros(no_sim)
total_traj_cost_70 = zeros(no_sim)
total_traj_cost_80 = zeros(no_sim)
total_traj_cost_90 = zeros(no_sim)
total_traj_cost_99 = zeros(no_sim)
total_traj_cost_999 = zeros(no_sim)
total_traj_cost_9999 = zeros(no_sim)
total_traj_cost_99999 = zeros(no_sim)
total_traj_cost_999999 = zeros(no_sim)
total_traj_cost = zeros(no_sim)
total_traj_cost_rmpc = zeros(no_sim)

constraint_violations_mpc = 0.0
constraint_violations_50 = 0.0
constraint_violations_60 = 0.0
constraint_violations_70 = 0.0
constraint_violations_80 = 0.0
constraint_violations_90 = 0.0
constraint_violations_99 = 0.0
constraint_violations_999 = 0.0
constraint_violations_9999 = 0.0
constraint_violations_99999 = 0.0
constraint_violations_999999 = 0.0
constraint_violations = 0.0
constraint_violations_rmpc = 0.0

# CoM initial state: [x, xdot, x_ddot].T
#                    [y, ydot, y_ddot].T
# --------------------------------------
x_init = array([0.0, 0.0])
y_init = array([-0.085, 0.0])

step_width = 2*absolute(y_init[0])

# discrete dynamics
A_d, B_d = discrete_LIP_dynamics(delta_t, g, h)
[P_ps, P_vs, P_pu, P_vu] = compute_recursive_matrices(delta_t, g, h, N)

# 2D-bounded polyhdron additive disturbance set on the motion model
wc_lb    = -0.0016
wc_ub    =  0.0016
wcdot_lb = -0.016
wcdot_ub =  0.016
P_A = array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
P_b = array([wc_ub, wc_ub, wcdot_ub, wcdot_ub])
P_b = P_b.reshape([P_b.shape[0],1])
W = polyhedron(P_A, P_b)

# control backoff KW (exact due to dead-beat gains)
KW = compute_CoP_backoff_dead_beat(k_dead_beat, W)
#print KW.vertices

# state backoff
epsilon = 10.0**(-6)  # absolute error of the mRPI outer-approximation
B_d =  B_d.reshape(B_d.shape[0], 1)
Omega, Fs_list = compute_mRPI(epsilon, W, A_d, B_d, k_dead_beat)
Omega.compute_Hrep()
#print Omega.vertices, '\n'
max_vertex = amax(Omega.vertices, axis=0)

com_constraint   = 0.04
com_backoff = com_constraint-max_vertex[0]
Sigma_w = array([wc_ub/2.0, wcdot_ub/2.0])
A_k = A_d + dot(B_d, k_dead_beat)
beta_u = 0.50
#-------------------------------------------------------------------------------
#                       monte-carlo simulations
#-------------------------------------------------------------------------------
for k in range(no_sim):
    counter = 1
    # compute CoP reference trajectory:
    # --------------------------------
    foot_step_0   = array([0.0, -0.085])    # initial foot step position in x-y

    desiredFoot_steps  = manual_foot_placement(foot_step_0, step_length,
                                               no_desired_steps)
    desired_Z_ref = create_CoP_trajectory(no_desired_steps, desiredFoot_steps,
                                          desired_walking_time, no_steps_per_T)
    desired_com_ref = create_CoM_trajectory(no_desired_steps, no_steps_per_T,
                                       desired_walking_time, com_constraint)

    # plan the last 2 steps CoP reference in the future to be the same as last step
    planned_Z_ref = zeros((planned_walking_time, 2))
    planned_Z_ref[0:desired_walking_time,:] =  desired_Z_ref
    planned_Z_ref[desired_walking_time:planned_walking_time,:] = \
                                        desired_Z_ref[desired_walking_time-1,:]

    planned_com_ref = zeros(planned_walking_time)
    planned_com_ref[0:desired_walking_time] =  desired_com_ref
    planned_com_ref[desired_walking_time:planned_walking_time] = \
                                         desired_com_ref[desired_walking_time-1]

    # pre-allocate memory
    X_cl_mpc   = zeros(((desired_walking_time)+1,2))
    Y_cl_mpc   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_mpc = zeros(((desired_walking_time)+1))
    Z_y_cl_mpc = zeros(((desired_walking_time)+1))

    X_cl_50   = zeros(((desired_walking_time)+1,2))
    Y_cl_50   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_50 = zeros(((desired_walking_time)+1))
    Z_y_cl_50 = zeros(((desired_walking_time)+1))

    X_cl_60   = zeros(((desired_walking_time)+1,2))
    Y_cl_60   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_60 = zeros(((desired_walking_time)+1))
    Z_y_cl_60 = zeros(((desired_walking_time)+1))

    X_cl_70   = zeros(((desired_walking_time)+1,2))
    Y_cl_70   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_70 = zeros(((desired_walking_time)+1))
    Z_y_cl_70 = zeros(((desired_walking_time)+1))

    X_cl_70   = zeros(((desired_walking_time)+1,2))
    Y_cl_70   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_70 = zeros(((desired_walking_time)+1))
    Z_y_cl_70 = zeros(((desired_walking_time)+1))

    X_cl_80   = zeros(((desired_walking_time)+1,2))
    Y_cl_80   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_80 = zeros(((desired_walking_time)+1))
    Z_y_cl_80 = zeros(((desired_walking_time)+1))

    X_cl_90   = zeros(((desired_walking_time)+1,2))
    Y_cl_90   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_90 = zeros(((desired_walking_time)+1))
    Z_y_cl_90 = zeros(((desired_walking_time)+1))

    X_cl_99   = zeros(((desired_walking_time)+1,2))
    Y_cl_99   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_99 = zeros(((desired_walking_time)+1))
    Z_y_cl_99 = zeros(((desired_walking_time)+1))

    X_cl_999   = zeros(((desired_walking_time)+1,2))
    Y_cl_999   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_999 = zeros(((desired_walking_time)+1))
    Z_y_cl_999 = zeros(((desired_walking_time)+1))

    X_cl_9999   = zeros(((desired_walking_time)+1,2))
    Y_cl_9999   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_9999 = zeros(((desired_walking_time)+1))
    Z_y_cl_9999 = zeros(((desired_walking_time)+1))

    X_cl_99999   = zeros(((desired_walking_time)+1,2))
    Y_cl_99999   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_99999 = zeros(((desired_walking_time)+1))
    Z_y_cl_99999 = zeros(((desired_walking_time)+1))

    X_cl_999999   = zeros(((desired_walking_time)+1,2))
    Y_cl_999999   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_999999 = zeros(((desired_walking_time)+1))
    Z_y_cl_999999 = zeros(((desired_walking_time)+1))

    X_cl   = zeros(((desired_walking_time)+1,2))
    Y_cl   = zeros(((desired_walking_time)+1,2))
    Z_x_cl = zeros(((desired_walking_time)+1))
    Z_y_cl = zeros(((desired_walking_time)+1))

    X_cl_rmpc   = zeros(((desired_walking_time)+1,2))
    Y_cl_rmpc   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_rmpc = zeros(((desired_walking_time)+1))
    Z_y_cl_rmpc = zeros(((desired_walking_time)+1))

    # set first values of the open and closed loop trajectories to be equal
    X_cl_mpc[0,:] = x_init
    Y_cl_mpc[0,:] = y_init
    Z_x_cl_mpc[0] = foot_step_0[0]
    Z_y_cl_mpc[0] = foot_step_0[1]

    X_cl_50[0,:] = x_init
    Y_cl_50[0,:] = y_init
    Z_x_cl_50[0] = foot_step_0[0]
    Z_y_cl_50[0] = foot_step_0[1]

    X_cl_60[0,:] = x_init
    Y_cl_60[0,:] = y_init
    Z_x_cl_60[0] = foot_step_0[0]
    Z_y_cl_60[0] = foot_step_0[1]

    X_cl_70[0,:] = x_init
    Y_cl_70[0,:] = y_init
    Z_x_cl_70[0] = foot_step_0[0]
    Z_y_cl_70[0] = foot_step_0[1]

    X_cl_80[0,:] = x_init
    Y_cl_80[0,:] = y_init
    Z_x_cl_80[0] = foot_step_0[0]
    Z_y_cl_80[0] = foot_step_0[1]

    X_cl_90[0,:] = x_init
    Y_cl_90[0,:] = y_init
    Z_x_cl_90[0] = foot_step_0[0]
    Z_y_cl_90[0] = foot_step_0[1]

    X_cl_99[0,:] = x_init
    Y_cl_99[0,:] = y_init
    Z_x_cl_99[0] = foot_step_0[0]
    Z_y_cl_99[0] = foot_step_0[1]

    X_cl_999[0,:] = x_init
    Y_cl_999[0,:] = y_init
    Z_x_cl_999[0] = foot_step_0[0]
    Z_y_cl_999[0] = foot_step_0[1]

    X_cl_9999[0,:] = x_init
    Y_cl_9999[0,:] = y_init
    Z_x_cl_9999[0] = foot_step_0[0]
    Z_y_cl_9999[0] = foot_step_0[1]

    X_cl_99999[0,:] = x_init
    Y_cl_99999[0,:] = y_init
    Z_x_cl_99999[0] = foot_step_0[0]
    Z_y_cl_99999[0] = foot_step_0[1]

    X_cl_999999[0,:] = x_init
    Y_cl_999999[0,:] = y_init
    Z_x_cl_999999[0] = foot_step_0[0]
    Z_y_cl_999999[0] = foot_step_0[1]

    X_cl[0,:] = x_init
    Y_cl[0,:] = y_init
    Z_x_cl[0] = foot_step_0[0]
    Z_y_cl[0] = foot_step_0[1]

    X_cl_rmpc[0,:] = x_init
    Y_cl_rmpc[0,:] = y_init
    Z_x_cl_rmpc[0] = foot_step_0[0]
    Z_y_cl_rmpc[0] = foot_step_0[1]

    # initialization
    Z_ref_k = planned_Z_ref[0:N,:]
    com_ref_k = planned_com_ref[0:N]

    x_0_mpc = x_init
    x_0_50 = x_init
    x_0_60 = x_init
    x_0_70 = x_init
    x_0_80 = x_init
    x_0_90 = x_init
    x_0_99 = x_init
    x_0_999 = x_init
    x_0_9999 = x_init
    x_0_99999 = x_init
    x_0_999999 = x_init
    x_0    = x_init
    x_0_rmpc    = x_init

    x_cl_mpc = x_init
    x_cl_50 = x_init
    x_cl_60 = x_init
    x_cl_70 = x_init
    x_cl_80 = x_init
    x_cl_90 = x_init
    x_cl_99 = x_init
    x_cl_999 = x_init
    x_cl_9999 = x_init
    x_cl_99999 = x_init
    x_cl_999999 = x_init
    x_cl    = x_init
    x_cl_rmpc  = x_init

    y_0_mpc = y_init
    y_0_50 = y_init
    y_0_60 = y_init
    y_0_70 = y_init
    y_0_80 = y_init
    y_0_90 = y_init
    y_0_99 = y_init
    y_0_999 = y_init
    y_0_9999 = y_init
    y_0_99999 = y_init
    y_0_999999 = y_init
    y_0    = y_init
    y_0_rmpc    = y_init

    y_cl_mpc = y_init
    y_cl_50 = y_init
    y_cl_60 = y_init
    y_cl_70 = y_init
    y_cl_80 = y_init
    y_cl_90 = y_init
    y_cl_99 = y_init
    y_cl_999 = y_init
    y_cl_9999 = y_init
    y_cl_99999 = y_init
    y_cl_999999 = y_init
    y_cl    = y_init
    y_cl_rmpc = y_init
    #---------------------------------------------------------------------------
    #                     MPC loop (every delta_t = 0.1 sec)
    # --------------------------------------------------------------------------
    for i in range(desired_walking_time):
        B_d = B_d.squeeze()
        # apply Gaussian distrubance
        wc = sample_from_truncated_normal(0.0, Sigma_w[0], wc_lb, wc_ub)
        wc_dot = sample_from_truncated_normal(0.0, Sigma_w[1], wcdot_lb,
                                                  wcdot_ub)
        wy_k = array([wc, wc_dot])
        # add noise after two the first two steps
        if i<N-1:
            wy_k = zeros(2)
        #-----------------------------------------------------------------------
        #                                nominal MPC
        # ----------------------------------------------------------------------

        [Q_mpc, p_k_mpc] = compute_objective_terms_box(alpha, beta, gamma,
              step_time, no_steps_per_T, N, step_length, step_width, P_ps, P_pu,
                               P_vs, P_vu, x_0_mpc, y_0_mpc, Z_ref_k, com_ref_k)

        [A_zmp_mpc, b_zmp_mpc] = add_ZMP_constraints(N, foot_length, foot_width,
                                                              Z_ref_k, x_0, y_0)
        if i > N-1 and i < desired_walking_time-N:
            [A_CoM_mpc, b_CoM_mpc] = add_CoM_constraints_box(N, y_0,
                                                     P_ps, P_pu, com_constraint)
            A_mpc = concatenate((A_CoM_mpc, A_zmp_mpc), axis = 0)
            b_mpc = concatenate((b_CoM_mpc, b_zmp_mpc), axis = 0)
        else:
            [A_zmp_mpc, b_zmp_mpc] = add_ZMP_constraints(N, foot_length,
                                                  foot_width, Z_ref_k, x_0, y_0)
            A_mpc = A_zmp_mpc
            b_mpc = b_zmp_mpc

        # solve the open-loop optimization problem
        U_OL_mpc = solve_qp(Q_mpc, -p_k_mpc, A_mpc.T, b_mpc)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_mpc, Y_OL_mpc] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu,
                                                  N, x_0_mpc, y_0_mpc, U_OL_mpc)

        cop_error_mpc = Z_ref_k[:,1]-U_OL_mpc[N:2*N]
        com_error_mpc = com_ref_k - Y_OL_mpc[:,0]
        com_dot_error_mpc = tile(step_width/step_time,N) - Y_OL_mpc[:,1]
        total_traj_cost_mpc[k] = total_traj_cost_mpc[k] \
                + 0.5*alpha*dot(cop_error_mpc.T, cop_error_mpc) \
                + 0.5*beta*dot(com_error_mpc.T, com_error_mpc) \
                + 0.5*gamma*dot(com_dot_error_mpc.T, com_dot_error_mpc)

        # first element of the optimal control trajectory
        vx_mpc = U_OL_mpc[0]
        vy_mpc = U_OL_mpc[N]

        # first element of the optimal state trajectory
        zx_mpc = X_OL_mpc[0,:]
        zy_mpc = Y_OL_mpc[0,:]

        ux_k_mpc = vx_mpc
        uy_k_mpc = vy_mpc

        # save closed-loop CoP trajectories
        Z_x_cl_mpc[i+1]  = ux_k_mpc
        Z_y_cl_mpc[i+1]  = uy_k_mpc

        # simulate closed-loop state
        x_cl_plus_mpc = dot(A_d, x_cl_mpc) + dot(B_d, ux_k_mpc.squeeze())
        y_cl_plus_mpc = dot(A_d, y_cl_mpc) + dot(B_d, uy_k_mpc.squeeze()) + wy_k
        x_cl_mpc   = x_cl_plus_mpc
        y_cl_mpc   = y_cl_plus_mpc

        if i==27 and y_cl_mpc[0] > com_constraint:
            constraint_violations_mpc = constraint_violations_mpc + 1.0

        # save closed-loop CoM trajectories
        X_cl_mpc[i+1,:] = x_cl_plus_mpc
        Y_cl_mpc[i+1,:] = y_cl_plus_mpc

        # reset state with measured state
        x_0_mpc = x_cl_mpc
        y_0_mpc = y_cl_mpc
        #-----------------------------------------------------------------------
        #                                50 %
        # ----------------------------------------------------------------------

        [Q_50, p_k_50] = compute_objective_terms_box(alpha, beta, gamma,
              step_time, no_steps_per_T, N, step_length, step_width, P_ps, P_pu,
                                 P_vs, P_vu, x_0_50, y_0_50, Z_ref_k, com_ref_k)

        [A_zmp_50, b_zmp_50] = add_CoP_chance_constraints(N, foot_length,
                         foot_width, Z_ref_k, k_dead_beat, A_k, Sigma_w, beta_u)
        # add com constraints right foot
        if i > 0 and i <= N/2:
            [A_right_50, b_right_50] = add_CoM_chance_constraints_rfoot(i, N,
                         y_0_50, P_ps, P_pu, Sigma_w, A_k, 0.50, com_constraint)
            A_50 = concatenate((A_right_50, A_zmp_50), axis = 0)
            b_50 = concatenate((b_right_50, b_zmp_50), axis = 0)
            # add com constraints left foot
        elif i > N/2 and i < N:

            [A_left_50, b_left_50] = add_CoM_chance_constraints_lfoot(\
                   counter,N, y_0_50, P_ps, P_pu, Sigma_w, A_k, 0.50,
                                                                 com_constraint)
            A_50 = concatenate((A_right_50, A_left_50, A_zmp_50), axis = 0)
            b_50 = concatenate((b_right_50, b_left_50, b_zmp_50), axis = 0)

        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box_50, b_box_50] = add_CoM_chance_constraints_box(N, y_0_50,
                                 P_ps, P_pu, Sigma_w, A_k, 0.50, com_constraint)
            A_50 = concatenate((A_box_50, A_zmp_50), axis = 0)
            b_50 = concatenate((b_box_50, b_zmp_50), axis = 0)
        else:
            A_50 = A_zmp_50
            b_50 = b_zmp_50

        # solve the open-loop optimization problem
        U_OL_50 = solve_qp(Q_50, -p_k_50, A_50.T, b_50)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_50, Y_OL_50] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu,
                                                     N, x_0_50, y_0_50, U_OL_50)

        cop_error_50 = Z_ref_k[:,1]-U_OL_50[N:2*N]
        com_error_50 = com_ref_k - Y_OL_50[:,0]
        com_dot_error_50 = tile(step_width/step_time,N) - Y_OL_50[:,1]
        total_traj_cost_50[k] = total_traj_cost_50[k] \
                        + 0.5*alpha*dot(cop_error_50.T, cop_error_50) \
                        + 0.5*beta*dot(com_error_50.T, com_error_50) \
                        + 0.5*gamma*dot(com_dot_error_50.T, com_dot_error_50)

        # first element of the optimal control trajectory
        vx_MPC_50 = U_OL_50[0]
        vy_MPC_50 = U_OL_50[N]

        # current error
        e_x_k_50 = x_cl_50 - x_0_50
        e_y_k_50 = y_cl_50 - y_0_50

        # apply control law
        ux_k_50 = vx_MPC_50 + dot(k_dead_beat, e_x_k_50)
        uy_k_50 = vy_MPC_50 + dot(k_dead_beat, e_y_k_50)

        # save closed-loop CoP trajectories
        Z_x_cl_50[i+1]  = ux_k_50
        Z_y_cl_50[i+1]  = uy_k_50

        # simulate closed-loop state
        x_plus_50 = dot(A_d, x_cl_50) + dot(B_d, ux_k_50.squeeze())
        y_plus_50 = dot(A_d, y_cl_50) + dot(B_d, uy_k_50.squeeze()) + wy_k

        x_cl_50   = x_plus_50
        y_cl_50   = y_plus_50

        if i==27 and y_cl_50[0] > com_constraint:
            constraint_violations_50 = constraint_violations_50 + 1.0

        # save closed-loop CoM trajectory
        X_cl_50[i+1,:]  = x_cl_50
        Y_cl_50[i+1,:]  = y_cl_50

        # reset state with measured state
        x_0_50 = x_cl_50
        y_0_50 = y_cl_50

        #-----------------------------------------------------------------------
        #                                60 %
        # ----------------------------------------------------------------------
        [Q_60, p_k_60] = compute_objective_terms_box(alpha, beta, gamma,
                    step_time, no_steps_per_T, N, step_length, step_width, P_ps,
                           P_pu, P_vs, P_vu, x_0_60, y_0_60, Z_ref_k, com_ref_k)

        [A_zmp_60, b_zmp_60] = add_CoP_chance_constraints(N, foot_length,
                         foot_width, Z_ref_k, k_dead_beat, A_k, Sigma_w, beta_u)
        # add com constraints right foot
        if i > 0 and i <= N/2:
            [A_right_60, b_right_60] = add_CoM_chance_constraints_rfoot(i, N,
                         y_0_60, P_ps, P_pu, Sigma_w, A_k, 0.60, com_constraint)
            A_60 = concatenate((A_right_60, A_zmp_60), axis = 0)
            b_60 = concatenate((b_right_60, b_zmp_60), axis = 0)
            # add com constraints left foot
        elif i > N/2 and i < N:

            [A_left_60, b_left_60] = add_CoM_chance_constraints_lfoot(\
                        counter, N, y_0_60, P_ps, P_pu, Sigma_w, A_k,
                                                           0.60, com_constraint)
            A_60 = concatenate((A_right_60, A_left_60, A_zmp_60), axis = 0)
            b_60 = concatenate((b_right_60, b_left_60, b_zmp_60), axis = 0)

        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box_60, b_box_60] = add_CoM_chance_constraints_box(N, y_0_60,
                                 P_ps, P_pu, Sigma_w, A_k, 0.60, com_constraint)
            A_60 = concatenate((A_box_60, A_zmp_60), axis = 0)
            b_60 = concatenate((b_box_60, b_zmp_60), axis = 0)
        else:
            A_60 = A_zmp_60
            b_60 = b_zmp_60
        # solve the open-loop optimization problem
        U_OL_60 = solve_qp(Q_60, -p_k_60, A_60.T, b_60)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_60, Y_OL_60] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu,
                                                     N, x_0_60, y_0_60, U_OL_60)

        cop_error_60 = Z_ref_k[:,1]-U_OL_60[N:2*N]
        com_error_60 = com_ref_k - Y_OL_60[:,0]
        com_dot_error_60 = tile(step_width/step_time,N) - Y_OL_60[:,1]
        total_traj_cost_60[k] = total_traj_cost_60[k] \
                + 0.5*alpha*dot(cop_error_60.T, cop_error_60) \
                + 0.5*beta*dot(com_error_60.T, com_error_60) \
                + 0.5*gamma*dot(com_dot_error_60.T, com_dot_error_60)

        # first element of the optimal control trajectory
        vx_MPC_60 = U_OL_60[0]
        vy_MPC_60 = U_OL_60[N]

        # current error
        e_x_k_60 = x_cl_60 - x_0_60
        e_y_k_60 = y_cl_60 - y_0_60

        # apply control law
        ux_k_60 = vx_MPC_60 + dot(k_dead_beat, e_x_k_60)
        uy_k_60 = vy_MPC_60 + dot(k_dead_beat, e_y_k_60)

        # save closed-loop CoP trajectories
        Z_x_cl_60[i+1]  = ux_k_60
        Z_y_cl_60[i+1]  = uy_k_60

        # simulate closed-loop state
        x_plus_60 = dot(A_d, x_cl_60) + dot(B_d, ux_k_60.squeeze())
        y_plus_60 = dot(A_d, y_cl_60) + dot(B_d, uy_k_60.squeeze()) + wy_k

        x_cl_60   = x_plus_60
        y_cl_60   = y_plus_60

        if i==27 and y_cl_60[0] > com_constraint:
            constraint_violations_60 = constraint_violations_60 + 1.0

        # save closed-loop CoM trajectory
        X_cl_60[i+1,:]  = x_cl_60
        Y_cl_60[i+1,:]  = y_cl_60

        # reset state with measured state
        x_0_60 = x_cl_60
        y_0_60 = y_cl_60
        #-----------------------------------------------------------------------
        #                                70 %
        # ----------------------------------------------------------------------
        [Q_70, p_k_70] = compute_objective_terms_box(alpha, beta, gamma,
                    step_time, no_steps_per_T, N, step_length, step_width, P_ps,
                           P_pu, P_vs, P_vu, x_0_70, y_0_70, Z_ref_k, com_ref_k)

        [A_zmp_70, b_zmp_70] = add_CoP_chance_constraints(N, foot_length,
                         foot_width, Z_ref_k, k_dead_beat, A_k, Sigma_w, beta_u)
        # add com constraints right foot
        if i > 0 and i <= N/2:
            [A_right_70, b_right_70] = add_CoM_chance_constraints_rfoot(i, N,
                         y_0_70, P_ps, P_pu, Sigma_w, A_k, 0.70, com_constraint)
            A_70 = concatenate((A_right_70, A_zmp_70), axis = 0)
            b_70 = concatenate((b_right_70, b_zmp_70), axis = 0)
            # add com constraints left foot
        elif i > N/2 and i < N:

            [A_left_70, b_left_70] = add_CoM_chance_constraints_lfoot(\
                        counter, N, y_0_70, P_ps, P_pu, Sigma_w, A_k,
                                                           0.70, com_constraint)
            A_70 = concatenate((A_right_70, A_left_70, A_zmp_70), axis = 0)
            b_70 = concatenate((b_right_70, b_left_70, b_zmp_70), axis = 0)

        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box_70, b_box_70] = add_CoM_chance_constraints_box(N, y_0_70,
                                 P_ps, P_pu, Sigma_w, A_k, 0.70, com_constraint)
            A_70 = concatenate((A_box_70, A_zmp_70), axis = 0)
            b_70 = concatenate((b_box_70, b_zmp_70), axis = 0)
        else:
            A_70 = A_zmp_70
            b_70 = b_zmp_70
        # solve the open-loop optimization problem
        U_OL_70 = solve_qp(Q_70, -p_k_70, A_70.T, b_70)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_70, Y_OL_70] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu,
                                                     N, x_0_70, y_0_70, U_OL_70)

        cop_error_70 = Z_ref_k[:,1]-U_OL_70[N:2*N]
        com_error_70 = com_ref_k - Y_OL_70[:,0]
        com_dot_error_70 = tile(step_width/step_time,N) - Y_OL_70[:,1]
        total_traj_cost_70[k] = total_traj_cost_70[k] \
                    + 0.5*alpha*dot(cop_error_70.T, cop_error_70) \
                    + 0.5*beta*dot(com_error_70.T, com_error_70) \
                    + 0.5*gamma*dot(com_dot_error_70.T, com_dot_error_70)

        # first element of the optimal control trajectory
        vx_MPC_70 = U_OL_70[0]
        vy_MPC_70 = U_OL_70[N]

        # current error
        e_x_k_70 = x_cl_70 - x_0_70
        e_y_k_70 = y_cl_70 - y_0_70

        # apply control law
        ux_k_70 = vx_MPC_70 + dot(k_dead_beat, e_x_k_70)
        uy_k_70 = vy_MPC_70 + dot(k_dead_beat, e_y_k_70)

        # save closed-loop CoP trajectories
        Z_x_cl_70[i+1]  = ux_k_70
        Z_y_cl_70[i+1]  = uy_k_70

        # simulate closed-loop state
        x_plus_70 = dot(A_d, x_cl_70) + dot(B_d, ux_k_70.squeeze())
        y_plus_70 = dot(A_d, y_cl_70) + dot(B_d, uy_k_70.squeeze()) + wy_k

        x_cl_70   = x_plus_70
        y_cl_70   = y_plus_70

        if i==27 and y_cl_70[0] > com_constraint:
            constraint_violations_70 = constraint_violations_70 + 1.0

        # save closed-loop CoM trajectory
        X_cl_70[i+1,:]  = x_cl_70
        Y_cl_70[i+1,:]  = y_cl_70

        # reset state with measured state
        x_0_70 = x_cl_70
        y_0_70 = y_cl_70
        #-----------------------------------------------------------------------
        #                                80 %
        # ----------------------------------------------------------------------
        [Q_80, p_k_80] = compute_objective_terms_box(alpha, beta, gamma,
                    step_time, no_steps_per_T, N, step_length, step_width, P_ps,
                           P_pu, P_vs, P_vu, x_0_80, y_0_80, Z_ref_k, com_ref_k)

        [A_zmp_80, b_zmp_80] = add_CoP_chance_constraints(N, foot_length,
                        foot_width, Z_ref_k, k_dead_beat, A_k, Sigma_w, beta_u)
        # add com constraints right foot
        if i > 0 and i <= N/2:
            [A_right_80, b_right_80] = add_CoM_chance_constraints_rfoot(i, N,
                         y_0_80, P_ps, P_pu, Sigma_w, A_k, 0.80, com_constraint)
            A_80 = concatenate((A_right_80, A_zmp_80), axis = 0)
            b_80 = concatenate((b_right_80, b_zmp_80), axis = 0)
            # add com constraints left foot
        elif i > N/2 and i < N:

            [A_left_80, b_left_80] = add_CoM_chance_constraints_lfoot(\
                  counter, N, y_0_80, P_ps, P_pu, Sigma_w, A_k, 0.80,
                                                                 com_constraint)
            A_80 = concatenate((A_right_80, A_left_80, A_zmp_80), axis = 0)
            b_80 = concatenate((b_right_80, b_left_80, b_zmp_80), axis = 0)

        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box_80, b_box_80] = add_CoM_chance_constraints_box(N, y_0_80,
                                 P_ps, P_pu, Sigma_w, A_k, 0.80, com_constraint)
            A_80 = concatenate((A_box_80, A_zmp_80), axis = 0)
            b_80 = concatenate((b_box_80, b_zmp_80), axis = 0)
        else:
            A_80 = A_zmp_80
            b_80 = b_zmp_80

        # solve the open-loop optimization problem
        U_OL_80 = solve_qp(Q_80, -p_k_80, A_80.T, b_80)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_80, Y_OL_80] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu,
                                                     N, x_0_80, y_0_80, U_OL_80)

        cop_error_80 = Z_ref_k[:,1]-U_OL_80[N:2*N]
        com_error_80 = com_ref_k - Y_OL_80[:,0]
        com_dot_error_80 = tile(step_width/step_time,N) - Y_OL_80[:,1]
        total_traj_cost_80[k] = total_traj_cost_80[k] \
                + 0.5*alpha*dot(cop_error_80.T, cop_error_80) \
                + 0.5*beta*dot(com_error_80.T, com_error_80) \
                + 0.5*gamma*dot(com_dot_error_80.T, com_dot_error_80)

        # first element of the optimal control trajectory
        vx_MPC_80 = U_OL_80[0]
        vy_MPC_80 = U_OL_80[N]

        # current error
        e_x_k_80 = x_cl_80 - x_0_80
        e_y_k_80 = y_cl_80 - y_0_80

        # apply control law
        ux_k_80 = vx_MPC_80 + dot(k_dead_beat, e_x_k_80)
        uy_k_80 = vy_MPC_80 + dot(k_dead_beat, e_y_k_80)

        # save closed-loop CoP trajectories
        Z_x_cl_80[i+1]  = ux_k_80
        Z_y_cl_80[i+1]  = uy_k_80

        # simulate closed-loop state
        x_plus_80 = dot(A_d, x_cl_80) + dot(B_d, ux_k_80.squeeze())
        y_plus_80 = dot(A_d, y_cl_80) + dot(B_d, uy_k_80.squeeze()) + wy_k

        x_cl_80   = x_plus_80
        y_cl_80   = y_plus_80

        if i==27 and y_cl_80[0] > com_constraint:
            constraint_violations_80 = constraint_violations_80 + 1.0

        # save closed-loop CoM trajectory
        X_cl_80[i+1,:]  = x_cl_80
        Y_cl_80[i+1,:]  = y_cl_80

        # reset state with measured state
        x_0_80 = x_cl_80
        y_0_80 = y_cl_80
        #-----------------------------------------------------------------------
        #                                90 %
        # ----------------------------------------------------------------------
        [Q_90, p_k_90] = compute_objective_terms_box(alpha, beta, gamma,
                    step_time, no_steps_per_T, N, step_length, step_width, P_ps,
                           P_pu, P_vs, P_vu, x_0_90, y_0_90, Z_ref_k, com_ref_k)

        [A_zmp_90, b_zmp_90] = add_CoP_chance_constraints(N,
                             foot_length, foot_width, Z_ref_k, k_dead_beat, A_k,
                                                                Sigma_w, beta_u)
        # add com constraints right foot
        if i > 0 and i <= N/2:
            [A_right_90, b_right_90] = add_CoM_chance_constraints_rfoot(i, N,
                         y_0_90, P_ps, P_pu, Sigma_w, A_k, 0.90, com_constraint)
            A_90 = concatenate((A_right_90, A_zmp_90), axis = 0)
            b_90 = concatenate((b_right_90, b_zmp_90), axis = 0)
            # add com constraints left foot
        elif i > N/2 and i < N:
            [A_left_90, b_left_90] = add_CoM_chance_constraints_lfoot(\
                  counter, N, y_0_90, P_ps, P_pu, Sigma_w, A_k, 0.90,
                                                                 com_constraint)
            A_90 = concatenate((A_right_90, A_left_90, A_zmp_90), axis = 0)
            b_90 = concatenate((b_right_90, b_left_90, b_zmp_90), axis = 0)

        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box_90, b_box_90] = add_CoM_chance_constraints_box(N, y_0_90,
                                 P_ps, P_pu, Sigma_w, A_k, 0.90, com_constraint)
            A_90 = concatenate((A_box_90, A_zmp_90), axis = 0)
            b_90 = concatenate((b_box_90, b_zmp_90), axis = 0)
        else:
            A_90 = A_zmp_90
            b_90 = b_zmp_90

        # solve the open-loop optimization problem
        U_OL_90 = solve_qp(Q_90, -p_k_90, A_90.T, b_90)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_90, Y_OL_90] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu,
                                                     N, x_0_90, y_0_90, U_OL_90)

        cop_error_90 = Z_ref_k[:,1]-U_OL_90[N:2*N]
        com_error_90 = com_ref_k - Y_OL_90[:,0]
        com_dot_error_90 = tile(step_width/step_time,N) - Y_OL_90[:,1]
        total_traj_cost_90[k] = total_traj_cost_90[k] \
                + 0.5*alpha*dot(cop_error_90.T, cop_error_90) \
                + 0.5*beta*dot(com_error_90.T, com_error_90) \
                + 0.5*gamma*dot(com_dot_error_90.T, com_dot_error_90)

        # first element of the optimal control trajectory
        vx_MPC_90 = U_OL_90[0]
        vy_MPC_90 = U_OL_90[N]

        # current error
        e_x_k_90 = x_cl_90 - x_0_90
        e_y_k_90 = y_cl_90 - y_0_90

        # apply control law
        ux_k_90 = vx_MPC_90 + dot(k_dead_beat, e_x_k_90)
        uy_k_90 = vy_MPC_90 + dot(k_dead_beat, e_y_k_90)

        # save closed-loop CoP trajectories
        Z_x_cl_90[i+1]  = ux_k_90
        Z_y_cl_90[i+1]  = uy_k_90

        # simulate closed-loop state
        x_plus_90 = dot(A_d, x_cl_90) + dot(B_d, ux_k_90.squeeze())
        y_plus_90 = dot(A_d, y_cl_90) + dot(B_d, uy_k_90.squeeze()) + wy_k

        x_cl_90   = x_plus_90
        y_cl_90   = y_plus_90

        if i==27 and y_cl_90[0] > com_constraint:
            constraint_violations_90 = constraint_violations_90 + 1.0

        # save closed-loop CoM trajectory
        X_cl_90[i+1,:]  = x_cl_90
        Y_cl_90[i+1,:]  = y_cl_90

        # reset state with measured state
        x_0_90 = x_cl_90
        y_0_90 = y_cl_90
        #-----------------------------------------------------------------------
        #                                99 %
        # ----------------------------------------------------------------------
        [Q_99, p_k_99] = compute_objective_terms_box(alpha, beta, gamma,
                    step_time, no_steps_per_T, N, step_length, step_width, P_ps,
                           P_pu, P_vs, P_vu, x_0_99, y_0_99, Z_ref_k, com_ref_k)

        [A_zmp_99, b_zmp_99] = add_CoP_chance_constraints(N, foot_length,
                         foot_width, Z_ref_k, k_dead_beat, A_k, Sigma_w, beta_u)
        # add com constraints right foot
        if i > 0 and i <= N/2:
            [A_right_99, b_right_99] = add_CoM_chance_constraints_rfoot(i, N,
                         y_0_99, P_ps, P_pu, Sigma_w, A_k, 0.99, com_constraint)
            A_99 = concatenate((A_right_99, A_zmp_99), axis = 0)
            b_99 = concatenate((b_right_99, b_zmp_99), axis = 0)
            # add com constraints left foot
        elif i > N/2 and i < N:
            [A_left_99, b_left_99] = add_CoM_chance_constraints_lfoot(\
                  counter, N, y_0_99, P_ps, P_pu, Sigma_w, A_k, 0.99,
                                                                 com_constraint)
            A_99 = concatenate((A_right_99, A_left_99, A_zmp_99), axis = 0)
            b_99 = concatenate((b_right_99, b_left_99, b_zmp_99), axis = 0)

        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box_99, b_box_99] = add_CoM_chance_constraints_box(N, y_0_99,
                                 P_ps, P_pu, Sigma_w, A_k, 0.99, com_constraint)
            A_99 = concatenate((A_box_99, A_zmp_99), axis = 0)
            b_99 = concatenate((b_box_99, b_zmp_99), axis = 0)
        else:
            A_99 = A_zmp_99
            b_99 = b_zmp_99

        # solve the open-loop optimization problem
        U_OL_99 = solve_qp(Q_99, -p_k_99, A_99.T, b_99)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_99, Y_OL_99] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu,
                                                     N, x_0_99, y_0_99, U_OL_99)

        cop_error_99 = Z_ref_k[:,1]-U_OL_99[N:2*N]
        com_error_99 = com_ref_k - Y_OL_99[:,0]
        com_dot_error_99 = tile(step_width/step_time,N) - Y_OL_99[:,1]
        total_traj_cost_99[k] = total_traj_cost_99[k] \
                    + 0.5*alpha*dot(cop_error_99.T, cop_error_99) \
                    + 0.5*beta*dot(com_error_99.T, com_error_99) \
                    + 0.5*gamma*dot(com_dot_error_99.T, com_dot_error_99)

        # first element of the optimal control trajectory
        vx_MPC_99 = U_OL_99[0]
        vy_MPC_99 = U_OL_99[N]

        # error dynamics
        e_x_k_99 = x_cl_99 - x_0_99
        e_y_k_99 = y_cl_99 - y_0_99

        # apply tube MPC control policy
        ux_k_99 = vx_MPC_99 + dot(k_dead_beat, e_x_k_99)
        uy_k_99 = vy_MPC_99 + dot(k_dead_beat, e_y_k_99)

        # save closed-loop CoP trajectories
        Z_x_cl_99[i+1]  = ux_k_99
        Z_y_cl_99[i+1]  = uy_k_99

        # update_closed-loop dynamics:
        x_plus_99 = dot(A_d, x_cl_99) + dot(B_d, ux_k_99.squeeze())
        y_plus_99 = dot(A_d, y_cl_99) + dot(B_d, uy_k_99.squeeze()) + wy_k

        x_cl_99   = x_plus_99
        y_cl_99   = y_plus_99

        if i==27 and y_cl_99[0] > com_constraint:
            constraint_violations_99 = constraint_violations_99 + 1.0

        # save closed-loop CoM trajectories
        X_cl_99[i+1,:]  = x_cl_99
        Y_cl_99[i+1,:]  = y_cl_99

        # reset initial state with closed-loop state
        x_0_99 = x_cl_99
        y_0_99 = y_cl_99

        #-----------------------------------------------------------------------
        #                                99.9 %
        # ----------------------------------------------------------------------
        [Q_999, p_k_999] = compute_objective_terms_box(alpha, beta, gamma,
                    step_time, no_steps_per_T, N, step_length, step_width, P_ps,
                         P_pu, P_vs, P_vu, x_0_999, y_0_999, Z_ref_k, com_ref_k)

        [A_zmp_999, b_zmp_999] = add_CoP_chance_constraints(N, foot_length,
                         foot_width, Z_ref_k, k_dead_beat, A_k, Sigma_w, beta_u)
        # add com constraints right foot
        if i > 0 and i <= N/2:
            [A_right_999, b_right_999] = add_CoM_chance_constraints_rfoot(i, N,
                       y_0_999, P_ps, P_pu, Sigma_w, A_k, 0.999, com_constraint)
            A_999 = concatenate((A_right_999, A_zmp_999), axis = 0)
            b_999 = concatenate((b_right_999, b_zmp_999), axis = 0)
            # add com constraints left foot
        elif i > N/2 and i < N:
            [A_left_999, b_left_999] = add_CoM_chance_constraints_lfoot(\
                counter, N, y_0_999, P_ps, P_pu, Sigma_w, A_k, 0.999,
                                                                 com_constraint)
            A_999 = concatenate((A_right_999, A_left_999, A_zmp_999), axis = 0)
            b_999 = concatenate((b_right_999, b_left_999, b_zmp_999), axis = 0)

        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box_999, b_box_999] = add_CoM_chance_constraints_box(N, y_0_999,
                                P_ps, P_pu, Sigma_w, A_k, 0.999, com_constraint)
            A_999 = concatenate((A_box_999, A_zmp_999), axis = 0)
            b_999 = concatenate((b_box_999, b_zmp_999), axis = 0)
        else:
            A_999 = A_zmp_999
            b_999 = b_zmp_999

        # solve the open-loop optimization problem
        U_OL_999 = solve_qp(Q_999, -p_k_999, A_999.T, b_999)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_999, Y_OL_999] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu,
                                                  N, x_0_999, y_0_999, U_OL_999)

        cop_error_999 = Z_ref_k[:,1]-U_OL_999[N:2*N]
        com_error_999 = com_ref_k - Y_OL_999[:,0]
        com_dot_error_999 = tile(step_width/step_time,N) - Y_OL_999[:,1]
        total_traj_cost_999[k] = total_traj_cost_999[k] \
                    + 0.5*alpha*dot(cop_error_999.T, cop_error_999) \
                    + 0.5*beta*dot(com_error_999.T, com_error_999) \
                    + 0.5*gamma*dot(com_dot_error_999.T, com_dot_error_999)

        # first element of the optimal control trajectory
        vx_MPC_999 = U_OL_999[0]
        vy_MPC_999 = U_OL_999[N]

        # error dynamics
        e_x_k_999 = x_cl_999 - x_0_999
        e_y_k_999 = y_cl_999 - y_0_999

        # apply tube MPC control policy
        ux_k_999 = vx_MPC_999 + dot(k_dead_beat, e_x_k_999)
        uy_k_999 = vy_MPC_999 + dot(k_dead_beat, e_y_k_999)

        # save closed-loop CoP trajectories
        Z_x_cl_999[i+1]  = ux_k_999
        Z_y_cl_999[i+1]  = uy_k_999

        # update_closed-loop dynamics:
        x_plus_999 = dot(A_d, x_cl_999) + dot(B_d, ux_k_999.squeeze())
        y_plus_999 = dot(A_d, y_cl_999) + dot(B_d, uy_k_999.squeeze()) + wy_k

        x_cl_999   = x_plus_999
        y_cl_999   = y_plus_999

        if i==27 and y_cl_999[0] > com_constraint:
            constraint_violations_999 = constraint_violations_999 + 1.0

        # save closed-loop CoM trajectories
        X_cl_999[i+1,:]  = x_cl_999
        Y_cl_999[i+1,:]  = y_cl_999

        # reset initial state with closed-loop state
        x_0_999 = x_cl_999
        y_0_999 = y_cl_999
        #-----------------------------------------------------------------------
        #                                99.99 %
        # ----------------------------------------------------------------------
        [Q_9999, p_k_9999] = compute_objective_terms_box(alpha, beta, gamma,
                    step_time, no_steps_per_T, N, step_length, step_width, P_ps,
                       P_pu, P_vs, P_vu, x_0_9999, y_0_9999, Z_ref_k, com_ref_k)

        [A_zmp_9999, b_zmp_9999] = add_CoP_chance_constraints(N, foot_length,
                         foot_width, Z_ref_k, k_dead_beat, A_k, Sigma_w, beta_u)
        # add com constraints right foot
        if i > 0 and i <= N/2:
            [A_right_9999, b_right_9999] = add_CoM_chance_constraints_rfoot(i,N,
                     y_0_9999, P_ps, P_pu, Sigma_w, A_k, 0.9999, com_constraint)
            A_9999 = concatenate((A_right_9999, A_zmp_9999), axis = 0)
            b_9999 = concatenate((b_right_9999, b_zmp_9999), axis = 0)
            # add com constraints left foot
        elif i > N/2 and i < N:

            [A_left_9999, b_left_9999] = add_CoM_chance_constraints_lfoot(\
                      counter, N, y_0_9999, P_ps, P_pu, Sigma_w, A_k,
                                                         0.9999, com_constraint)
            A_9999 = concatenate((A_right_9999, A_left_9999, A_zmp_9999),
                                                                       axis = 0)
            b_9999 = concatenate((b_right_9999, b_left_9999, b_zmp_9999),
                                                                       axis = 0)
        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box_9999, b_box_9999] = add_CoM_chance_constraints_box(N,
                     y_0_9999, P_ps, P_pu, Sigma_w, A_k, 0.9999, com_constraint)
            A_9999 = concatenate((A_box_9999, A_zmp_9999), axis = 0)
            b_9999 = concatenate((b_box_9999, b_zmp_9999), axis = 0)
        else:
            A_9999 = A_zmp_9999
            b_9999 = b_zmp_9999

        # solve the open-loop optimization problem
        U_OL_9999 = solve_qp(Q_9999, -p_k_9999, A_9999.T, b_9999)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_9999, Y_OL_9999] = compute_recursive_dynamics(P_ps, P_vs, P_pu,
                                         P_vu, N, x_0_9999, y_0_9999, U_OL_9999)

        cop_error_9999 = Z_ref_k[:,1]-U_OL_9999[N:2*N]
        com_error_9999 = com_ref_k - Y_OL_9999[:,0]
        com_dot_error_9999 = tile(step_width/step_time,N) - Y_OL_9999[:,1]
        total_traj_cost_9999[k] = total_traj_cost_9999[k] \
                    + 0.5*alpha*dot(cop_error_9999.T, cop_error_9999) \
                    + 0.5*beta*dot(com_error_9999.T, com_error_9999) \
                    + 0.5*gamma*dot(com_dot_error_9999.T, com_dot_error_9999)

        # first element of the optimal control trajectory
        vx_MPC_9999 = U_OL_9999[0]
        vy_MPC_9999 = U_OL_9999[N]

        # error dynamics
        e_x_k_9999 = x_cl_9999 - x_0_9999
        e_y_k_9999 = y_cl_9999 - y_0_9999

        # apply tube MPC control policy
        ux_k_9999 = vx_MPC_9999 + dot(k_dead_beat, e_x_k_9999)
        uy_k_9999 = vy_MPC_9999 + dot(k_dead_beat, e_y_k_9999)

        # save closed-loop CoP trajectories
        Z_x_cl_9999[i+1]  = ux_k_9999
        Z_y_cl_9999[i+1]  = uy_k_9999

        # update_closed-loop dynamics:
        x_plus_9999 = dot(A_d, x_cl_9999) + dot(B_d, ux_k_9999.squeeze())
        y_plus_9999 = dot(A_d, y_cl_9999) + dot(B_d, uy_k_9999.squeeze()) + wy_k

        x_cl_9999   = x_plus_9999
        y_cl_9999   = y_plus_9999

        if i==27 and y_cl_9999[0] > com_constraint:
            constraint_violations_9999 = constraint_violations_9999 + 1.0

        # save closed-loop CoM trajectories
        X_cl_9999[i+1,:]  = x_cl_9999
        Y_cl_9999[i+1,:]  = y_cl_9999

        # reset initial state with closed-loop state
        x_0_9999 = x_cl_9999
        y_0_9999 = y_cl_9999

        #-----------------------------------------------------------------------
        #                                99.999 %
        # ----------------------------------------------------------------------

        [Q_99999, p_k_99999] = compute_objective_terms_box(alpha, beta, gamma,
                    step_time, no_steps_per_T, N, step_length, step_width, P_ps,
                     P_pu, P_vs, P_vu, x_0_99999, y_0_99999, Z_ref_k, com_ref_k)

        [A_zmp_99999, b_zmp_99999] = add_CoP_chance_constraints(N, foot_length,
                         foot_width, Z_ref_k, k_dead_beat, A_k, Sigma_w, beta_u)
        # add com constraints right foot
        if i > 0 and i <= N/2:
            [A_right_99999, b_right_99999] = add_CoM_chance_constraints_rfoot(i,
                N, y_0_99999, P_ps, P_pu, Sigma_w, A_k, 0.99999, com_constraint)
            A_99999 = concatenate((A_right_99999, A_zmp_99999), axis = 0)
            b_99999 = concatenate((b_right_99999, b_zmp_99999), axis = 0)
            # add com constraints left foot
        elif i > N/2 and i < N:
            [A_left_99999, b_left_99999] = add_CoM_chance_constraints_lfoot(\
                    counter, N, y_0_99999, P_ps, P_pu, Sigma_w, A_k,
                                                        0.99999, com_constraint)
            A_99999 = concatenate((A_right_99999, A_left_99999, A_zmp_99999),
                                                                       axis = 0)
            b_99999 = concatenate((b_right_99999, b_left_99999, b_zmp_99999),
                                                                       axis = 0)

        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box_99999, b_box_99999] = add_CoM_chance_constraints_box(N, \
                   y_0_99999, P_ps, P_pu, Sigma_w, A_k, 0.99999, com_constraint)
            A_99999 = concatenate((A_box_99999, A_zmp_99999), axis = 0)
            b_99999 = concatenate((b_box_99999, b_zmp_99999), axis = 0)
        else:
            A_99999 = A_zmp_99999
            b_99999 = b_zmp_99999

        # solve the open-loop optimization problem
        U_OL_99999 = solve_qp(Q_99999, -p_k_99999, A_99999.T, b_99999)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_99999, Y_OL_99999] = compute_recursive_dynamics(P_ps, P_vs, P_pu,
                                      P_vu, N, x_0_99999, y_0_99999, U_OL_99999)

        cop_error_99999 = Z_ref_k[:,1]-U_OL_99999[N:2*N]
        com_error_99999 = com_ref_k - Y_OL_99999[:,0]
        com_dot_error_99999 = tile(step_width/step_time,N) - Y_OL_99999[:,1]
        total_traj_cost_99999[k] = total_traj_cost_99999[k] \
                    + 0.5*alpha*dot(cop_error_99999.T, cop_error_99999) \
                    + 0.5*beta*dot(com_error_99999.T, com_error_99999) \
                    + 0.5*gamma*dot(com_dot_error_99999.T, com_dot_error_99999)

        # first element of the optimal control trajectory
        vx_MPC_99999 = U_OL_99999[0]
        vy_MPC_99999 = U_OL_99999[N]

        # error dynamics
        e_x_k_99999 = x_cl_99999 - x_0_99999
        e_y_k_99999 = y_cl_99999 - y_0_99999

        # apply tube MPC control policy
        ux_k_99999 = vx_MPC_99999 + dot(k_dead_beat, e_x_k_99999)
        uy_k_99999 = vy_MPC_99999 + dot(k_dead_beat, e_y_k_99999)

        # save closed-loop CoP trajectories
        Z_x_cl_99999[i+1]  = ux_k_99999
        Z_y_cl_99999[i+1]  = uy_k_99999

        # update_closed-loop dynamics:
        x_plus_99999 = dot(A_d, x_cl_99999) + dot(B_d, ux_k_99999.squeeze())
        y_plus_99999 = dot(A_d, y_cl_99999) + dot(B_d, uy_k_99999.squeeze()) + wy_k

        x_cl_99999   = x_plus_99999
        y_cl_99999   = y_plus_99999

        if i==27 and y_cl_99999[0] > com_constraint:
            constraint_violations_99999 = constraint_violations_99999 + 1.0

        # save closed-loop CoM trajectories
        X_cl_99999[i+1,:]  = x_cl_99999
        Y_cl_99999[i+1,:]  = y_cl_99999

        # reset initial state with closed-loop state
        x_0_99999 = x_cl_99999
        y_0_99999 = y_cl_99999

        #-----------------------------------------------------------------------
        #                                99.9999 %
        # ----------------------------------------------------------------------
        [Q_999999, p_k_999999] = compute_objective_terms_box(alpha, beta, gamma,
              step_time, no_steps_per_T, N, step_length, step_width, P_ps, P_pu,
                         P_vs, P_vu, x_0_999999, y_0_999999, Z_ref_k, com_ref_k)

        [A_zmp_999999, b_zmp_999999] = add_CoP_chance_constraints(N,
                             foot_length, foot_width, Z_ref_k, k_dead_beat, A_k,
                                                                Sigma_w, beta_u)
        # add com constraints right foot
        if i > 0 and i <= N/2:
            [A_right_999999, b_right_999999] = add_CoM_chance_constraints_rfoot(i,
                                        N, y_0_999999, P_ps, P_pu, Sigma_w, A_k,
                                                       0.999999, com_constraint)
            A_999999 = concatenate((A_right_999999, A_zmp_999999), axis = 0)
            b_999999 = concatenate((b_right_999999, b_zmp_999999), axis = 0)
            # add com constraints left foot
        elif i > N/2 and i < N:
            [A_left_999999, b_left_999999] = add_CoM_chance_constraints_lfoot(\
                     counter,N, y_0_999999, P_ps, P_pu, Sigma_w, A_k,
                                                       0.999999, com_constraint)
            A_999999 = concatenate((A_right_999999, A_left_999999, A_zmp_999999),
                                                                       axis = 0)
            b_999999 = concatenate((b_right_999999, b_left_999999, b_zmp_999999),
                                                                       axis = 0)
        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box_999999, b_box_999999] = add_CoM_chance_constraints_box(N, \
                 y_0_999999, P_ps, P_pu, Sigma_w, A_k, 0.999999, com_constraint)
            A_999999 = concatenate((A_box_999999, A_zmp_999999), axis = 0)
            b_999999 = concatenate((b_box_999999, b_zmp_999999), axis = 0)
        else:
            A_999999 = A_zmp_999999
            b_999999 = b_zmp_999999

        # solve the open-loop optimization problem
        U_OL_999999 = solve_qp(Q_999999, -p_k_999999, A_999999.T, b_999999)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_999999, Y_OL_999999] = compute_recursive_dynamics(P_ps, P_vs, P_pu,
                                   P_vu, N, x_0_999999, y_0_999999, U_OL_999999)

        cop_error_999999 = Z_ref_k[:,1]-U_OL_999999[N:2*N]
        com_error_999999 = com_ref_k - Y_OL_999999[:,0]
        com_dot_error_999999 = tile(step_width/step_time,N) - Y_OL_999999[:,1]
        total_traj_cost_999999[k] = total_traj_cost_999999[k] \
                    + 0.5*alpha*dot(cop_error_999999.T, cop_error_999999) \
                    + 0.5*beta*dot(com_error_999999.T, com_error_999999) \
                    + 0.5*gamma*dot(com_dot_error_999999.T, com_dot_error_999999)

        # first element of the optimal control trajectory
        vx_MPC_999999 = U_OL_999999[0]
        vy_MPC_999999 = U_OL_999999[N]

        # error dynamics
        e_x_k_999999 = x_cl_999999 - x_0_999999
        e_y_k_999999 = y_cl_999999 - y_0_999999

        # apply tube MPC control policy
        ux_k_999999 = vx_MPC_999999 + dot(k_dead_beat, e_x_k_999999)
        uy_k_999999 = vy_MPC_999999 + dot(k_dead_beat, e_y_k_999999)

        # save closed-loop CoP trajectories
        Z_x_cl_999999[i+1]  = ux_k_999999
        Z_y_cl_999999[i+1]  = uy_k_999999

        # update_closed-loop dynamics:
        x_plus_999999 = dot(A_d, x_cl_999999) + dot(B_d, ux_k_999999.squeeze())
        y_plus_999999 = dot(A_d, y_cl_999999) + dot(B_d, uy_k_999999.squeeze())\
                                                                          + wy_k
        x_cl_999999   = x_plus_999999
        y_cl_999999   = y_plus_999999

        if i==27 and y_cl_999999[0] > com_constraint:
            constraint_violations_999999 = constraint_violations_999999 + 1.0

        # save closed-loop CoM trajectories
        X_cl_999999[i+1,:]  = x_cl_999999
        Y_cl_999999[i+1,:]  = y_cl_999999

        # reset initial state with closed-loop state
        x_0_999999 = x_cl_999999
        y_0_999999 = y_cl_999999
        #-----------------------------------------------------------------------
        #                                99.99999 %
        # ----------------------------------------------------------------------
        [Q, p_k] = compute_objective_terms_box(alpha, beta, gamma, step_time,
                         no_steps_per_T, N, step_length, step_width, P_ps, P_pu,
                                       P_vs, P_vu, x_0, y_0, Z_ref_k, com_ref_k)

        [A_zmp, b_zmp] = add_CoP_chance_constraints(N, foot_length, foot_width,
                                     Z_ref_k, k_dead_beat, A_k, Sigma_w, beta_u)
        # add com constraints right foot
        if i > 0 and i <= N/2:
            [A_right, b_right] = add_CoM_chance_constraints_rfoot(i, N, y_0,
                     P_ps, P_pu, Sigma_w, A_k, 0.9999999, com_constraint)
            A = concatenate((A_right, A_zmp), axis = 0)
            b = concatenate((b_right, b_zmp), axis = 0)
            # add com constraints left foot
        elif i > N/2 and i < N:
            [A_left, b_left] = add_CoM_chance_constraints_lfoot(\
                counter, N, y_0, P_ps, P_pu, Sigma_w, A_k, 0.9999999,
                                                                 com_constraint)
            A = concatenate((A_right, A_left, A_zmp), axis = 0)
            b = concatenate((b_right, b_left, b_zmp), axis = 0)
        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box, b_box] = add_CoM_chance_constraints_box(N, y_0, P_ps,
                           P_pu, Sigma_w, A_k, 0.9999999, com_constraint)
            A = concatenate((A_box, A_zmp), axis = 0)
            b = concatenate((b_box, b_zmp), axis = 0)
        else:
            A = A_zmp
            b = b_zmp

        # solve the open-loop optimization problem
        U_OL = solve_qp(Q, -p_k, A.T, b)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL, Y_OL] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu, N,
                                                                 x_0, y_0, U_OL)

        cop_error = Z_ref_k[:,1]-U_OL[N:2*N]
        com_error = com_ref_k - Y_OL[:,0]
        com_dot_error = tile(step_width/step_time,N) - Y_OL[:,1]

        total_traj_cost[k] = total_traj_cost[k] \
                    + 0.5*alpha*dot(cop_error.T, cop_error) \
                    + 0.5*beta*dot(com_error.T, com_error) \
                    + 0.5*gamma*dot(com_dot_error.T, com_dot_error)
        # first element of the optimal control trajectory
        vx_MPC = U_OL[0]
        vy_MPC = U_OL[N]

        # error dynamics
        e_x_k = x_cl - x_0
        e_y_k = y_cl - y_0

        # apply tube MPC control policy
        ux_k = vx_MPC + dot(k_dead_beat, e_x_k)
        uy_k = vy_MPC + dot(k_dead_beat, e_y_k)

        # save closed-loop CoP trajectories
        Z_x_cl[i+1]  = ux_k
        Z_y_cl[i+1]  = uy_k

        # update_closed-loop dynamics:
        x_plus = dot(A_d, x_cl) + dot(B_d, ux_k.squeeze())
        y_plus = dot(A_d, y_cl) + dot(B_d, uy_k.squeeze()) + wy_k

        x_cl   = x_plus
        y_cl   = y_plus

        if i==27 and y_cl[0] > com_constraint:
            constraint_violations = constraint_violations + 1.0

        # save closed-loop CoM trajectories
        X_cl[i+1,:]  = x_cl
        Y_cl[i+1,:]  = y_cl

        # reset initial state with closed-loop state
        x_0 = x_cl
        y_0 = y_cl
        #-----------------------------------------------------------------------
        #                               RMPC
        # ----------------------------------------------------------------------
        [Q_rmpc, p_k_rmpc] = compute_objective_terms_box(alpha, beta, gamma,
                          step_time, no_steps_per_T, N, step_length, step_width,
                 P_ps, P_pu, P_vs, P_vu, x_0_rmpc, y_0_rmpc, Z_ref_k, com_ref_k)
        [A_zmp_rmpc, b_zmp_rmpc] = add_CoP_robust_constraints(N,foot_length,
                                                        foot_width, Z_ref_k, KW)
        # add com constraints right foot
        if i > 0 and i <= N/2:
            [A_right_rmpc, b_right_rmpc] = add_CoM_robust_constraints_rfoot(i, N,
                                y_0_rmpc,
                                      P_ps, P_pu, com_constraint, max_vertex[0])
            A_rmpc = concatenate((A_right_rmpc, A_zmp_rmpc), axis = 0)
            b_rmpc = concatenate((b_right_rmpc, b_zmp_rmpc), axis = 0)
        # add com constraints left foot
        elif i > N/2 and i < N:
            [A_left_rmpc, b_left_rmpc] = add_CoM_robust_constraints_lfoot(\
                    counter, N, y_0_rmpc, P_ps, P_pu, com_constraint,
                                                                  max_vertex[0])
            A_rmpc = concatenate((A_right_rmpc, A_left_rmpc, A_zmp_rmpc),
                                                                       axis = 0)
            b_rmpc = concatenate((b_right_rmpc, b_left_rmpc, b_zmp_rmpc),
                                                                       axis = 0)
        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box_rmpc, b_box_rmpc] = add_CoM_robust_constraints_box(N, x_0_rmpc,
                               y_0_rmpc, P_ps, P_vs, P_pu, P_vu, com_constraint,
                                                                  max_vertex[0])
            A_rmpc = concatenate((A_box_rmpc, A_zmp_rmpc), axis = 0)
            b_rmpc = concatenate((b_box_rmpc, b_zmp_rmpc), axis = 0)
        else:
            A_rmpc = A_zmp_rmpc
            b_rmpc = b_zmp_rmpc
        try:
            # solve the open-loop optimization problem
            U_OL_rmpc = solve_qp(Q_rmpc, -p_k_rmpc, A_rmpc.T, b_rmpc)[0]


            [X_OL_rmpc, Y_OL_rmpc] = compute_recursive_dynamics(P_ps, P_vs, P_pu,
                                       P_vu, N, x_cl_rmpc, y_cl_rmpc, U_OL_rmpc)

            cop_error_rmpc = Z_ref_k[:,1]-U_OL_rmpc[N:2*N]
            com_error_rmpc = com_ref_k - Y_OL_rmpc[:,0]
            com_dot_error_rmpc = tile(step_width/step_time,N) - Y_OL_rmpc[:,1]

            total_traj_cost_rmpc[k] = total_traj_cost_rmpc[k]\
            + 0.5*alpha*dot(cop_error_rmpc.T, cop_error_rmpc) \
            + 0.5*beta*dot(com_error_rmpc.T, com_error_rmpc) \
            + 0.5*gamma*dot(com_dot_error_rmpc.T, com_dot_error_rmpc)

            # first element of the optimal control trajectory
            vx_rmpc = U_OL_rmpc[0]
            vy_rmpc = U_OL_rmpc[N]

            # first element of the optimal state trajectory
            zx_rmpc = X_OL_rmpc[0,:]
            zy_rmpc = Y_OL_rmpc[0,:]

            # error dynamics
            e_x_k_rmpc = x_cl_rmpc - x_0_rmpc
            e_y_k_rmpc = y_cl_rmpc - y_0_rmpc

            # apply tube MPC control policy
            ux_k_rmpc = vx_rmpc + dot(k_dead_beat, e_x_k_rmpc)
            uy_k_rmpc = vy_rmpc + dot(k_dead_beat, e_y_k_rmpc)

            # save closed-loop CoP trajectories
            Z_x_cl_rmpc[i+1]  = ux_k_rmpc
            Z_y_cl_rmpc[i+1]  = uy_k_rmpc

            # update_closed-loop dynamics:
            x_plus_rmpc = dot(A_d, x_cl_rmpc) + dot(B_d, ux_k_rmpc.squeeze())
            y_plus_rmpc = dot(A_d, y_cl_rmpc) + dot(B_d, uy_k_rmpc.squeeze()) + wy_k

            x_cl_rmpc   = x_plus_rmpc
            y_cl_rmpc   = y_plus_rmpc

            if i==27 and y_cl_rmpc[0] > com_constraint:
                constraint_violations_rmpc = constraint_violations_rmpc + 1.0
        except:
            # error dynamics
            e_x_k = x_cl_rmpc - X_OL_rmpc[0,:]
            e_y_k = y_cl_rmpc - Y_OL_rmpc[0,:]
            #print("error = ",  e_y_k)

            # apply tube MPC control policy
            ux_k_rmpc = U_OL_rmpc[1] + dot(k_dead_beat, e_x_k)
            uy_k_rmpc = U_OL_rmpc[N+1] + dot(k_dead_beat, e_y_k)

            # reset initial state with the previous
            x_cl_rmpc = X_OL_rmpc[1,:]
            y_cl_rmpc = Y_OL_rmpc[1,:]

        # update initial state
        x_0_rmpc = x_cl_rmpc
        y_0_rmpc = y_cl_rmpc

        # save closed-loop CoM trajectories
        X_cl[i+1,:] = x_cl
        Y_cl[i+1,:] = y_cl

        # save closed-loop CoP trajectories
        Z_x_cl[i+1]  = ux_k
        Z_y_cl[i+1]  = uy_k

        # update CoP reference trajectory for next iteration
        Z_ref_k   = planned_Z_ref[i+1:i+N+1,:]
        com_ref_k = planned_com_ref[i+1:i+N+1]
        counter+= 1
    # --------------------------------------------------------------------------
    #                  visualize your closed-loop trajectories
    # --------------------------------------------------------------------------

    reference_time_stamp = arange(0,
                            round((desired_walking_time+1)*delta_t, 2), delta_t)

    desired_Z_ref = append([desired_Z_ref[0,:]], desired_Z_ref,axis=0)
    min_admissible_cop = desired_Z_ref - tile([foot_length/2, foot_width/2],
                        (desired_walking_time+1,1))
    max_admissible_cop = desired_Z_ref + tile([foot_length/2, foot_width/2],
                        (desired_walking_time+1,1))

    min_admissible_cop_back_off = desired_Z_ref - tile([foot_length/2,
     foot_width/2], (desired_walking_time+1,1)) + tile([foot_length/2, KW.b[0]],
                                                    (desired_walking_time+1,1))
    max_admissible_cop_back_off = desired_Z_ref + tile([foot_length/2,
     foot_width/2], (desired_walking_time+1,1)) - tile([foot_length/2, KW.b[0]],
                                                    (desired_walking_time+1,1))

    CoM_constraint_vector = tile(com_constraint, (desired_walking_time+1,1))
    CoM_back_off_vector = tile(com_backoff, (desired_walking_time+1,1))

violations = array([constraint_violations_50, constraint_violations_60,
                     constraint_violations_70, constraint_violations_80,
                     constraint_violations_90,constraint_violations_99,
                     constraint_violations_999, constraint_violations_9999,
                     constraint_violations_99999, constraint_violations_999999,
                     constraint_violations, constraint_violations_rmpc])

# compute the cost mean and std of the simulated trajectories
avg_cost = array([mean(total_traj_cost_50), mean(total_traj_cost_60),
mean(total_traj_cost_70), mean(total_traj_cost_80), mean(total_traj_cost_90),
mean(total_traj_cost_99), mean(total_traj_cost_999), mean(total_traj_cost_9999),
mean(total_traj_cost_99999), mean(total_traj_cost_999999), mean(total_traj_cost),
mean(total_traj_cost_rmpc)])

cost_error = array([std(total_traj_cost_50), std(total_traj_cost_60),
std(total_traj_cost_70), std(total_traj_cost_80), std(total_traj_cost_90),
std(total_traj_cost_99), std(total_traj_cost_999), std(total_traj_cost_9999),
std(total_traj_cost_99999), std(total_traj_cost_999999), std(total_traj_cost),
std(total_traj_cost_rmpc)])
print('std-dev = ', cost_error, '\n')

print('avg_traj_cost_50 = ', mean(total_traj_cost_50))
print('avg_traj_cost_60 = ', mean(total_traj_cost_60))
print('avg_traj_cost_70 = ', mean(total_traj_cost_70))
print('avg_traj_cost_80 = ', mean(total_traj_cost_80))
print('avg_traj_cost_90 = ', mean(total_traj_cost_90))
print('avg_traj_cost_99 = ', mean(total_traj_cost_99))
print('avg_traj_cost_999 = ', mean(total_traj_cost_999))
print('avg_traj_cost_9999 = ', mean(total_traj_cost_9999))
print('avg_traj_cost_99999 = ', mean(total_traj_cost_99999))
print('avg_traj_cost_999999 = ', mean(total_traj_cost_999999))
print('avg_traj_cost = ', mean(total_traj_cost))
print('avg_traj_cost_rmpc = ', mean(total_traj_cost_rmpc), '\n')

# using the variable axs for multiple Axes
fig, ax2 = plt.subplots(1, 1)
ax2 = plt.gca()
plt.rc('text', usetex = True)
plt.rc('font', family ='serif')
label = ['$\beta_{x_j} = 50\%$/mpc', '$\beta_{x_j} = 40\%$',
'$\beta_{x_j} = 30\%$', '$\beta_{x_j} = 20\%$', '$\beta_{x_j} = 10\%$',
'$\beta_{x_j} = 1\%$', '$\beta_{x_j} = 0.1\%$', '$\beta_{x_j} = 0.01\%$',
'$\beta_{x_j} = 0.001\%$', '$\beta_{x_j} = 0.0001\%$',
'$\beta_{x_j} = 0.00001\%$','RMPC']

plt_50 = ax2.scatter(violations[0],
mean(total_traj_cost_50)/mean(total_traj_cost_50), marker='o', s=50,
c='limegreen', label = r"$\beta_{x_j} = 50\%$/MPC, $\, \sigma_{J_N} = \,$"+str(round(std(total_traj_cost_50),6)))
plt_40 = ax2.scatter(violations[1], mean(total_traj_cost_60)/mean(total_traj_cost_50), marker='o',s=50,  c='turquoise', label = r'$\beta_{x_j} = 40\%, \, \sigma_{J_N} =  \,$'+str(round(std(total_traj_cost_60),6)))
plt_30 = ax2.scatter(violations[2], mean(total_traj_cost_70)/mean(total_traj_cost_50), marker='o', s=50, c='orange', label = r'$\beta_{x_j} = 30\%, \, \sigma_{J_N} =  \,$'+str(round(std(total_traj_cost_70),6)))
plt_20 = ax2.scatter(violations[3], mean(total_traj_cost_80)/mean(total_traj_cost_50), marker='o', s=50, c='gold', label = r'$\beta_{x_j} = 20\%, \, \sigma_{J_N} =  \,$'+str(round(std(total_traj_cost_80),6)))
plt_10 = ax2.scatter(violations[4], mean(total_traj_cost_90)/mean(total_traj_cost_50), marker='o', s=50, c='olive', label = r'$\beta_{x_j} = 10\%, \, \sigma_{J_N} =  \,$'+str(round(std(total_traj_cost_90),6)))
plt_1 = ax2.scatter(violations[5], mean(total_traj_cost_99)/mean(total_traj_cost_50), marker='o', s=50, c='purple', label = r'$\beta_{x_j} = 1\%, \, \sigma_{J_N} =  \,$'+str(round(std(total_traj_cost_99),6)))
plt_01 = ax2.scatter(violations[6], mean(total_traj_cost_999)/mean(total_traj_cost_50), marker='o', s=50, c='coral', label = r'$\beta_{x_j} = 0.1\%, \, \sigma_{J_N} =  \,$'+str(round(std(total_traj_cost_999),6)))
plt_001 = ax2.scatter(violations[7], mean(total_traj_cost_9999)/mean(total_traj_cost_50), marker='o', s=50,  c='teal', label = r'$\beta_{x_j} = 0.01\%, \, \sigma_{J_N} =  \,$'+str(round(std(total_traj_cost_9999),6)))
plt_0001 = ax2.scatter(violations[8], mean(total_traj_cost_99999)/mean(total_traj_cost_50), marker='o', s=50,  c='cyan', label = r'$\beta_{x_j} = 0.001\%, \, \sigma_{J_N} =  \,$'+str(round(std(total_traj_cost_99999),6)))
plt_00001 = ax2.scatter(violations[9], mean(total_traj_cost_999999)/mean(total_traj_cost_50), marker='o', s=50,  c='magenta', label = r'$\beta_{x_j} = 0.0001\%, \, \sigma_{J_N} =  \,$'+str(round(std(total_traj_cost_999999),6)))
plt_bas = ax2.scatter(violations[10], mean(total_traj_cost)/mean(total_traj_cost_50), marker='o',s=50,  c='blue', label = r'$\beta_{x_j} = 0.00001\%, \, \sigma_{J_N} =  \,$'+str(round(std(total_traj_cost),6)))
plt_rmpc = ax2.scatter(violations[11], mean(total_traj_cost_rmpc)/mean(total_traj_cost_50), marker='o', s=50, c='red', label = r'RMPC, $\, \sigma_{J_N} =  \,$'+str(round(std(total_traj_cost_rmpc),5)))
s = linspace(0, violations[0])
ax2.plot(s , tile(1, s.shape[0]) , c='black',  linestyle = '-.')
plt.rc('text', usetex = True)
plt.rc('font', family ='serif')
plt.ylabel('average $J_N$ / average $J_N$ nominal MPC',fontsize=19)
plt.legend(fontsize=15)
plt.xlabel('number of CoM position constraint violations at $t=2.7$ s',fontsize=19)
ax2.set_yscale('log')

print('constraint_violations_50 = ', constraint_violations_50)
print('constraint_violations_60 = ',  constraint_violations_60)
print('constraint_violations_70 = ',  constraint_violations_70)
print('constraint_violations_80 = ',  constraint_violations_80)
print('constraint_violations_90 = ',  constraint_violations_90)
print('constraint_violations_99 = ',  constraint_violations_99)
print('constraint_violations_999 = ',  constraint_violations_999)
print('constraint_violations_9999 = ',  constraint_violations_9999)
print('constraint_violations_99999 = ',  constraint_violations_99999)
print('constraint_violations_999999 = ',  constraint_violations_999999)
print('constraint_violations = ',  constraint_violations)
print('constraint_violations_rmpc = ',  constraint_violations_rmpc ,'\n')
plt.show()
