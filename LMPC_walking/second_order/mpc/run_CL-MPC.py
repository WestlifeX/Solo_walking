# headers:
# -------
from numpy import array, absolute, zeros, dot, tile, arange, append, concatenate
from second_order.constraints import add_capturability_terminal_constraints
from second_order.motion_model import compute_recursive_disturbed_dynamics
from second_order.reference_trajectories import manual_foot_placement
from second_order.reference_trajectories import create_CoP_trajectory
from second_order.motion_model import compute_recursive_matrices
from second_order.motion_model import compute_recursive_dynamics
from second_order.cost_function import compute_objective_terms
from second_order.motion_model import discrete_LIP_dynamics
from second_order.constraints import add_ZMP_constraints
from second_order.constraints import add_CoM_constraints
from second_order import plot_utils
from quadprog import solve_qp
from numpy import random
import matplotlib.pyplot as plt

# cost weights in the objective function:
# ---------------------------------------
alpha       = 10**(-1)     # CoP error squared cost weight
beta        = 10**(-4)     # CoM position error squared cost weight
gamma       = 10**(-4)     # CoM velocity error squared cost weight

# Inverted pendulum parameters:
# ----------------------------
h           = 0.80
g           = 9.81
foot_length = 0.20
foot_width  = 0.14

# MPC Parameters:
# --------------
delta_t        = 0.1                            # sampling time interval
step_time      = 0.8                            # time needed for every step
no_steps_per_T = int(round(step_time/delta_t))
N              = 16                             # preceding horizon

# walking parameters:
# ------------------
step_length           = 0.25                              # fixed step length in the xz-plane
no_desired_steps      = 2                                 # number of desired walking steps
no_planned_steps      = 2+no_desired_steps                # planning 2 steps ahead (increase if you want to increase the horizon)
desired_walking_time  = no_desired_steps * no_steps_per_T # number of desired walking intervals
planned_walking_time  = no_planned_steps * no_steps_per_T # number of planned walking intervals

# 2D-bounded polyhdron additive disturbance on the dynamics
wc_lb    = -0.002
wc_ub    =  0.002
wcdot_lb = -0.02
wcdot_ub =  0.02
total_simulations = 100 # increase if you want to run more than one simulation in parallel
com_constraint = 0.05
CoM_constraint_vector = tile(com_constraint, (desired_walking_time+1,1))
plot_legend = True

for sim in range(total_simulations):

    # CoM initial state: [x, xdot, x_ddot].T
    #                    [y, ydot, y_ddot].T
    # --------------------------------------
    x_init = array([0.0, 0.0])
    y_init = array([-0.10, 0.0])

    step_width = 2*absolute(y_init[0])

    # compute CoP reference trajectory:
    # --------------------------------
    foot_step_0   = array([0.0, -0.10])    # initial foot step position in x-y

    desiredFoot_steps  = manual_foot_placement(foot_step_0,
                                                step_length, no_desired_steps)
    desired_Z_ref = create_CoP_trajectory(no_desired_steps, desiredFoot_steps,
                                          desired_walking_time, no_steps_per_T)

    # plan the last 2 steps CoP reference in the future to be the same as last step
    planned_Z_ref = zeros((planned_walking_time, 2))
    planned_Z_ref[0:desired_walking_time,:] =  desired_Z_ref
    planned_Z_ref[desired_walking_time:planned_walking_time,:] = \
                                        desired_Z_ref[desired_walking_time-1,:]

    # pre-allocate memory
    X_cl   = zeros(((desired_walking_time)+1,2))
    Y_cl   = zeros(((desired_walking_time)+1,2))
    Z_x_cl = zeros(((desired_walking_time)+1))
    Z_y_cl = zeros(((desired_walking_time)+1))

    # set first values of the open and closed loop trajectories to be equal
    X_cl[0,:] = x_init
    Y_cl[0,:] = y_init
    Z_x_cl[0] = foot_step_0[0]
    Z_y_cl[0] = foot_step_0[1]

    [P_ps, P_vs, P_pu, P_vu] = compute_recursive_matrices(delta_t, g, h, N)
    A_d, B_d = discrete_LIP_dynamics(delta_t, g, h)

    # Initialization
    Z_ref_k = planned_Z_ref[0:N,:]
    x_0     = x_init
    x_cl    = x_init
    y_0     = y_init
    y_cl    = y_init
    Sigma_w = array([wc_ub/2.0, wcdot_ub/2.0])
    # --------------------------------------------------------------------------
    #                            MPC loop (every 0.1 sec)
    # --------------------------------------------------------------------------
    for i in range(desired_walking_time):

        [Q, p_k] = compute_objective_terms(alpha, beta, gamma, step_time,
                                     no_steps_per_T, N, step_length, step_width,
                                     P_ps, P_pu, P_vs, P_vu, x_0, y_0, Z_ref_k)
        [A_zmp, b_zmp] = add_ZMP_constraints(N, foot_length, foot_width, Z_ref_k,
                                             x_0, y_0)
        [A_com , b_com] = add_CoM_constraints(N, y_0, P_ps, P_pu, com_constraint)
        #[A_capture, b_capture] = add_capturability_terminal_constraints(N, g, h,
        #                                   x_0, y_0, Z_ref_k, P_ps, P_vs,
        #                                   P_pu, P_vu, foot_width, foot_length)

        A = concatenate((A_com, A_zmp), axis = 0)
        b = concatenate((b_com, b_zmp), axis = 0)

        # call quadprog solver:
        #U_OL = solve_qp(Q, -p_k, A_zmp.T, b_zmp)[0]
        U_OL = solve_qp(Q, -p_k, A.T, b)[0]

        # first element of the optimal control trajectory
        vx_MPC = U_OL[0]
        vy_MPC = U_OL[N]

        # evaluate your recursive dynamics over the current horizon:
        [X_OL, Y_OL] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu, N, x_0,
                                                y_0, U_OL)

        # first element of the optimal state trajectory
        zx_MPC = X_OL[0,:]
        zy_MPC = Y_OL[0,:]
        # ----------------------------------------------------------------------
        #                   simulation (adding disturbances)
        # ----------------------------------------------------------------------
        # sample white gaussian from the bounded disturbance set W
        # set flag to True if you want to add truncated gaussian disturbance
        flag = True
        wy_k = zeros(2)
        while flag == True:
            wc     = random.normal(0, Sigma_w[0])
            wc_dot = random.normal(0, Sigma_w[1])
            if  wc >= wc_lb    and wc <= wc_ub and \
            wc_dot >= wcdot_lb and wc_dot <= wcdot_ub:
                flag = False
            wy_k = array([wc, wc_dot])

        # uncomment if you want to add worst-case distrubance
        # worst case disturbance:
        #wy_k = array([wc_ub, wcdot_ub])

        # save closed-loop CoP trajectories
        Z_x_cl[i+1]  = vx_MPC
        Z_y_cl[i+1]  = vy_MPC

        # update open-loop dynamics
        x_0_plus = dot(A_d, x_0) + dot(B_d, vx_MPC)
        y_0_plus = dot(A_d, y_0) + dot(B_d, vy_MPC)

        # update_closed-loop dynamics:
        x_cl_plus = dot(A_d, x_cl) + dot(B_d, vx_MPC.squeeze())
        y_cl_plus = dot(A_d, y_cl) + dot(B_d, vy_MPC.squeeze()) + wy_k
        x_cl   = x_cl_plus
        y_cl   = y_cl_plus

        # save closed-loop CoM trajectories
        X_cl[i+1,:]  = x_cl
        Y_cl[i+1,:]  = y_cl

        # ----------------------------------------------------------------------
        #           update initial conditions for next MPC iteration
        # ----------------------------------------------------------------------
        x_0  = x_cl
        y_0  = y_cl

        # update CoP reference trajectory for next iteration
        Z_ref_k   = planned_Z_ref[i+1:i+N+1,:]

    # --------------------------------------------------------------------------
    #                 visualize closed-loop trajectories
    # --------------------------------------------------------------------------
    reference_time_stamp = arange(0, round((desired_walking_time+1)*delta_t, 2),
                                                                        delta_t)
    desired_Z_ref = append([desired_Z_ref[0,:]], desired_Z_ref,axis=0)
    min_admissible_cop = desired_Z_ref - tile([foot_length/2, foot_width/2],
                        (desired_walking_time+1,1))
    max_admissible_cop = desired_Z_ref + tile([foot_length/2, foot_width/2],
                        (desired_walking_time+1,1))

    if plot_legend:
        plot_utils.plot_y(plot_legend, reference_time_stamp, desired_walking_time,
        min_admissible_cop, max_admissible_cop,  Z_y_cl, Y_cl, desired_Z_ref,
        CoM_constraint_vector)
        plot_legend = False
    else:
        plot_utils.plot_y(plot_legend, reference_time_stamp, desired_walking_time,
        min_admissible_cop, max_admissible_cop,  Z_y_cl, Y_cl, desired_Z_ref,
        CoM_constraint_vector)
plt.show()
