# headers:
# -------
from numpy import array, dot, tile, arange, absolute, zeros, append, sqrt, concatenate, exp, random
from second_order.rmpc.robust_constraints import compute_CoP_backoff_dead_beat
from second_order.rmpc.mrpi.polyhedron import polyhedron, plot_polygon_list
from second_order.reference_trajectories import manual_foot_placement
from second_order.reference_trajectories import create_CoP_trajectory
from second_order.reference_trajectories import create_CoM_trajectory
from second_order.cost_function import compute_objective_terms_box
from second_order.motion_model import compute_recursive_matrices
from second_order.motion_model import compute_recursive_dynamics
from chance_constraints import add_CoM_chance_constraints_lfoot
from chance_constraints import add_CoM_chance_constraints_rfoot
from chance_constraints import add_CoM_chance_constraints_box
from second_order.motion_model import discrete_LIP_dynamics
from chance_constraints import add_CoP_chance_constraints
from truncated_normal import sample_from_truncated_normal
from matplotlib.ticker import MaxNLocator
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
foot_length = 0.20
foot_width  = 0.10
omega       = sqrt(g/h) # sqrt(g/h)

# MPC Parameters:
# --------------
delta_t        = 0.1   # MPC sampling period
step_time      = 0.8   # step period
N              = 16    # preceding horizon
no_steps_per_T = int(step_time/delta_t)

# walking parameters:
# ------------------
step_length           = 0.20                              # fixed step length in the xz-plane
no_desired_steps      = 8                                 # number of desired walking steps
no_planned_steps      = 2+no_desired_steps                # planning 2 steps ahead (increase if you want to increase the horizon)
desired_walking_time  = no_desired_steps * no_steps_per_T # number of desired walking intervals
planned_walking_time  = no_planned_steps * no_steps_per_T # number of planned walking intervals
total_no_simulations  = 200
plot_legend = True

# dead-beat choice of LIPM pre-stabilizing gains
# ----------------------------------------------
k = exp(omega*delta_t)/((exp(omega*delta_t))-1.0)
k_dead_beat = array([[k, k/omega]])
constraint_violations_19 = 0.0
constraint_violations_27 = 0.0
constraint_violations_35 = 0.0
constraint_violations_43 = 0.0

for no_sim in range(total_no_simulations):
    # CoM initial state: [x, xdot, x_ddot].T
    #                    [y, ydot, y_ddot].T
    # --------------------------------------
    x_init = array([0.0, 0.0])
    y_init = array([-0.085, 0.0])
    step_width = 2*absolute(y_init[0])

    # discrete dynamics for tracking control law
    A_d, B_d = discrete_LIP_dynamics(delta_t, g, h)

    # 2D-bounded polyhdron additive disturbance set on the motion model
    wc_lb    = -0.0016
    wc_ub    =  0.0016
    wcdot_lb = -0.016
    wcdot_ub =  0.016
    P_A = array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    P_b = array([wc_ub, wc_ub, wcdot_ub, wcdot_ub])
    P_b = P_b.reshape([P_b.shape[0],1])
    W = polyhedron(P_A, P_b)

    Sigma_w = array([wc_ub/2.0, wcdot_ub/2.0]) # covariance vector
    com_constraint   = 0.04
    beta_x  = 0.95 # probability level of com constraints satisfaction
    beta_u  = 0.50 # probability level of cop constraints satisfaction

    # control back-off KW (exact due to dead-beat gains)
    KW = compute_CoP_backoff_dead_beat(k_dead_beat, W)
    #print KW.vertices

    # compute CoP reference trajectory:
    # --------------------------------
    foot_step_0   = array([0.0, y_init[0]])    # initial foot step position in x-y

    desiredFoot_steps  = manual_foot_placement(foot_step_0, step_length,
                                                               no_desired_steps)
    desiredFoot_steps[1:,0] -= step_length
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
    X_ol   = zeros((desired_walking_time+1,2))
    Y_ol   = zeros((desired_walking_time+1,2))
    Z_x_ol = zeros((desired_walking_time+1))
    Z_y_ol = zeros((desired_walking_time+1))

    X_cl   = zeros(((desired_walking_time)+1,2))
    Y_cl   = zeros(((desired_walking_time)+1,2))
    Z_x_cl = zeros(((desired_walking_time)+1))
    Z_y_cl = zeros(((desired_walking_time)+1))

    # set first values of the open and closed loop trajectories to be equal
    X_ol[0,:] = x_init
    Y_ol[0,:] = y_init
    X_cl[0,:] = x_init
    Y_cl[0,:] = y_init
    Z_x_ol[0] = foot_step_0[0]
    Z_y_ol[0] = foot_step_0[1]
    Z_x_cl[0] = foot_step_0[0]
    Z_y_cl[0] = foot_step_0[1]

    # initialization
    [P_ps, P_vs, P_pu, P_vu] = compute_recursive_matrices(delta_t, g, h, N)
    Z_ref_k = planned_Z_ref[0:N,:]
    com_ref_k = planned_com_ref[0:N]

    x_0     = x_init
    x_cl    = x_init
    y_0     = y_init
    y_cl    = y_init

    B_d =  B_d.reshape(B_d.shape[0], 1)
    A_k = A_d + dot(B_d, k_dead_beat)
    #ONLY for plot function call, but not used
    com_constraint_vector = tile(com_constraint, (desired_walking_time+1,1))
    counter = 1
    #---------------------------------------------------------------------------
    #                               MPC loop
    # --------------------------------------------------------------------------
    for i in range(desired_walking_time):
        [Q, p_k] = compute_objective_terms_box(alpha, beta, gamma, step_time,
                                     no_steps_per_T, N, step_length, step_width,
                           P_ps, P_pu, P_vs, P_vu, x_0, y_0, Z_ref_k, com_ref_k)

        [A_zmp_stoch, b_zmp_stoch] = add_CoP_chance_constraints(N, foot_length,
                         foot_width, Z_ref_k, k_dead_beat, A_k, Sigma_w, beta_u)

        # add com constraints right foot
        if i > 0 and i <= N/2:
            #print('right')
            #print(i)
            [A_right, b_right] = add_CoM_chance_constraints_rfoot(i, N, y_0, P_ps,
                                     P_pu, Sigma_w, A_k, beta_x, com_constraint)
            #print(A_right.shape)
            #print(b_right.shape)
            A = concatenate((A_right, A_zmp_stoch), axis = 0)
            b = concatenate((b_right, b_zmp_stoch), axis = 0)
        # add com constraints left foot
        elif i > N/2 and i < N:
            #print('both')
            [A_left, b_left] = add_CoM_chance_constraints_lfoot(counter,
                       N, y_0, P_ps, P_pu, Sigma_w, A_k, beta_x, com_constraint)
            A = concatenate((A_right, A_left, A_zmp_stoch), axis = 0)
            b = concatenate((b_right, b_left, b_zmp_stoch), axis = 0)
            counter+= 1
        # add com constraints box
        elif i>=N and i < desired_walking_time-N:
            [A_box, b_box] = add_CoM_chance_constraints_box(N, y_0, P_ps, P_pu,
                                           Sigma_w, A_k, beta_x, com_constraint)
            A = concatenate((A_box, A_zmp_stoch), axis = 0)
            b = concatenate((b_box, b_zmp_stoch), axis = 0)
        else:
            #print('only once')
            A = A_zmp_stoch
            b = b_zmp_stoch

        # solve the open-loop optimization problem
        U_OL = solve_qp(Q, -p_k, A.T, b)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL, Y_OL] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu, N, x_0,
                                                                      y_0, U_OL)
        # first element of the optimal control trajectory
        vx_MPC = U_OL[0]
        vy_MPC = U_OL[N]

        # first element of the optimal state trajectory
        zx_MPC = X_OL[0,:]
        zy_MPC = Y_OL[0,:]
        # ----------------------------------------------------------------------
        #                    simulation (apply disturbances)
        # ----------------------------------------------------------------------
        B_d = B_d.squeeze()
        # sample white gaussian from the bounded disturbance set W
        wc = sample_from_truncated_normal(0.0, Sigma_w[0], wc_lb, wc_ub)
        wc_dot = sample_from_truncated_normal(0.0, Sigma_w[1], wcdot_lb,
                                                                       wcdot_ub)
        wy_k = array([wc, wc_dot])

        # add noise after two the first two steps
        if i<N:
            wy_k = zeros(2)

        # error dynamics
        e_x_k = x_cl - x_0
        e_y_k = y_cl - y_0

        # apply tube MPC control policy
        ux_k = vx_MPC + dot(k_dead_beat, e_x_k)
        uy_k = vy_MPC + dot(k_dead_beat, e_y_k)

        # save closed-loop CoP trajectories
        Z_x_cl[i+1]  = ux_k
        Z_y_cl[i+1]  = uy_k

        # update open-loop dynamics
        ZX_MPC_plus = dot(A_d, x_0) + dot(B_d, vx_MPC)
        ZY_MPC_plus = dot(A_d, y_0) + dot(B_d, vy_MPC)
        x_0       = ZX_MPC_plus
        y_0       = ZY_MPC_plus

        # update_closed-loop dynamics:
        x_plus = dot(A_d, x_cl) + dot(B_d, ux_k.squeeze())
        y_plus = dot(A_d, y_cl) + dot(B_d, uy_k.squeeze()) + wy_k

        x_cl   = x_plus
        y_cl   = y_plus

        # count CoM Constraint violations
        if i == 19:
            if y_cl[0] < -com_constraint:
                print('I hit right at 1.9 seconds')
                constraint_violations_19+= 1
        if i == 27:
            if y_cl[0] > com_constraint:
                print('I hit left at 2.7 seconds')
                constraint_violations_27+= 1
        if i == 35:
            if y_cl[0] < -com_constraint:
                print('I hit right at 3.5 seconds')
                constraint_violations_35+= 1
        if i == 43:
            if y_cl[0] > com_constraint:
                print('I hit left at 4.4 seconds')
                constraint_violations_43+= 1
        # save closed-loop CoM trajectories
        X_cl[i+1,:]  = x_cl
        Y_cl[i+1,:]  = y_cl
        #-----------------------------------------------------------------------
        #                       save trajectories
        #-----------------------------------------------------------------------
        x_0 = x_cl
        y_0 = y_cl

        # open-loop CoP trajectories
        Z_x_ol[i+1]  = vx_MPC
        Z_y_ol[i+1]  = vy_MPC

        # open-loop CoM trajectories
        X_ol[i+1,:]  = zx_MPC
        Y_ol[i+1,:]  = zy_MPC

        # update CoP reference trajectory for next iteration
        Z_ref_k   = planned_Z_ref[i+1:i+N+1,:]
        com_ref_k = planned_com_ref[i+1:i+N+1]
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

    min_admissible_cop_back_off = desired_Z_ref - \
    tile([foot_length/2, foot_width/2], (desired_walking_time+1,1)) + \
                      tile([foot_length/2, KW.b[0]], (desired_walking_time+1,1))
    max_admissible_cop_back_off = desired_Z_ref + \
    tile([foot_length/2, foot_width/2], (desired_walking_time+1,1)) - \
                      tile([foot_length/2, KW.b[0]], (desired_walking_time+1,1))
    if plot_legend:
        plot_utils.plot_y_stochastic_MPC_box(plot_legend, reference_time_stamp,
                   desired_walking_time, min_admissible_cop, max_admissible_cop,
         min_admissible_cop_back_off, max_admissible_cop_back_off, Z_y_cl, Y_cl,
                    desired_Z_ref, com_constraint_vector, N)
        plot_legend = False
    else:
        plot_utils.plot_y_stochastic_MPC_box(plot_legend, reference_time_stamp,
                   desired_walking_time, min_admissible_cop, max_admissible_cop,
         min_admissible_cop_back_off, max_admissible_cop_back_off, Z_y_cl, Y_cl,
                    desired_Z_ref, com_constraint_vector, N)


# plot no of CoM constraint violations of smpc
fig, ax = plt.subplots()
plt.rc('text', usetex = True)
#plt.rc('font', family ='serif')
labels = ['1.9', '2.7', '3.5', '4.3']
ax.set_xlabel('{time} (s)', fontsize=19)
ax.set_ylabel('no of constraint violations', fontsize=19)
x = array([1.9, 2.7, 3.5, 4.3])
width = 0.35
smpc = ax.bar(x+width/10, array([constraint_violations_19,
                             constraint_violations_27, constraint_violations_35,
                                               constraint_violations_43]), width)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticks(x)
ax.set_xticklabels(labels)
#autolabel(smpc)
fig.tight_layout()
plt.show()
