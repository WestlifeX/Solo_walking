# headers:
# -------
from numpy import array, dot, tile, arange, absolute, zeros, append, concatenate, exp
from second_order.rmpc.robust_constraints import compute_CoP_backoff_dead_beat
from second_order.rmpc.mrpi.polyhedron import polyhedron, plot_polygon_list
from second_order.reference_trajectories import manual_foot_placement
from second_order.reference_trajectories import create_CoP_trajectory
from second_order.motion_model import compute_recursive_matrices
from second_order.motion_model import compute_recursive_dynamics
from second_order.cost_function import compute_objective_terms
from second_order.motion_model import discrete_LIP_dynamics
from truncated_normal import sample_from_truncated_normal
from chance_constraints import add_CoM_chance_constraints
from chance_constraints import add_CoP_chance_constraints
from second_order.constraints import add_ZMP_constraints
from second_order.constraints import add_CoM_constraints
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
h           = 0.80
g           = 9.81
foot_length = 0.20
foot_width  = 0.14
omega       = 3.5   # sqrt(h/g)

# MPC Parameters:
# --------------
delta_t               = 0.1                         # MPC sampling period
step_time             = 0.8                         # step period
N                     = 16                          # preceding horizon
no_steps_per_T        = int(step_time/delta_t)

# dead-beat choice of LIPM pre-stabilizing gains
# ----------------------------------------------
k = exp(omega*delta_t)/((exp(omega*delta_t))-1.0)
k_dead_beat = array([[k, k/omega]])

# walking parameters:
# ------------------
step_length           = 0.25                              # fixed step length in the xz-plane
no_desired_steps      = 2                                 # number of desired walking steps
no_planned_steps      = 2+no_desired_steps                # planning 2 steps ahead (increase if you want to increase the horizon)
desired_walking_time  = no_desired_steps * no_steps_per_T # number of desired walking intervals
planned_walking_time  = no_planned_steps * no_steps_per_T # number of planned walking intervals

constraint_violations_mpc =  zeros(desired_walking_time+1)
constraint_violations_smpc = zeros(desired_walking_time+1)
plot_legend = True
total_sim = 200

for no_sim in range(total_sim):

    # CoM initial state: [x, xdot, x_ddot].T
    #                    [y, ydot, y_ddot].T
    # --------------------------------------
    x_init = array([0.0, 0.0])
    y_init = array([-0.10, 0.0])

    step_width = 2*absolute(y_init[0])

    # 2D-bounded polyhdron additive disturbance set on the motion model
    wc_lb    = -0.002
    wc_ub    =  0.002
    wcdot_lb = -0.02
    wcdot_ub =  0.02
    P_A = array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    P_b = array([wc_ub, wc_ub, wcdot_ub, wcdot_ub])
    P_b = P_b.reshape([P_b.shape[0],1])
    W = polyhedron(P_A, P_b)
    KW = compute_CoP_backoff_dead_beat(k_dead_beat, W)

    # compute CoP reference trajectory:
    # --------------------------------
    foot_step_0   = array([0.0, -0.10])    # initial foot step position in x-y

    desiredFoot_steps  = manual_foot_placement(foot_step_0, step_length,
                                               no_desired_steps)
    desired_Z_ref = create_CoP_trajectory(no_desired_steps, desiredFoot_steps,
                                          desired_walking_time, no_steps_per_T)

    # plan the last 2 steps CoP reference in the future to be the same as last step
    planned_Z_ref = zeros((planned_walking_time, 2))
    planned_Z_ref[0:desired_walking_time,:] =  desired_Z_ref
    planned_Z_ref[desired_walking_time:planned_walking_time,:] = \
                                        desired_Z_ref[desired_walking_time-1,:]

    # pre-allocate memory
    X_cl_smpc   = zeros(((desired_walking_time)+1,2))
    Y_cl_smpc   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_smpc = zeros(((desired_walking_time)+1))
    Z_y_cl_smpc = zeros(((desired_walking_time)+1))

    X_cl_mpc   = zeros(((desired_walking_time)+1,2))
    Y_cl_mpc   = zeros(((desired_walking_time)+1,2))
    Z_x_cl_mpc = zeros(((desired_walking_time)+1))
    Z_y_cl_mpc = zeros(((desired_walking_time)+1))

    # set first values of the open and closed loop trajectories to be equal
    X_cl_smpc[0,:] = x_init
    Y_cl_smpc[0,:] = y_init
    Z_x_cl_smpc[0] = foot_step_0[0]
    Z_y_cl_smpc[0] = foot_step_0[1]

    X_cl_mpc[0,:] = x_init
    Y_cl_mpc[0,:] = y_init
    Z_x_cl_mpc[0] = foot_step_0[0]
    Z_y_cl_mpc[0] = foot_step_0[1]

    # discrete dynamics
    A_d, B_d = discrete_LIP_dynamics(delta_t, g, h)
    [P_ps, P_vs, P_pu, P_vu] = compute_recursive_matrices(delta_t, g, h, N)
    Z_ref_k = planned_Z_ref[0:N,:]

    x_0_smpc     = x_init
    x_cl_smpc    = x_init
    y_0_smpc     = y_init
    y_cl_smpc    = y_init

    x_0_mpc     = x_init
    x_cl_mpc    = x_init
    y_0_mpc     = y_init
    y_cl_mpc    = y_init

    confidence_level = 0.95
    CoM_constraint   = 0.05
    Sigma_w = array([wc_ub/2.0, wcdot_ub/2.0])
    B_d =  B_d.reshape(B_d.shape[0], 1)
    A_k = A_d + dot(B_d, k_dead_beat)
    #---------------------------------------------------------------------------
    #                           MPC loop
    # --------------------------------------------------------------------------
    for i in range(desired_walking_time):

        [Q_smpc, p_k_smpc]  = compute_objective_terms(alpha, beta, gamma, step_time,
             no_steps_per_T, N, step_length, step_width, P_ps, P_pu, P_vs, P_vu,
             x_0_smpc, y_0_smpc, Z_ref_k)

        [A_zmp_smpc, b_zmp_smpc] = add_CoP_chance_constraints(N, foot_length,
                           foot_width, Z_ref_k, k_dead_beat, A_k, Sigma_w, 0.50)

        [A_CoM_smpc, b_CoM_smpc] = add_CoM_chance_constraints(N, y_0_smpc, P_ps,
                           P_pu, Sigma_w, A_k, confidence_level, CoM_constraint)

        A_smpc = concatenate((A_CoM_smpc, A_zmp_smpc), axis = 0)
        b_smpc = concatenate((b_CoM_smpc, b_zmp_smpc), axis = 0)

        # solve the open-loop optimization problem
        U_OL_smpc = solve_qp(Q_smpc, -p_k_smpc, A_smpc.T, b_smpc)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_smpc, Y_OL_smpc] = compute_recursive_dynamics(P_ps, P_vs, P_pu,
                                         P_vu, N, x_0_smpc, y_0_smpc, U_OL_smpc)

        # first element of the optimal control trajectory
        vx_smpc = U_OL_smpc[0]
        vy_smpc = U_OL_smpc[N]

        # first element of the optimal state trajectory
        zx_smpc = X_OL_smpc[0,:]
        zy_smpc = Y_OL_smpc[0,:]

        #-----------------------------------------------------------------------
        [Q_mpc, p_k_mpc] = compute_objective_terms(alpha, beta, gamma, step_time,
                                  no_steps_per_T, N, step_length, step_width,
                            P_ps, P_pu, P_vs, P_vu, x_0_mpc, y_0_mpc, Z_ref_k)

        [A_zmp_mpc, b_zmp_mpc] = add_ZMP_constraints(N, foot_length, foot_width,
                                                    Z_ref_k, x_0_mpc, y_0_mpc)
        [A_CoM_mpc , b_CoM_mpc] = add_CoM_constraints(N, y_0_mpc, P_ps, P_pu,
                                                      CoM_constraint)

        A_mpc = concatenate((A_CoM_mpc, A_zmp_mpc), axis = 0)
        b_mpc = concatenate((b_CoM_mpc, b_zmp_mpc), axis = 0)

        # solve the open-loop optimization problem
        U_OL_mpc = solve_qp(Q_mpc, -p_k_mpc, A_mpc.T, b_mpc)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL_mpc, Y_OL_mpc] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu,
                                                N, x_0_mpc, y_0_mpc, U_OL_mpc)

        # first element of the optimal control trajectory
        vx_mpc = U_OL_mpc[0]
        vy_mpc = U_OL_mpc[N]

        # first element of the optimal state trajectory
        zx_mpc = X_OL_mpc[0,:]
        zy_mpc = Y_OL_mpc[0,:]

        B_d = B_d.squeeze()

        # sample white gaussian from the bounded disturbance set W
        wc = sample_from_truncated_normal(0.0, Sigma_w[0], wc_lb, wc_ub)
        wc_dot = sample_from_truncated_normal(0.0, Sigma_w[1], wcdot_lb, wcdot_ub)
        wy_k = array([wc, wc_dot])

        # error dynamics
        e_x_k_smpc = x_cl_smpc - x_0_smpc
        e_y_k_smpc = y_cl_smpc - y_0_smpc
        #print e_y_k_smpc

        # apply tube MPC control policy
        ux_k_smpc = vx_smpc + dot(k_dead_beat, e_x_k_smpc)
        uy_k_smpc = vy_smpc + dot(k_dead_beat, e_y_k_smpc)

        ux_k_mpc = vx_mpc
        uy_k_mpc = vy_mpc

        # save closed-loop CoP trajectories
        Z_x_cl_smpc[i+1] = ux_k_smpc
        Z_y_cl_smpc[i+1] = uy_k_smpc

        Z_x_cl_mpc[i+1]  = ux_k_mpc
        Z_y_cl_mpc[i+1]  = uy_k_mpc

        # update_closed-loop dynamics:
        x_cl_plus_smpc = dot(A_d, x_cl_smpc) + dot(B_d, ux_k_smpc.squeeze())
        y_cl_plus_smpc = dot(A_d, y_cl_smpc) + dot(B_d, uy_k_smpc.squeeze()) + wy_k
        x_cl_smpc   = x_cl_plus_smpc
        y_cl_smpc   = y_cl_plus_smpc

        x_cl_plus_mpc = dot(A_d, x_cl_mpc) + dot(B_d, ux_k_mpc.squeeze())
        y_cl_plus_mpc = dot(A_d, y_cl_mpc) + dot(B_d, uy_k_mpc.squeeze()) + wy_k
        x_cl_mpc   = x_cl_plus_mpc
        y_cl_mpc   = y_cl_plus_mpc

        # count CoM Constraint violations
        if y_cl_mpc[0] > CoM_constraint:
            constraint_violations_mpc[i+1] = constraint_violations_mpc[i+1] + 1

        if y_cl_smpc[0] > CoM_constraint:
            constraint_violations_smpc[i+1] = constraint_violations_smpc[i+1] + 1

        # save closed-loop CoM trajectories
        X_cl_smpc[i+1,:] = x_cl_plus_smpc
        Y_cl_smpc[i+1,:] = y_cl_plus_smpc

        X_cl_mpc[i+1,:] = x_cl_plus_mpc
        Y_cl_mpc[i+1,:] = y_cl_plus_mpc

        #-----------------------------------------------------------------------
        #                update initial states and references
        #-----------------------------------------------------------------------
        x_0_smpc = x_cl_plus_smpc
        y_0_smpc = y_cl_plus_smpc

        x_0_mpc = x_cl_plus_mpc
        y_0_mpc = y_cl_plus_mpc

        # update CoP reference trajectory for next iteration
        Z_ref_k   = planned_Z_ref[i+1:i+N+1,:]
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

    CoM_constraint = tile(CoM_constraint, (desired_walking_time+1,1))
    CoM_back_off = tile(0.0381, (desired_walking_time+1,1))

    if plot_legend:
        plot_utils.plot_y_MPC_vs_SMPC(plot_legend, reference_time_stamp,
        desired_walking_time, min_admissible_cop, max_admissible_cop,
        min_admissible_cop_back_off, max_admissible_cop_back_off, Z_y_cl_mpc,
        Y_cl_mpc, Z_y_cl_smpc, Y_cl_smpc, desired_Z_ref, CoM_constraint, CoM_back_off)
        plot_legend = False
    else:
        plot_utils.plot_y_MPC_vs_SMPC(plot_legend, reference_time_stamp,
        desired_walking_time, min_admissible_cop, max_admissible_cop,
        min_admissible_cop_back_off, max_admissible_cop_back_off, Z_y_cl_mpc,
        Y_cl_mpc, Z_y_cl_smpc, Y_cl_smpc, desired_Z_ref, CoM_constraint, CoM_back_off)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

print('constraint_violations_smpc point-wise in time = ', constraint_violations_smpc)
print('constraint_violations_mpc point-wise in time = ', constraint_violations_mpc)

# plot no of CoM constraint violations of mpc vs smpc
fig, ax = plt.subplots()
labels = ['$1.1$', '$1.2$', '$1.3$', '$1.4$', '$1.5$', '$1.6$']
ax.set_xlabel(r'\textbf{time} (s)')
ax.set_ylabel('no of constraint violations')
x = arange(11, 16)
width = 0.35
mpc = ax.bar(x-width/2.0 , constraint_violations_mpc[12::], width,  label ='nominal MPC')
smpc = ax.bar(x+width/2.0, constraint_violations_smpc[12::], width, label ='SMPC')
ax.set_xticks([])
ax.set_xticklabels([])
ax.legend()
autolabel(mpc)
autolabel(smpc)
fig.tight_layout()
plt.show()
