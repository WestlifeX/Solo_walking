# headers:
# -------
from numpy import concatenate, array, dot, tile, arange, absolute, zeros, append, savez, exp, amax, sqrt
from second_order.rmpc.robust_constraints import add_CoM_robust_constraints_lfoot
from second_order.rmpc.robust_constraints import add_CoM_robust_constraints_rfoot
from second_order.rmpc.robust_constraints import add_CoM_robust_constraints_box
from second_order.rmpc.robust_constraints import compute_CoP_backoff_dead_beat
from second_order.stmpc.truncated_normal import sample_from_truncated_normal
from second_order.rmpc.robust_constraints import add_CoP_robust_constraints
from second_order.rmpc.mrpi.polyhedron import polyhedron, plot_polygon_list
from second_order.motion_model import compute_recursive_disturbed_dynamics
from second_order.reference_trajectories import manual_foot_placement
from second_order.reference_trajectories import create_CoP_trajectory
from second_order.reference_trajectories import create_CoM_trajectory
from second_order.cost_function import compute_objective_terms_box
from second_order.motion_model import compute_recursive_matrices
from second_order.motion_model import compute_recursive_dynamics
from second_order.motion_model import discrete_LIP_dynamics
from second_order.rmpc.mrpi.mRPI_set import compute_mRPI
from second_order.constraints import add_ZMP_constraints
from second_order import plot_utils
import matplotlib.pyplot as plt
from quadprog import solve_qp
import conf_talos as conf

def run_rmpc(conf):

    foot_length           = conf.lxn + conf.lxp   # foot size in the x-direction
    foot_width            = conf.lyn + conf.lyp   # foot size in the y-direciton
    nb_dt_per_step        = int(round(conf.T_step/conf.dt_mpc))
    desired_walking_time  = conf.nb_steps * nb_dt_per_step
    planned_walking_time  = (2+conf.nb_steps) * nb_dt_per_step


    # dead-beat choice of LIPM pre-stabilizing gains
    # ----------------------------------------------
    k = exp(conf.omega*conf.dt_mpc)/((exp(conf.omega*conf.dt_mpc))-1.0)
    k_dead_beat = array([[k, k/conf.omega]])

    # CoM initial state: [x, xdot, x_ddot].T
    #                    [y, ydot, y_ddot].T
    # --------------------------------------
    x_init = array([0.0, 0.0])
    y_init = array([conf.foot_step_0[1], 0.0])

    step_width = 2*absolute(conf.foot_step_0[1])

    # discrete dynamics for tracking control law
    A_d, B_d = discrete_LIP_dynamics(conf.dt_mpc, conf.g, conf.h)
    B_d =  B_d.reshape(B_d.shape[0], 1)

    # 2D-bounded polyhdron additive disturbance set on the motion model
    P_A = array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    P_b = array([conf.wc_ub, conf.wc_ub, conf.wcdot_ub, conf.wcdot_ub])
    P_b = P_b.reshape([P_b.shape[0],1])
    W = polyhedron(P_A, P_b)

    # compute constraints backoffs:
    # ----------------------------

    # state back-off \Omega
    Omega, Fs_list = compute_mRPI(conf.epsilon, W, A_d, B_d, k_dead_beat)
    Omega.compute_Hrep()
    #print Omega.vertices, '\n'
    max_vertex = amax(Omega.vertices, axis=0)
    com_constraint_vector = tile(conf.com_constraint,(desired_walking_time+1,1))
    com_backoff = conf.com_constraint - max_vertex[0]
    print('state-back-off magnitude = ', max_vertex[0])
    com_backoff_vector = tile(com_backoff, (desired_walking_time+1,1))
    #plot_polygon_list(Fs_list)
    #plt.figure()
    # control back-off KBW (exact)
    KW = compute_CoP_backoff_dead_beat(k_dead_beat, W)
    #KW.plot_polygon(title='control backoff magitude KW')
    #print KW.vertices, '\n'
    plot_legend = conf.plot_legend

    # compute CoP reference trajectory:
    # --------------------------------
    desiredFoot_steps  = manual_foot_placement(conf.foot_step_0,
                                                conf.step_length, conf.nb_steps)
    desiredFoot_steps[1:,0] -= conf.step_length                                        
    desired_Z_ref = create_CoP_trajectory(conf.nb_steps, desiredFoot_steps,
                                           desired_walking_time, nb_dt_per_step)

    desired_com_ref = create_CoM_trajectory(conf.nb_steps, nb_dt_per_step,
                                      desired_walking_time, conf.com_constraint)
    #print(desired_com_ref)
    # plan the last 2 steps cop reference in the future to be the same as last step
    planned_Z_ref = zeros((planned_walking_time, 2))
    planned_Z_ref[0:desired_walking_time,:] =  desired_Z_ref
    planned_Z_ref[desired_walking_time:planned_walking_time,:] = \
                                         desired_Z_ref[desired_walking_time-1,:]
    # plan the last 2 steps com reference in the future to be the same as last step
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
    X_cl[0,:] = x_init
    Y_cl[0,:] = y_init
    Z_x_cl[0] = conf.foot_step_0[0]
    Z_y_cl[0] = conf.foot_step_0[1]

    # initialization
    [P_ps, P_vs, P_pu, P_vu] = compute_recursive_matrices(conf.dt_mpc,
                                                         conf.g, conf.h, conf.N)
    Z_ref_k = planned_Z_ref[0:conf.N,:]
    com_ref_k = planned_com_ref[0:conf.N]
    x_0     = x_init
    x_cl    = x_init
    y_0     = y_init
    y_cl    = y_init
    Sigma_w = array([conf.wc_ub/2.0, conf.wcdot_ub/2.0])
    counter = 1
    #---------------------------------------------------------------------------
    #                             MPC loop
    # --------------------------------------------------------------------------
    for i in range(desired_walking_time):
        #print(i)
        [Q, p_k] = compute_objective_terms_box(conf.alpha, conf.beta, conf.gamma,
                          conf.T_step, nb_dt_per_step, conf.N, conf.step_length,
                          conf.step_width, P_ps, P_pu, P_vs, P_vu, x_0, y_0,
                                                             Z_ref_k, com_ref_k)
        [A_zmp_robust, b_zmp_robust] = add_CoP_robust_constraints(conf.N,
                                           foot_length, foot_width, Z_ref_k, KW)
        # add com constraints right foot
        if i > 0 and i <= conf.N/2:
            [A_right, b_right] = add_CoM_robust_constraints_rfoot(i, conf.N,
                            y_0, P_ps, P_pu, conf.com_constraint, max_vertex[0])
            A = concatenate((A_right, A_zmp_robust), axis = 0)
            b = concatenate((b_right, b_zmp_robust), axis = 0)
        # add com constraints left foot
        elif i > conf.N/2 and i < conf.N:
            [A_left, b_left] = add_CoM_robust_constraints_lfoot(counter,
                    conf.N, y_0, P_ps, P_pu, conf.com_constraint, max_vertex[0])
            A = concatenate((A_right, A_left, A_zmp_robust), axis = 0)
            b = concatenate((b_right, b_left, b_zmp_robust), axis = 0)
            counter+= 1
        # add com constraints box
        elif i>=conf.N and i < desired_walking_time-conf.N:
            [A_box, b_box] = add_CoM_robust_constraints_box(conf.N, x_0, y_0,
                                    P_ps, P_vs, P_pu, P_vu, conf.com_constraint,
                                                                  max_vertex[0])
            A = concatenate((A_box, A_zmp_robust), axis = 0)
            b = concatenate((b_box, b_zmp_robust), axis = 0)
        else:
            A = A_zmp_robust
            b = b_zmp_robust

        # solve the open-loop optimization problem
        U_OL = solve_qp(Q, -p_k, A.T, b)[0]

        # simulate your recursive nominal dynamics over the current horizon
        [X_OL, Y_OL] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu,
                                                     conf.N, x_0, y_0, U_OL)

        # first element of the optimal control trajectory
        vx_MPC = U_OL[0]
        vy_MPC = U_OL[conf.N]

        # first element of the optimal state trajectory
        zx_MPC = X_OL[0,:]
        zy_MPC = Y_OL[0,:]
        # ----------------------------------------------------------------------
        #                   simulation (add disturbances)
        # ----------------------------------------------------------------------
        B_d = B_d.squeeze()

        # sample white gaussian noise from the bounded disturbance set W
        wc = sample_from_truncated_normal(0, Sigma_w[0], conf.wc_lb,
                                                                     conf.wc_ub)
        wc_dot = sample_from_truncated_normal(0, Sigma_w[1], conf.wcdot_lb,
                                                                  conf.wcdot_ub)
        wy_k = array([wc, wc_dot])

        # uncomment to add worst case distrubance
        if i<conf.N:
            wy_k = zeros(2)
        #if i>=N:
        #    wy_k = array([wc_ub, wcdot_ub])

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
        x_0_plus = dot(A_d, x_0) + dot(B_d, vx_MPC)
        y_0_plus = dot(A_d, y_0) + dot(B_d, vy_MPC)
        x_0  = x_0_plus
        y_0  = y_0_plus

        # update_closed-loop dynamics:
        x_cl_plus = dot(A_d, x_cl) + dot(B_d, ux_k.squeeze())
        y_cl_plus = dot(A_d, y_cl) + dot(B_d, uy_k.squeeze()) + wy_k

        x_cl   = x_cl_plus
        y_cl   = y_cl_plus

        # save closed-loop CoM trajectories
        X_cl[i+1,:] = x_cl_plus
        Y_cl[i+1,:] = y_cl_plus
        #-----------------------------------------------------------------------
        #                       save trajectories
        #-----------------------------------------------------------------------
        # update the open-loop and closed-loop initial states for next iteration
        x_0   = zx_MPC
        y_0   = zy_MPC

        # update CoP reference trajectory for next iteration
        Z_ref_k  = planned_Z_ref[i+1:i+conf.N+1,:]
        com_ref_k = planned_com_ref[i+1:i+conf.N+1]
    # --------------------------------------------------------------------------
    #                  visualize your closed-loop trajectories
    # --------------------------------------------------------------------------
    savez(conf.DATA_FILE_LIPM, com_state_x= X_cl, com_state_y= Y_cl,
              cop_ref = desired_Z_ref, cop_x = Z_x_cl[1::], cop_y = Z_y_cl[1::],
                                                 foot_steps = desiredFoot_steps)

    reference_time_stamp = arange(0,
                round((desired_walking_time+1)*conf.dt_mpc, 2), conf.dt_mpc)

    desired_Z_ref = append([desired_Z_ref[0,:]], desired_Z_ref,axis=0)
    min_admissible_cop = desired_Z_ref - tile([foot_length/2, foot_width/2],
                                                     (desired_walking_time+1,1))
    max_admissible_cop = desired_Z_ref + tile([foot_length/2, foot_width/2],
                                                     (desired_walking_time+1,1))

    min_admissible_cop_back_off = desired_Z_ref - \
    tile([foot_length/2, foot_width/2], (desired_walking_time+1,1)) +\
                      tile([foot_length/2, KW.b[0]], (desired_walking_time+1,1))
    max_admissible_cop_back_off = desired_Z_ref + \
    tile([foot_length/2, foot_width/2], (desired_walking_time+1,1)) -\
                      tile([foot_length/2, KW.b[0]], (desired_walking_time+1,1))
    if plot_legend:
        plot_utils.plot_y_robust_MPC_box(conf.plot_legend,
                 reference_time_stamp, desired_walking_time, min_admissible_cop,
                                max_admissible_cop, min_admissible_cop_back_off,
                       max_admissible_cop_back_off, Z_y_cl, Y_cl, desired_Z_ref,
                              com_constraint_vector, com_backoff_vector, conf.N)
        plot_legend = False
    else:
        plot_utils.plot_y_robust_MPC_box(plot_legend, reference_time_stamp,
                   desired_walking_time, min_admissible_cop, max_admissible_cop,
               min_admissible_cop_back_off, max_admissible_cop_back_off, Z_y_cl,
          Y_cl, desired_Z_ref, com_constraint_vector, com_backoff_vector, conf.N)

    #plt.show()
