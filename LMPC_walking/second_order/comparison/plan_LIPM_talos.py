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
from second_order.stmpc.chance_constraints import add_CoM_chance_constraints_lfoot
from second_order.stmpc.chance_constraints import add_CoM_chance_constraints_rfoot
from second_order.stmpc.chance_constraints import add_CoM_chance_constraints_box
from second_order.rmpc.robust_constraints import add_CoM_robust_constraints_lfoot
from second_order.rmpc.robust_constraints import add_CoM_robust_constraints_rfoot
from second_order.rmpc.robust_constraints import add_CoM_robust_constraints_box
from second_order.rmpc.robust_constraints import compute_CoP_backoff_dead_beat
from second_order.stmpc.truncated_normal import sample_from_truncated_normal
from second_order.stmpc.chance_constraints import add_CoP_chance_constraints
from second_order.rmpc.robust_constraints import add_CoP_robust_constraints
from second_order.rmpc.mrpi.polyhedron import polyhedron, plot_polygon_list
import second_order.reference_trajectories as reference_trajectories
from second_order.rmpc.mrpi.mRPI_set import compute_mRPI
import second_order.cost_function as cost_function
import second_order.motion_model as motion_model
import second_order.constraints as constraints
import second_order.plot_utils as plot_utils
import matplotlib.pyplot as plt
from quadprog import solve_qp
import numpy as np

def run_mpc(conf):

    # Inverted pendulum parameters
    foot_length = conf.lxn + conf.lxp   # foot size in the x-direction
    foot_width  = conf.lyn + conf.lyp   # foot size in the y-direciton
    nb_dt_per_step = int(round(conf.T_step/conf.dt_mpc))
    desired_walking_time  = conf.nb_steps * nb_dt_per_step
    planned_walking_time  = (2+conf.nb_steps) * nb_dt_per_step

    # CoM initial state: [x_0, xdot_0].T
    #                    [y_0, ydot_0].T
    x_0_mpc = np.array([conf.foot_step_0[0], 0.0])
    y_0_mpc = np.array([conf.foot_step_0[1], 0.0])
    x_0_rmpc = np.array([conf.foot_step_0[0], 0.0])
    y_0_rmpc = np.array([conf.foot_step_0[1], 0.0])
    x_0_smpc = np.array([conf.foot_step_0[0], 0.0])
    y_0_smpc = np.array([conf.foot_step_0[1], 0.0])
    x_cl_rmpc = x_0_rmpc
    y_cl_rmpc = y_0_rmpc
    x_cl_smpc = x_0_smpc
    y_cl_smpc = y_0_smpc
    step_width = 2*np.absolute(conf.foot_step_0[1])
    # --------------------------------------------------------------------------
    #                    compute constraints backoffs
    # --------------------------------------------------------------------------

    # dead-beat choice of LIPM pre-stabilizing gains
    k = np.exp(conf.omega*conf.dt_mpc)/((np.exp(conf.omega*conf.dt_mpc))-1.0)
    k_dead_beat = np.array([[k, k/conf.omega]])

    # 2D-bounded polyhdron additive disturbance set on the motion model
    P_A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    P_b = np.array([conf.wc_ub, conf.wc_ub, conf.wcdot_ub, conf.wcdot_ub])
    P_b = P_b.reshape([P_b.shape[0],1])
    W = polyhedron(P_A, P_b)
    Sigma_w = np.array([conf.wc_ub/2.0, conf.wcdot_ub/2.0])

    # state back-off \Omega
    A_d, B_d = motion_model.discrete_LIP_dynamics(conf.dt_mpc, conf.g, conf.h)
    B_d =  B_d.reshape(B_d.shape[0], 1)
    A_k = A_d + np.dot(B_d, k_dead_beat)
    Omega, Fs_list = compute_mRPI(conf.epsilon, W, A_d, B_d, k_dead_beat)
    Omega.compute_Hrep()
    #print Omega.vertices, '\n'
    max_vertex = np.amax(Omega.vertices, axis=0)
    com_constraint_vector = np.tile(conf.com_constraint,(desired_walking_time+1,1))
    com_backoff = conf.com_constraint - max_vertex[0]
    #print('state-back-off magnitude = ', max_vertex[0])
    com_backoff_vector = np.tile(com_backoff, (desired_walking_time+1,1))
    #plot_polygon_list(Fs_list)
    #plt.figure()
    # control back-off KBW (exact)
    KW = compute_CoP_backoff_dead_beat(k_dead_beat, W)
    #KW.plot_polygon(title='control backoff magitude KW')
    #print KW.vertices, '\n'
    # --------------------------------------------------------------------------
    #                   compute cop and com reference trajectories
    # --------------------------------------------------------------------------
    foot_steps  = reference_trajectories.manual_foot_placement(conf.foot_step_0,
                                                conf.step_length, conf.nb_steps)
    foot_steps[1:,0] -= conf.step_length
    desired_Z_ref = reference_trajectories.create_CoP_trajectory(conf.nb_steps,
                               foot_steps, desired_walking_time, nb_dt_per_step)

    desired_com_ref = reference_trajectories.create_CoM_trajectory(conf.nb_steps,
                      nb_dt_per_step, desired_walking_time, conf.com_constraint)

    # plan the last 2 steps CoP reference in the future to be the same as last step
    planned_Z_ref = np.zeros((planned_walking_time, 2))
    planned_Z_ref[0:desired_walking_time,:] =  desired_Z_ref
    planned_Z_ref[desired_walking_time:planned_walking_time,:] = \
                                        desired_Z_ref[desired_walking_time-1,:]

    # plan the last 2 steps CoM reference in the future to be the same as last step
    planned_com_ref = np.zeros((planned_walking_time))
    planned_com_ref[0:desired_walking_time] =  desired_com_ref
    planned_com_ref[desired_walking_time:planned_walking_time] = \
                                         desired_com_ref[desired_walking_time-1]
    # pre-allocate memory
    com_state_x_mpc   = np.zeros(((desired_walking_time)+1,2))
    com_state_y_mpc   = np.zeros(((desired_walking_time)+1,2))
    com_state_x_rmpc  = np.zeros(((desired_walking_time)+1,2))
    com_state_y_rmpc  = np.zeros(((desired_walking_time)+1,2))
    com_state_x_smpc  = np.zeros(((desired_walking_time)+1,2))
    com_state_y_smpc  = np.zeros(((desired_walking_time)+1,2))
    cop_x_mpc  = np.zeros(((desired_walking_time)))
    cop_y_mpc  = np.zeros(((desired_walking_time)))
    cop_x_rmpc = np.zeros(((desired_walking_time)))
    cop_y_rmpc = np.zeros(((desired_walking_time)))
    cop_x_smpc = np.zeros(((desired_walking_time)))
    cop_y_smpc = np.zeros(((desired_walking_time)))

    # set first values of the open and closed loop trajectories to be equal
    com_state_x_mpc[0,:]  = x_0_mpc
    com_state_y_mpc[0,:]  = y_0_mpc
    com_state_x_rmpc[0,:] = x_0_rmpc
    com_state_y_rmpc[0,:] = y_0_rmpc
    com_state_x_smpc[0,:] = x_0_smpc
    com_state_y_smpc[0,:] = y_0_smpc

    cop_ref = planned_Z_ref[0:conf.N,:]
    com_ref = planned_com_ref[0:conf.N]

    # construct preview system
    [P_ps, P_vs, P_pu, P_vu] = motion_model.compute_recursive_matrices(\
                                            conf.dt_mpc, conf.g, conf.h, conf.N)
    # --------------------------------------------------------------------------
    #                                   MPC LOOP
    # --------------------------------------------------------------------------
    for i in range(desired_walking_time):
        # sample white gaussian noise from the bounded disturbance set W
        # after the second step
        if i > conf.N-1:
            wc = sample_from_truncated_normal(0, Sigma_w[0], conf.wc_lb,
                                                                     conf.wc_ub)
            wc_dot = sample_from_truncated_normal(0, Sigma_w[1], conf.wcdot_lb,
                                                                  conf.wcdot_ub)
            wy_k = np.array([wc, wc_dot])
        else:
            wy_k = np.zeros(2)
        # ----------------------------------------------------------------------
        #                              NOMINAL
        # ----------------------------------------------------------------------
        [Q_mpc, p_k_mpc] = cost_function.compute_objective_terms_box(conf.alpha,
                      conf.beta, conf.gamma,conf.T_step, nb_dt_per_step, conf.N,
                           conf.step_length, step_width, P_ps, P_pu, P_vs, P_vu,
                                             x_0_mpc, y_0_mpc, cop_ref, com_ref)

        [A_zmp_mpc, b_zmp_mpc] = constraints.add_ZMP_constraints(conf.N,
                             foot_length, foot_width, cop_ref, x_0_mpc, y_0_mpc)
        if i > conf.N-1 and i < desired_walking_time-conf.N:
            [A_com_mpc, b_com_mpc] = constraints.add_CoM_constraints_box(conf.N,
                                       y_0_mpc, P_ps, P_pu, conf.com_constraint)
            A_mpc = np.concatenate((A_com_mpc, A_zmp_mpc), axis = 0)
            b_mpc = np.concatenate((b_com_mpc, b_zmp_mpc), axis = 0)
        else:
            [A_zmp_mpc, b_zmp_mpc] = constraints.add_ZMP_constraints(conf.N,
                             foot_length, foot_width, cop_ref, x_0_mpc, y_0_mpc)
            A_mpc = A_zmp_mpc
            b_mpc = b_zmp_mpc

        # call quadprog solver:
        U_mpc = solve_qp(Q_mpc, -p_k_mpc, A_mpc.T, b_mpc)[0]

        # first element of the optimal control trajectory
        vx_mpc = U_mpc[0]
        vy_mpc = U_mpc[conf.N]

        # save cop
        cop_x_mpc[i] = vx_mpc
        cop_y_mpc[i] = vy_mpc

        # update_closed-loop dynamics:
        B_d = B_d.squeeze()
        x_cl_plus_mpc = np.dot(A_d, x_0_mpc) + np.dot(B_d, vx_mpc.squeeze())
        y_cl_plus_mpc = np.dot(A_d, y_0_mpc) + np.dot(B_d,
                                                        vy_mpc.squeeze()) + wy_k
        x_0_mpc  = x_cl_plus_mpc
        y_0_mpc  = y_cl_plus_mpc

        com_state_x_mpc[i+1] = x_0_mpc
        com_state_y_mpc[i+1] = y_0_mpc
        # ----------------------------------------------------------------------
        #                              ROBUST
        # ----------------------------------------------------------------------
        [Q_rmpc, p_k_rmpc] = cost_function.compute_objective_terms_box(conf.alpha,
                      conf.beta, conf.gamma,conf.T_step, nb_dt_per_step, conf.N,
                           conf.step_length, step_width, P_ps, P_pu, P_vs, P_vu,
                                           x_0_rmpc, y_0_rmpc, cop_ref, com_ref)
        [A_zmp_rmpc, b_zmp_rmpc] = add_CoP_robust_constraints(conf.N,
                                           foot_length, foot_width, cop_ref, KW)
        # add com constraints right foot
        if i > 0 and i <= conf.N/2:
            [A_right_rmpc, b_right_rmpc] = add_CoM_robust_constraints_rfoot(i,
                                                   conf.N, y_0_rmpc, P_ps, P_pu,
                                             conf.com_constraint, max_vertex[0])
            A_rmpc = np.concatenate((A_right_rmpc, A_zmp_rmpc), axis = 0)
            b_rmpc = np.concatenate((b_right_rmpc, b_zmp_rmpc), axis = 0)
        # add com constraints left foot
        elif i > conf.N/2 and i < conf.N:
            no_com_constraints = 1
            [A_left_rmpc, b_left_rmpc] = add_CoM_robust_constraints_lfoot(\
                               no_com_constraints, conf.N, y_0_rmpc, P_ps, P_pu,
                                             conf.com_constraint, max_vertex[0])
            A_rmpc = np.concatenate((A_right_rmpc, A_left_rmpc, A_zmp_rmpc),
                                                                       axis = 0)
            b_rmpc = np.concatenate((b_right_rmpc, b_left_rmpc, b_zmp_rmpc),
                                                                      axis = 0)
            no_com_constraints+= 1
        # add com constraints box
        elif i>=conf.N and i < desired_walking_time-conf.N:
            [A_box_rmpc, b_box_rmpc] = add_CoM_robust_constraints_box(conf.N,
                                     x_0_rmpc, y_0_rmpc, P_ps, P_vs, P_pu, P_vu,
                                             conf.com_constraint, max_vertex[0])
            A_rmpc = np.concatenate((A_box_rmpc, A_zmp_rmpc), axis = 0)
            b_rmpc = np.concatenate((b_box_rmpc, b_zmp_rmpc), axis = 0)
        else:
            A_rmpc = A_zmp_rmpc
            b_rmpc = b_zmp_rmpc

        # solve the open-loop optimization problem
        U_rmpc = solve_qp(Q_rmpc, -p_k_rmpc, A_rmpc.T, b_rmpc)[0]

        # first element of the optimal control trajectory
        vx_rmpc  = U_rmpc[0]
        vy_rmpc = U_rmpc[conf.N]

        # error dynamics
        e_x_rmpc = x_cl_rmpc - x_0_rmpc
        e_y_rmpc = y_cl_rmpc - y_0_rmpc

        # apply tube MPC control policy
        ux_rmpc = vx_rmpc + np.dot(k_dead_beat, e_x_rmpc)
        uy_rmpc = vy_rmpc + np.dot(k_dead_beat, e_y_rmpc)

        # save closed-loop CoP trajectories
        cop_x_rmpc[i]  = ux_rmpc
        cop_y_rmpc[i]  = uy_rmpc

        # update open-loop dynamics
        x_0_plus_rmpc = np.dot(A_d, x_0_rmpc) + np.dot(B_d, vx_rmpc)
        y_0_plus_rmpc = np.dot(A_d, y_0_rmpc) + np.dot(B_d, vy_rmpc)
        x_0_rmpc  = x_0_plus_rmpc
        y_0_rmpc  = y_0_plus_rmpc

        # update_closed-loop dynamics:
        x_cl_plus_rmpc = np.dot(A_d, x_cl_rmpc) + np.dot(B_d, ux_rmpc.squeeze())
        y_cl_plus_rmpc = np.dot(A_d, y_cl_rmpc) + np.dot(B_d, uy_rmpc.squeeze())\
                                                                         + wy_k
        x_cl_rmpc = x_cl_plus_rmpc
        y_cl_rmpc = y_cl_plus_rmpc

        # save closed-loop CoM trajectories
        com_state_x_rmpc[i+1,:] = x_cl_plus_rmpc
        com_state_y_rmpc[i+1,:] = y_cl_plus_rmpc

        # update the open-loop and closed-loop initial states for next iteration
        x_0_rmpc = x_0_plus_rmpc
        y_0_rmpc = y_0_plus_rmpc

        # ----------------------------------------------------------------------
        #                              STOCHASTIC
        # ----------------------------------------------------------------------
        [Q_smpc, p_smpc] = cost_function.compute_objective_terms_box(conf.alpha,
                    conf.beta, conf.gamma, conf.T_step, nb_dt_per_step , conf.N,
             conf.step_length, conf.step_width, P_ps, P_pu, P_vs, P_vu, x_0_smpc,
                                                     y_0_smpc, cop_ref, com_ref)
        [A_zmp_smpc, b_zmp_smpc] = add_CoP_chance_constraints(conf.N,
                             foot_length, foot_width, cop_ref, k_dead_beat, A_k,
                                                           Sigma_w, conf.beta_u)
        # add com constraints right foot
        if i > 0 and i <= conf.N/2:
            [A_right_smpc, b_right_smpc] = add_CoM_chance_constraints_rfoot(i,
                        conf.N, y_0_smpc, P_ps, P_pu, Sigma_w, A_k, conf.beta_x,
                                                            conf.com_constraint)
            A_smpc = np.concatenate((A_right_smpc, A_zmp_smpc), axis = 0)
            b_smpc = np.concatenate((b_right_smpc, b_zmp_smpc), axis = 0)
            # add com constraints left foot
        elif i > conf.N/2 and i < conf.N:
            no_com_constraints = 1
            [A_left_smpc, b_left_smpc] = add_CoM_chance_constraints_lfoot(\
                      no_com_constraints, conf.N, y_0_smpc, P_ps, P_pu, Sigma_w,
                                          A_k, conf.beta_x, conf.com_constraint)
            A_smpc = np.concatenate((A_right_smpc, A_left_smpc, A_zmp_smpc),
                                                                       axis = 0)
            b_smpc = np.concatenate((b_right_smpc, b_left_smpc, b_zmp_smpc),
                                                                       axis = 0)
            no_com_constraints+= 1
        # add com constraints box
        elif i>=conf.N and i < desired_walking_time-conf.N:
            [A_box_smpc, b_box_smpc] = add_CoM_chance_constraints_box(conf.N,
                                y_0_smpc, P_ps, P_pu, Sigma_w, A_k, conf.beta_x,
                                                            conf.com_constraint)
            A_smpc = np.concatenate((A_box_smpc, A_zmp_smpc), axis = 0)
            b_smpc = np.concatenate((b_box_smpc, b_zmp_smpc), axis = 0)
        else:
            A_smpc = A_zmp_smpc
            b_smpc = b_zmp_smpc

        # solve the open-loop optimization problem
        U_smpc = solve_qp(Q_smpc, -p_smpc, A_smpc.T, b_smpc)[0]

        # first element of the optimal control trajectory
        vx_smpc = U_smpc[0]
        vy_smpc = U_smpc[conf.N]

        # error dynamics
        e_x_smpc = x_cl_smpc - x_0_smpc
        e_y_smpc = y_cl_smpc - y_0_smpc

        # apply tube MPC control policy
        ux_smpc = vx_smpc + np.dot(k_dead_beat, e_x_smpc)
        uy_smpc = vy_smpc + np.dot(k_dead_beat, e_y_smpc)

        # save closed-loop CoP trajectories
        cop_x_smpc[i]  = ux_smpc
        cop_y_smpc[i]  = uy_smpc

        # update open-loop dynamics
        x_0_plus_smpc = np.dot(A_d, x_0_smpc) + np.dot(B_d, vx_smpc)
        y_0_plus_smpc = np.dot(A_d, y_0_smpc) + np.dot(B_d, vy_smpc)
        x_0_smpc  = x_0_plus_smpc
        y_0_smpc  = y_0_plus_smpc

        # update_closed-loop dynamics:
        x_cl_plus_smpc = np.dot(A_d, x_cl_smpc) + np.dot(B_d, ux_smpc.squeeze())
        y_cl_plus_smpc = np.dot(A_d, y_cl_smpc) + np.dot(B_d, uy_smpc.squeeze())\
                                                                          + wy_k
        x_cl_smpc = x_cl_plus_smpc
        y_cl_smpc = y_cl_plus_smpc

        # save closed-loop CoM trajectories
        com_state_x_smpc[i+1,:] = x_cl_plus_smpc
        com_state_y_smpc[i+1,:] = y_cl_plus_smpc

        # update the open-loop and closed-loop initial states for next iteration
        x_0_smpc   = x_cl_smpc
        y_0_smpc   = y_cl_smpc

        # update reference trajectories
        cop_ref = planned_Z_ref[i+1:i+conf.N+1,:]
        com_ref = planned_com_ref[i+1:i+conf.N+1]
    # --------------------------------------------------------------------------
    #                            save trajectories
    # --------------------------------------------------------------------------
    np.savez(conf.DATA_FILE_LIPM_MPC, com_state_x = com_state_x_mpc,
                       com_state_y = com_state_y_mpc, cop_y = cop_y_mpc,
        cop_ref = desired_Z_ref, cop_x = cop_x_mpc, foot_steps = foot_steps)

    np.savez(conf.DATA_FILE_LIPM_RMPC, com_state_x = com_state_x_rmpc,
                   com_state_y = com_state_y_rmpc, cop_y = cop_y_rmpc,
      cop_ref = desired_Z_ref, cop_x = cop_x_rmpc, foot_steps = foot_steps)

    np.savez(conf.DATA_FILE_LIPM_SMPC, com_state_x = com_state_x_smpc,
                   com_state_y = com_state_y_smpc, cop_y = cop_y_smpc,
      cop_ref = desired_Z_ref, cop_x = cop_x_smpc, foot_steps = foot_steps)
    # --------------------------------------------------------------------------
    #                     plot closed-loop trajectories
    # --------------------------------------------------------------------------
    time = np.arange(0, round((desired_walking_time+1)*conf.dt_mpc, 2),
                                                                    conf.dt_mpc)
    desired_Z_ref = np.append([desired_Z_ref[0,:]], desired_Z_ref,axis=0)
    x_init = np.array([conf.foot_step_0[0], 0.0])
    y_init = np.array([conf.foot_step_0[1], 0.0])
    cop_y_mpc = np.append([conf.foot_step_0[1]], cop_y_mpc, axis=0)
    cop_y_rmpc = np.append([conf.foot_step_0[1]], cop_y_rmpc, axis=0)
    cop_y_smpc = np.append([conf.foot_step_0[1]], cop_y_smpc, axis=0)

    min_admissible_cop = desired_Z_ref - np.tile([foot_length/2, foot_width/2],
                                                     (desired_walking_time+1,1))
    max_admissible_cop = desired_Z_ref + np.tile([foot_length/2, foot_width/2],
                                                     (desired_walking_time+1,1))
    min_admissible_cop_back_off = desired_Z_ref - \
    np.tile([foot_length/2, foot_width/2], (desired_walking_time+1,1)) +\
                   np.tile([foot_length/2, KW.b[0]], (desired_walking_time+1,1))
    max_admissible_cop_back_off = desired_Z_ref + \
        np.tile([foot_length/2, foot_width/2], (desired_walking_time+1,1)) -\
                   np.tile([foot_length/2, KW.b[0]], (desired_walking_time+1,1))

    if conf.PLOT_LIPM:
        mpc_figure = plt.figure()
        plot_utils.plot_y_box(True, time, desired_walking_time,
             min_admissible_cop, max_admissible_cop, cop_y_mpc, com_state_y_mpc,
                                         cop_ref, com_constraint_vector, conf.N)
        rmpc_figure = plt.figure()
        plot_utils.plot_y_robust_MPC_box(True, time,
            desired_walking_time, min_admissible_cop, max_admissible_cop,
            min_admissible_cop_back_off, max_admissible_cop_back_off, cop_y_rmpc,
                         com_state_y_rmpc, desired_Z_ref, com_constraint_vector,
                                                     com_backoff_vector, conf.N)
        smpc_figure = plt.figure()
        plot_utils.plot_y_stochastic_MPC_box(True, time,
            desired_walking_time, min_admissible_cop, max_admissible_cop,
            min_admissible_cop_back_off, max_admissible_cop_back_off, cop_y_smpc,
                         com_state_y_smpc, desired_Z_ref, com_constraint_vector,
                                                                         conf.N)
# ------------------------------------------------------------------------------
#                              RED PILL OR BLUE PILL
# ------------------------------------------------------------------------------
if __name__=='__main__':
    import conf
    run_mpc(conf)
