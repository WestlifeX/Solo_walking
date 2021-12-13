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
import second_order.reference_trajectories as reference_trajectories
import second_order.cost_function as cost_function
import second_order.motion_model as motion_model
import second_order.constraints as constraints
import second_order.plot_utils as plot_utils
import matplotlib.pyplot as plt
from quadprog import solve_qp
import conf_talos as conf
import numpy as np

def run_nominal_mpc(conf):
    # Inverted pendulum parameters:
    # ----------------------------
    foot_length = conf.lxn + conf.lxp   # foot size in the x-direction
    foot_width  = conf.lyn + conf.lyp   # foot size in the y-direciton
    nb_dt_per_step = int(round(conf.T_step/conf.dt_mpc))
    desired_walking_time  = conf.nb_steps * nb_dt_per_step
    planned_walking_time  = (2+conf.nb_steps) * nb_dt_per_step

    #pre-allocate memory
    com_state_x_total  = np.zeros((conf.total_simulations,
                                                    (desired_walking_time)+1,2))
    com_state_y_total  = np.zeros((conf.total_simulations,
                                                    (desired_walking_time)+1,2))
    cop_x_total = np.zeros((conf.total_simulations, (desired_walking_time)))
    cop_y_total = np.zeros((conf.total_simulations, (desired_walking_time)))
    PLOT_LEGEND = conf.plot_legend
    Sigma_w = np.array([conf.wc_ub/2.0, conf.wcdot_ub/2.0])

    #for sim_nb in range(conf.total_simulations):

    # CoM initial state: [x_0, xdot_0].T
    #                    [y_0, ydot_0].T
    # ----------------------------------
    x_0 = np.array([conf.foot_step_0[0], 0.0])
    y_0 = np.array([conf.foot_step_0[1], 0.0])

    step_width = 2*np.absolute(y_0[0])

    # compute CoP reference trajectory:
    # --------------------------------
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
    com_state_x   = np.zeros(((desired_walking_time)+1,2))
    com_state_y   = np.zeros(((desired_walking_time)+1,2))
    cop_x = np.zeros(((desired_walking_time)))
    cop_y = np.zeros(((desired_walking_time)))

    # set first values of the open and closed loop trajectories to be equal
    com_state_x[0,:] = x_0
    com_state_y[0,:] = y_0

    cop_ref = planned_Z_ref[0:conf.N,:]
    com_ref = planned_com_ref[0:conf.N]

    # construct your preview system:
    # ------------------------------
    [P_ps, P_vs, P_pu, P_vu] = motion_model.compute_recursive_matrices(\
                                            conf.dt_mpc, conf.g, conf.h, conf.N)
    for i in range(desired_walking_time):
        [Q, p_k] = cost_function.compute_objective_terms_box(conf.alpha,
                      conf.beta, conf.gamma,conf.T_step, nb_dt_per_step, conf.N,
                           conf.step_length, step_width, P_ps, P_pu, P_vs, P_vu,
                                                     x_0, y_0, cop_ref, com_ref)
        [A_zmp, b_zmp] = constraints.add_ZMP_constraints(conf.N, foot_length,
                                                  foot_width, cop_ref, x_0, y_0)
        if i > conf.N-1 and i < desired_walking_time-conf.N:
            [A_CoM, b_CoM] = constraints.add_CoM_constraints_box(conf.N, y_0,
                                                P_ps, P_pu, conf.com_constraint)
            A = np.concatenate((A_CoM, A_zmp), axis = 0)
            b = np.concatenate((b_CoM, b_zmp), axis = 0)
        else:
            [A_zmp, b_zmp] = constraints.add_ZMP_constraints(conf.N, foot_length,
                                                  foot_width, cop_ref, x_0, y_0)
            A = A_zmp
            b = b_zmp

        # call quadprog solver:
        # --------------------
        U = solve_qp(Q, -p_k, A.T, b)[0]

        # first element of the optimal control trajectory
        vx_MPC = U[0]
        vy_MPC = U[conf.N]
        # ----------------------------------------------------------------------
        #                   simulation (adding disturbances)
        # ----------------------------------------------------------------------
        # sample white gaussian from the bounded disturbance set W
        # set flag to True if you want to add truncated gaussian disturbance
        if i > conf.N-1:
            flag = True
            while flag == True:
                wc     = np.random.normal(0, Sigma_w[0])
                wc_dot = np.random.normal(0, Sigma_w[1])
                if  wc >= conf.wc_lb    and wc <= conf.wc_ub and \
                    wc_dot >= conf.wcdot_lb and wc_dot <= conf.wcdot_ub:
                        flag = False
                        wy_k = np.array([wc, wc_dot])
        else:
            wy_k = np.zeros(2)

        # save cop
        cop_x[i] = vx_MPC
        cop_y[i] = vy_MPC

        # update_closed-loop dynamics:
        A_d, B_d = motion_model.discrete_LIP_dynamics(conf.dt_mpc, conf.g,
                                                                         conf.h)
        x_cl_plus = np.dot(A_d, x_0) + np.dot(B_d, vx_MPC.squeeze())
        y_cl_plus = np.dot(A_d, y_0) + np.dot(B_d, vy_MPC.squeeze()) + wy_k
        x_0   = x_cl_plus
        y_0   = y_cl_plus

        com_state_x[i+1] = x_0
        com_state_y[i+1] = y_0

        # update reference trajectory
        cop_ref = planned_Z_ref[i+1:i+conf.N+1,:]
        com_ref = planned_com_ref[i+1:i+conf.N+1]

    np.savez(conf.DATA_FILE_LIPM, com_state_x = com_state_x,
              com_state_y = com_state_y, cop_y = cop_y, cop_ref = desired_Z_ref,
                                         cop_x = cop_x, foot_steps = foot_steps)
    # --------------------------------------------------------------------------
    #               visualize your closed-loop trajectories
    # --------------------------------------------------------------------------
    time = np.arange(0, round((desired_walking_time+1)*conf.dt_mpc, 2),
                                                                    conf.dt_mpc)
    desired_Z_ref = np.append([desired_Z_ref[0,:]], desired_Z_ref,axis=0)
    x_init = np.array([conf.foot_step_0[0], 0.0])
    y_init = np.array([conf.foot_step_0[1], 0.0])
    cop_y = np.append([conf.foot_step_0[1]], cop_y, axis=0)
    min_admissible_CoP = desired_Z_ref - np.tile([foot_length/2, foot_width/2],
                                                     (desired_walking_time+1,1))
    max_admissible_cop = desired_Z_ref + np.tile([foot_length/2, foot_width/2],
                                                     (desired_walking_time+1,1))
    CoM_constraint_vector = np.tile(conf.com_constraint,
                                                     (desired_walking_time+1,1))

    # time VS CoP and CoM in y:
    # -------------------------
    if PLOT_LEGEND:
        plot_utils.plot_y_box(PLOT_LEGEND, time, desired_walking_time,
        min_admissible_CoP, max_admissible_cop, cop_y, com_state_y, cop_ref,
                                                  CoM_constraint_vector, conf.N)
        PLOT_LEGEND = False
    else:
        plot_utils.plot_y_box(PLOT_LEGEND, time, desired_walking_time,
        min_admissible_CoP, max_admissible_cop, cop_y, com_state_y, cop_ref,
                                                  CoM_constraint_vector, conf.N)
    #plt.show()
