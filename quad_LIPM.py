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
from quadprog import solve_qp
import LMPC_walking_quadruped.second_order.reference_trajectories as reference_trajectories
import LMPC_walking_quadruped.second_order.motion_model as motion_model
import LMPC_walking_quadruped.second_order.cost_function as cost_function
import LMPC_walking_quadruped.second_order.constraints as constraints
import LMPC_walking_quadruped.second_order.plot_utils as plot_utils
import matplotlib.pyplot as plt
import matplotlib

import conf as conf

#matplotlib.use('TkAgg')


# Simulation parameters
nb_steps = conf.nb_steps
nb_dt_per_step        = conf.nb_dt_per_step
N  = conf.N  # number of desired walking intervals

# Inverted pendulum parameters:
# ----------------------------
# Walking parameters
fixed_step_x = conf.fixed_step_x
stride_length = conf.stride_length
d1 = conf.d1
d2 = conf.d2
l = conf.l


foot_step_zero = conf.foot_step_zero
Foot_steps = reference_trajectories.manual_foot_placement_quad(foot_step_zero, fixed_step_x, nb_steps, d1, d2, l)

Foot_steps[1:,0::2] -= conf.fixed_step_x
# compute CoP reference trajectory:
# --------------------------------
[CoP_ref, full_steps, D] = reference_trajectories.create_CoP_trajectory_quad(Foot_steps, N, nb_dt_per_step, d1, d2, l)

# CoM initial state: [x_0, xdot_0].T
#                    [y_0, ydot_0].T
# ----------------------------------
x_0 = np.array([Foot_steps[0,0] + l/2*d1[0], 0])
y_0 = np.array([Foot_steps[0,1] + l/2*d1[1], 0])

[P_ps, P_vs, P_pu, P_vu] = motion_model.compute_recursive_matrices(conf.dt_mpc, conf.g,
                                                                   conf.h, N)

[Q, p_k] = cost_function.compute_objective_terms_quad(conf.alpha, conf.beta, conf.gamma,
                        conf.T_step, nb_dt_per_step, N, stride_length, fixed_step_x,
                        P_ps, P_pu, P_vs, P_vu, x_0, y_0, CoP_ref)

[A_zmp, b_zmp] = constraints.add_ZMP_constraints_quad(N, full_steps, D, l)

x_terminal = np.array([CoP_ref[N-1, 0], 0.0])  # CoM terminal constraint in x : [x, xdot].T
y_terminal = np.array([CoP_ref[N-1, 1], 0.0])  # CoM terminal constraint in y : [y, ydot].T
nb_terminal_constraints = 4
terminal_index = N-1

[A_terminal, b_terminal] = constraints.add_terminal_constraints_quad(N,
                            terminal_index, x_0, y_0, x_terminal,
                             y_terminal, P_ps, P_vs, P_pu, P_vu, full_steps,D)

A = np.concatenate((A_terminal, A_zmp), axis = 0)
b = np.concatenate((b_terminal, b_zmp), axis = 0)



U = solve_qp(Q, -p_k, A.T, b, N+ nb_terminal_constraints)[0]

cop_x = U[0:N]
alpha = U[N:2*N]

cop_y = full_steps[:,1] + alpha*D[:,1] # Get the cop position on y by the relation y = Phind + alpha * D
my_U = np.concatenate((cop_x, cop_y), axis = 0)


[com_state_x, com_state_y] = motion_model.compute_recursive_dynamics(P_ps, P_vs, P_pu,
                                                             P_vu, N, x_0,
                                                             y_0, my_U)


# Plot stuff
plt.scatter(Foot_steps[:,0], Foot_steps[:,1], marker="8", c="chocolate", s = 200)
plt.scatter(Foot_steps[:,2], Foot_steps[:,3], marker="8", c="cornflowerblue", s = 200)
plt.plot(CoP_ref[:,0], CoP_ref[:,1])
plt.scatter(cop_x, cop_y, c='green')
plt.plot(com_state_x[:,0], com_state_y[:,0], c='coral')
plt.legend(["CoP Ref", "Computed CoP", "Computed CoM"])
plt.show()

com_state_x = np.vstack((x_0, com_state_x))
com_state_y = np.vstack((y_0, com_state_y))

# Saving trajectories
np.savez(conf.DATA_FILE_LIPM,
         com_state_x=com_state_x, com_state_y=com_state_y, cop_ref=CoP_ref,
         cop_x=cop_x, cop_y=cop_y, foot_steps=Foot_steps, foot_steps_trj= full_steps)
