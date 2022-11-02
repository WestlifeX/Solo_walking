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
from second_order.reference_trajectories import manual_foot_placement
from second_order.reference_trajectories import create_CoP_trajectory
from second_order.motion_model import compute_recursive_matrices
from second_order.motion_model import compute_recursive_dynamics
from second_order.cost_function import compute_objective_terms
from second_order.constraints import add_terminal_constraints
from second_order.motion_model import discrete_LIP_dynamics
from second_order.constraints import add_ZMP_constraints
from second_order.constraints import add_CoM_constraints
from second_order import plot_utils
import matplotlib.pyplot as plt
from quadprog import solve_qp
import numpy as np

# cost weights in the objective function:
# ---------------------------------------
alpha       = 10**(-1)   # CoP error squared cost weight
beta        = 10**(-4) # CoM position error squared cost weight
gamma       = 10**(-4)   # CoM velocity error squared cost weight

# Inverted pendulum parameters:
# ----------------------------
h           = 0.80   # fixed CoM height (assuming walking on a flat terrain)
g           = 9.81   # norm of the gravity vector
foot_length = 0.20   # foot size in the x-direction
foot_width  = 0.10   # foot size in the y-direciton

# MPC Parameters:
# --------------
delta_t               = 0.1                         # sampling time interval
step_time             = 0.8                         # time needed for every step
no_steps_per_T        = int(round(step_time/delta_t))

# walking parameters:
# ------------------
step_length           = 0.21                  # fixed step length in the xz-plane
no_desired_steps      = 6                     # number of desired walking steps
desired_walking_time  = no_desired_steps * no_steps_per_T  # number of desired walking intervals
N                     = desired_walking_time  # preceding horizon

# CoM initial state: [x_0, xdot_0].T
#                    [y_0, ydot_0].T
# ----------------------------------
x_0 = np.array([0.0, 0.0])
y_0 = np.array([-0.09, 0.0])

step_width = 2*np.absolute(y_0[0])

# compute CoP reference trajectory:
# --------------------------------
foot_step_0   = np.array([0.0, -0.09])   # initial foot step position in x-y

desiredFoot_steps  = manual_foot_placement(foot_step_0,
                                                step_length, no_desired_steps)
desired_Z_ref = create_CoP_trajectory(no_desired_steps,
                    desiredFoot_steps, desired_walking_time, no_steps_per_T)

# used in case you want to have terminal constraints
# --------------------------------------------------
x_terminal = np.array([desired_Z_ref[N-1, 0], 0.0])  # CoM terminal constraint in x : [x, xdot].T
y_terminal = np.array([desired_Z_ref[N-1, 1], 0.0])  # CoM terminal constraint in y : [y, ydot].T
no_terminal_constraints = 4
terminal_index = N-1

com_constraint = 0.05
CoM_constraint_vector = np.tile(com_constraint, (desired_walking_time+1,1))
plot_legend = True
# construct your preview system: 'Go pokemon !'
# --------------------------------------------
[P_ps, P_vs, P_pu, P_vu] = compute_recursive_matrices(delta_t, g,  h, N)
[Q, p_k] = compute_objective_terms(alpha, beta, gamma, step_time, no_steps_per_T,
    N, step_length, step_width, P_ps, P_pu, P_vs, P_vu, x_0, y_0, desired_Z_ref)
[A_zmp, b_zmp] = add_ZMP_constraints(N, foot_length, foot_width, desired_Z_ref,
                                     x_0, y_0)
[A_com , b_com] = add_CoM_constraints(N, y_0, P_ps, P_pu, com_constraint)

# used in case you want to add both terminal add_ZMP_constraints
# --------------------------------------------------------------
[A_terminal, b_terminal] = add_terminal_constraints(N, terminal_index, x_0, y_0,
                            x_terminal, y_terminal, P_ps, P_vs, P_pu, P_vu)
A = np.concatenate((A_zmp, A_com), axis = 0)
b = np.concatenate((b_zmp, b_com), axis = 0)

# call quadprog solver:
# --------------------
U =  solve_qp(Q, -p_k, A.T, b)[0]
#U = solve_qp(Q, -p_k, A_zmp.T, b_zmp)[0]
Z_x_total = U[0:N]
Z_y_total = U[N:2*N]

# Trajectory optimization: (based on the initial state x_hat_0, y_hat_0)
# -------------------------------------------------------------------------
[X_total, Y_total] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu, N, x_0,
                                                y_0, U)
# ------------------------------------------------------------------------------
#               visualize  open-loop trajectoties
# ------------------------------------------------------------------------------

time = np.arange(0, round((desired_walking_time+1)*delta_t, 2), delta_t)
desired_Z_ref = np.append([desired_Z_ref[0,:]], desired_Z_ref,axis=0)

# append initial states to the solution
Z_x_total = np.append([desired_Z_ref[0,0]], Z_x_total,axis=0)
Z_y_total = np.append([desired_Z_ref[0,1]], Z_y_total,axis=0)
X_total = np.append([x_0], X_total,axis=0)
Y_total = np.append([y_0], Y_total,axis=0)

min_admissible_CoP = desired_Z_ref - np.tile([foot_length/2, foot_width/2],
                     (desired_walking_time+1,1))
max_admissible_cop = desired_Z_ref + np.tile([foot_length/2, foot_width/2],
                     (desired_walking_time+1,1))

# time vs CoP and CoM in x:
# -------------------------
plot_utils.plot_x(True, time, desired_walking_time, min_admissible_CoP,
                  max_admissible_cop, Z_x_total, X_total, desired_Z_ref)

# time VS CoP and CoM in y:
# -------------------------
plot_utils.plot_y(True, time, desired_walking_time, min_admissible_CoP,
max_admissible_cop, Z_y_total, Y_total, desired_Z_ref, CoM_constraint_vector)

# plot CoP, CoM in x Vs Cop, CoM in y:
# -----------------------------------
plot_utils.plot_xy(time, desired_walking_time, foot_length, foot_width,
                   desired_Z_ref, Z_x_total, Z_y_total, X_total, Y_total)
plt.show()
