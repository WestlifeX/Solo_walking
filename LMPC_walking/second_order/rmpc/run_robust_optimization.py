# headers:
# -------
from numpy import array, dot, tile, arange, absolute, zeros, random, append, exp
from mrpi.polyhedron import polyhedron, plot_polygon_list
from second_order.motion_model import compute_recursive_disturbed_dynamics
from second_order.reference_trajectories import manual_foot_placement
from second_order.reference_trajectories import create_CoP_trajectory
from second_order.motion_model import compute_recursive_matrices
from second_order.motion_model import compute_recursive_dynamics
from second_order.cost_function import compute_objective_terms
from robust_constraints import add_CoP_robust_constraints
from robust_constraints import add_CoM_robust_constraints
from robust_constraints import compute_CoP_backoff_dead_beat
from robust_constraints import add_capturability_robust_terminal_constraints
from second_order.motion_model import discrete_LIP_dynamics
from second_order.stmpc.truncated_normal import sample_from_truncated_normal
from second_order.constraints import add_ZMP_constraints
from mrpi.mRPI_set import compute_mRPI
from second_order import plot_utils
from numpy import concatenate, eye
import matplotlib.pyplot as plt
from quadprog import solve_qp

# cost weights in the objective function:
# ---------------------------------------
alpha = 10 ** (-1)  # CoP error squared cost weight
beta = 10 ** (-4)  # CoM position error squared cost weight
gamma = 10 ** (-4)  # CoM velocity error squared cost weight

# Inverted pendulum parameters:
# ----------------------------
h = 0.80
g = 9.81
foot_length = 0.20
foot_width = 0.14
omega = 3.5  # sqrt(h/g)

# MPC Parameters:
# --------------
delta_t = 0.1  # MPC sampling period
step_time = 0.8  # step period
no_steps_per_T = int(step_time / delta_t)

# walking parameters:
# ------------------
step_length = 0.25  # fixed step length in the xz-plane
no_desired_steps = 3  # number of desired walking steps
desired_walking_time = no_desired_steps * no_steps_per_T  # number of desired walking intervals
N = desired_walking_time

# CoM initial state: [x, xdot, x_ddot].T
#                    [y, ydot, y_ddot].T
# --------------------------------------
x_init = array([0.0, 0.0])
y_init = array([-0.10, 0.0])

step_width = 2 * absolute(y_init[0])

# discrete dynamics for tracking control law
A_d, B_d = discrete_LIP_dynamics(delta_t, g, h)
B_d = B_d.reshape(B_d.shape[0], 1)

# 2D-bounded polyhdron additive disturbance set on the motion model
wc_lb = -0.002
wc_ub = 0.002
wcdot_lb = -0.02
wcdot_ub = 0.02
P_A = array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
P_b = array([wc_ub, wc_ub, wcdot_ub, wcdot_ub])
P_b = P_b.reshape([P_b.shape[0], 1])
W = polyhedron(P_A, P_b)

# dead-beat choice of LIPM pre-stabilizing gains
# ----------------------------------------------
k = exp(omega * delta_t) / ((exp(omega * delta_t)) - 1.0)
k_dead_beat = array([[k, k / omega]])

# compute constraints backoffs:
# ----------------------------
epsilon = 10.0 ** (-6)  # absolute of the mRPI outer-approximation error

# state back-off \Omega
Omega, Fs_list = compute_mRPI(epsilon, W, A_d, B_d, k_dead_beat)
Omega.compute_Hrep()
CoM_constraint = 0.05
CoM_backoff = 0.05 - 0.01062912

# control back-off KBW (exact)
KW = compute_CoP_backoff_dead_beat(k_dead_beat, W)

# compute CoP reference trajectory:
# --------------------------------
foot_step_0 = array([0.0, -0.10])  # initial foot step position in x-y

desiredFoot_steps = manual_foot_placement(foot_step_0, step_length,
                                          no_desired_steps)
desired_Z_ref = create_CoP_trajectory(no_desired_steps, desiredFoot_steps,
                                      desired_walking_time, no_steps_per_T)

# pre-allocate memory
X_ol = zeros((N + 1, 2))
Y_ol = zeros((N + 1, 2))
Z_x_ol = zeros((N + 1))
Z_y_ol = zeros((N + 1))

# initial states and controls
X_ol[0, :] = x_init
Y_ol[0, :] = y_init

Z_x_ol[0] = foot_step_0[0]
Z_y_ol[0] = foot_step_0[1]

# initialization
[P_ps, P_vs, P_pu, P_vu] = compute_recursive_matrices(delta_t, g, h, N)
x_0 = x_init
x_cl = x_init
y_0 = y_init
y_cl = y_init
# Sigma_w = array([wc_ub/3.0, wcdot_ub/3.0])
plot_legend = True
# -------------------------------------------------------------------------------
#                     MPC loop (every delta_t = 0.1 sec)
# ------------------------------------------------------------------------------

[Q, p_k] = compute_objective_terms(alpha, beta, gamma, step_time,
                                   no_steps_per_T, N, step_length, step_width,
                                   P_ps, P_pu, P_vs, P_vu, x_0, y_0, desired_Z_ref)

[A_zmp, b_zmp] = add_CoP_robust_constraints(N, foot_length, foot_width,
                                            desired_Z_ref, KW)

[A_CoM, b_CoM] = add_CoM_robust_constraints(N, x_0, y_0, P_ps,
                                            P_vs, P_pu, P_vu, CoM_backoff)

A = concatenate((A_CoM, A_zmp), axis=0)
b = concatenate((b_CoM, b_zmp), axis=0)

# solve the open-loop optimization problem
U_OL = solve_qp(Q, -p_k, A.T, b)[0]

# simulate your recursive nominal dynamics over the current horizon
[X_OL, Y_OL] = compute_recursive_dynamics(P_ps, P_vs, P_pu, P_vu, N,
                                          x_0, y_0, U_OL)
# -------------------------------------------------------------------------------
#                       save trajectories
# -------------------------------------------------------------------------------
# open-loop optimal CoP trajectories
Z_x_ol[1:N + 1] = U_OL[0:N]
Z_y_ol[1:N + 1] = U_OL[N:2 * N]

# open-loop optimal CoM trajectories
X_ol[1::, :] = X_OL
Y_ol[1::, :] = Y_OL

# ------------------------------------------------------------------------------
#                  visualize your closed-loop trajectories
# ------------------------------------------------------------------------------
reference_time_stamp = arange(0, round((desired_walking_time + 1) * delta_t, 2),
                              delta_t)

desired_Z_ref = append([desired_Z_ref[0, :]], desired_Z_ref, axis=0)
min_admissible_cop = desired_Z_ref - tile([foot_length / 2, foot_width / 2],
                                          (desired_walking_time + 1, 1))
max_admissible_cop = desired_Z_ref + tile([foot_length / 2, foot_width / 2],
                                          (desired_walking_time + 1, 1))

min_admissible_cop_back_off = desired_Z_ref - tile([foot_length / 2, foot_width / 2],
                                                   (desired_walking_time + 1, 1)) + tile([foot_length / 2, KW.b[0]],
                                                                                         (desired_walking_time + 1, 1))
max_admissible_cop_back_off = desired_Z_ref + tile([foot_length / 2, foot_width / 2],
                                                   (desired_walking_time + 1, 1)) - tile([foot_length / 2, KW.b[0]],
                                                                                         (desired_walking_time + 1, 1))

CoM_constraint = tile(CoM_constraint, (desired_walking_time + 1, 1))
CoM_back_off = tile(CoM_backoff, (desired_walking_time + 1, 1))

# time VS CoP and CoM in y: 'A.K.A what goes up must go down'
plt.figure()
plot_utils.plot_y_robust_MPC(plot_legend, reference_time_stamp, desired_walking_time,
                             min_admissible_cop, max_admissible_cop, min_admissible_cop_back_off,
                             max_admissible_cop_back_off, Z_y_ol, Y_ol, desired_Z_ref, CoM_constraint,
                             CoM_back_off)

plt.show()
