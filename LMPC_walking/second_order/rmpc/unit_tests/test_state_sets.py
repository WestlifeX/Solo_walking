from second_order.rmpc.mrpi.polyhedron import polyhedron, plot_polygon_list
from second_order.rmpc.mrpi.mRPI_set import compute_mRPI
from numpy import array, empty, vstack, matrix
from math import sqrt, cosh, sinh
from scipy.linalg import inv, solve_discrete_are
import matplotlib.pyplot as plt

# LIPM discrete-time dynamics
delta_t = 0.005
g = 9.81
h = 0.8
omega   = 3.5
A_d = array([[cosh(omega*delta_t)  , (1/omega)*sinh(omega*delta_t)],
                [omega*sinh(omega*delta_t), cosh(omega*delta_t)]])

B_d = array([1 - cosh(omega*delta_t), -omega*sinh(omega*delta_t)])
B_d = B_d.reshape([B_d.shape[0],1])

# dead-beat choice of pre-stabilizing gains
k = 57.6
k_dead_beat = array([[k, k/omega]])

# LQR choice gains
Q = array([[1.0, 0.0],[0.0,1.0]])
R = 10**(-3)
X = matrix(solve_discrete_are(A_d, B_d, Q, R))
K_lqr = array(matrix(inv(B_d.T*X*B_d+R)*(B_d.T*X*A_d)))

# algorithm initialization
alpha      = 0.0
logicalVar = 1.0
epsilon    = 10.0**(-6)    # error threshold

# 1D-polyhdron disturbance set on the control input

P_A = array([[1.0], [-1.0]])
P_b = array([0.05, 0.05])
P_b = P_b.reshape([P_b.shape[0],1])
W_bounds = polyhedron(P_A, P_b)
#W = W_bounds.affineMap(A_d)


# 2D-polyhdron disturbance set on the control input
P_A = array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
P_b = array([0.002, 0.002, 0.02, 0.02])
P_b = P_b.reshape([P_b.shape[0],1])
W_bounds = polyhedron(P_A, P_b)
W = W_bounds.affineMap(A_d)
#print 'W_vertices = ','\n', W_vertices
#print 'W_rays = ','\n', W_rays
# plot CoP disturbance set
#W.plot_polygon(title='CoP distrurbance set W')
KZ = W.affineMap(k_dead_beat)
KZ.plot_polygon()

# compute the outer-epsilon approximation of the mRPI set
F_alpha_s, Fs_list = compute_mRPI(epsilon, W, A_d, B_d, k_dead_beat)
F_alpha_s.compute_Hrep()
print(F_alpha_s.A)
F_alpha_s.plot_polygon()
P_A = array([[1.0, 0.0],[-1.0, 0.0],[0.0, 1.0],[0.0, -1.0]])
P_b = array([0.10, 1.0, 3.0, 3.0])
X = polyhedron(P_A, P_b)
Y = polyhedron.minkowskiDiff(X , F_alpha_s)
Y.compute_Hrep()
print(Y.A)
print(Y.b)
Y.plot_polygon()
"""
P_A = array([[1.0, 0.0],[-1.0, 0.0], [1.0, 0.0],[-1.0, 0.0],
            [0.0, 1.0],[0.0, -1.0], [0.0, 1.0],[0.0, -1.0]])

P_b = array([0.04, 0.04, 0.0391, 0.0391, 3.0, 3.0, 2.9938, 2.9938])
X = polyhedron(P_A, P_b)
X.plot_polygon()
"""
