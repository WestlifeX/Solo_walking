"""
Description:
------------
this unit test is for testing the recursive affine map of the mRPI set of LIPM
by comparing the results with MPT3.0 toolbox in matlab
"""
# Headers:
# -------
from second_order.rmpc.mrpi.polyhedron import polyhedron, plot_polygon_list
from second_order.rmpc.mrpi.mRPI_set import compute_mRPI
from numpy import array, empty, vstack, dot
from numpy.linalg import matrix_power
from math import sqrt, cosh, sinh
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

# closed-loop dynamics
A_K = A_d + dot(B_d, k_dead_beat)

# algorithm initialization
alpha      = 0.0
logicalVar = 1.0
epsilon    = 10.0**(-6)    # error threshold

# 1D-polyhdron disturbance set on the control input
P_A = array([[1.0], [-1.0]])
P_b = array([0.05, 0.05])
P_b = P_b.reshape([P_b.shape[0],1])

W_bounds = polyhedron(P_A, P_b)
W = W_bounds.affineMap(B_d)
W.compute_Hrep()
s = 288
alpha = 0.0131

# construct an empty polyhedron with zero interior
Fs_polyhedron_list = []
print('alpha = ', alpha, end=' ')
print('s = ', s)
Fs = polyhedron([], [], [], [], array([[0.0, 0.0]]))
for i in range(s):
    Fs_curr = W.affineMap(matrix_power(A_K, i))
    Fs      = polyhedron.minkowskiSum(Fs, Fs_curr)
    if i > 0: # convex hull can not be constructed with less than 3 points
        Fs.minVrep()
    print('Fs_vertices_curr = ', Fs_curr.vertices)
    print('Fs_vertices = ', Fs.vertices)
    Fs_polyhedron_list.append(Fs)
F_alpha_s = Fs.scale(1.0/(1.0-alpha))
Fs_polyhedron_list.append(F_alpha_s)
