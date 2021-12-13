# Headers:
# -------
from second_order.rmpc.mrpi.polyhedron import polyhedron, plot_polygon_list
from second_order.rmpc.mrpi.mRPI_set import compute_mRPI
import matplotlib.pyplot as plt
from numpy import array

#initialization
alpha      = 0.0
logicalVar = 1.0
epsilon    = 5.0*10.0**(-5)    # error threshold as in the paper
s          = 0

# Dynamics
A  = array([[1,1],[0,1]])
B  = array([[1],[1]])

# Disturbance set
P_A = array([[1.0, 0.0],[-1.0, 0.0],[0.0, 1.0],[0.0, -1.0]])
P_b = array([1.0, 1.0, 1.0, 1.0])
W   = polyhedron(P_A, P_b)


# specific choice of optimal control gains in the paper
K = -array([[1.17, 1.03]])

# compute the outer-epsilon approximation of the mRPI set
F_alpha_s, Fs_list = compute_mRPI(epsilon, W, A, B, K)

# plot the outer-epsilon approximation of the mRPI set
F_alpha_s.plot_polygon()
#plot_polygon_list(Fs_list)
