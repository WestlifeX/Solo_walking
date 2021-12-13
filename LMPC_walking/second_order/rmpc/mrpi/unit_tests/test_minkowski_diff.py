# Headers:
# -------
from second_order.rmpc.mrpi.polyhedron import polyhedron, plot_polygon_list
from numpy.linalg import matrix_power
from math import sqrt, cosh, sinh
import matplotlib.pyplot as plt
from numpy import array

# construct a 2D box
P_A1 = array([[1.0, 0.0],[-1.0, 0.0],[0.0, 1.0],[0.0, -1.0]])
P_b1 = array([[5.0], [5.0], [5.0], [5.0]])
#print 'P_A1 = ',P_A1
#print 'P_b1 = ',P_b1
P1 = polyhedron(P_A1, P_b1)
P1.plot_polygon()


# construct a smaller 2D box
P_A2 = array([[1.0, 0.0],[-1.0, 0.0],[0.0, 1.0],[0.0, -1.0]])
P_b2 = array([1.0, 1.0, 1.0, 1.0])
P2 = polyhedron(P_A2, P_b2)

P3 = polyhedron.minkowskiDiff(P1, P2)
P3.plot_polygon()
