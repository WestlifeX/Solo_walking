# Headers:
# --------
from second_order.rmpc.mrpi.polyhedron import polyhedron
from numpy import array, cos, sin
import matplotlib.pyplot as plt

P_A = array([[1.0, 0.0],[-1.0, 0.0],[0.0, 1.0],[0.0, -1.0]])
P_b = array([1.0, 1.0, 1.0, 1.0])

# construct a 2D box
box  = polyhedron(P_A, P_b)
box.plot_polygon()

# rotation matrix of 45 degrees
rot_mat = array([[cos(45), -sin(45)],
                     [sin(45), cos(45)]])
# apply affine Map
rotated_box = box.affineMap(rot_mat)
rotated_box.plot_polygon(title='rotation of 2D box by 45 degrees')
