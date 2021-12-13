# Headers:
# -------
from second_order.rmpc.mrpi.polyhedron import polyhedron
from numpy import array

# 1D-polyhdron disturbance set on the control input
P_A = array([[1.0], [-1.0]])
P_b = array([0.05, 0.05])
P_b = P_b.reshape([P_b.shape[0],1])

# construct an H-Polyhedron and convert to V-polyhedron
W = polyhedron(P_A, P_b)

W.compute_Vrep()
print('W_A =', W.A)
print('W_b =', W.b)
print('W_Aeq =', W.Aeq)
print('W_beq =', W.beq)
print('W_vertices =', W.vertices)
print('W_rays =', W.rays)
print('W_dim =', W.dim)
print('W_hasVrep =', W.hasVrep)
print('W_hasHrep = ', W.hasHrep,'\n')

# construct a V-polyhedron then convert to H-polyhedron
v = array([[0.05], [-0.05]])
W = polyhedron([], [],[],[],v,[])
W.compute_Hrep()
W.minVrep()
print('W_A =', W.A)
print('W_b =', W.b)
print('W_Aeq =', W.Aeq)
print('W_beq =', W.beq)
print('W_vertices =', W.vertices)
print('W_rays =', W.rays)
print('W_dim =', W.dim)
print('W_hasVrep =', W.hasVrep)
print('W_hasHrep = ', W.hasHrep, '\n')
