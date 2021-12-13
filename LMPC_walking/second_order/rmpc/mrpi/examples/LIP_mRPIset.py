# Headers:
# -------
from second_order.rmpc.mrpi.polyhedron import polyhedron, plot_polygon_list
from second_order.rmpc.mrpi.mRPI_set import compute_mRPI
from numpy import array, empty, vstack, eye, amax, exp, asarray, random, dot
from math import sqrt, cosh, sinh
import matplotlib.pyplot as plt

# LIPM discrete-time dynamics
delta_t = 0.1
g = 9.81
h = 0.8
omega   = 3.5
A_d = array([[cosh(omega*delta_t)  , (1/omega)*sinh(omega*delta_t)],
                [omega*sinh(omega*delta_t), cosh(omega*delta_t)]])

B_d = array([1 - cosh(omega*delta_t), -omega*sinh(omega*delta_t)])
B_d = B_d.reshape([B_d.shape[0],1])

# dead-beat choice of pre-stabilizing gains
#k = 57.6
#k_dead_beat = array([[k, k/omega]])
k = exp(omega*delta_t)/((exp(omega*delta_t))-1.0)
k_dead_beat = array([[k, k/omega]])

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

# 2D-bounded polyhdron additive disturbance set on the motion model
wc_lb    = -0.0016
wc_ub    =  0.0016
wcdot_lb = -0.016
wcdot_ub =  0.016
P_A = array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
P_b = array([wc_ub, wc_ub, wcdot_ub, wcdot_ub])
P_b = P_b.reshape([P_b.shape[0],1])
W = polyhedron(P_A, P_b)
A_k = A_d + dot(B_d, k_dead_beat)

#print 'W_vertices = ','\n', W_vertices
#print 'W_rays = ','\n', W_rays
# plot CoP disturbance set
#W.plot_polygon(title='CoP distrurbance set W')
#for i in range(array(W.vertices)):
#    print 'i = ', i
# compute the outer-epsilon approximation of the mRPI set
F_alpha_s, Fs_list = compute_mRPI(epsilon, W, A_d, B_d, k_dead_beat)
F_alpha_s.compute_Hrep()
#max = amax(F_alpha_s.vertices[:,0])

print("Omega vertices = ", F_alpha_s.vertices)
print("max = ", max)
print("Omega.A = ", F_alpha_s.A)
print("Omega.b = ", F_alpha_s.b)
# plot the outer-epsilon approximation of the mRPI set
#F_alpha_s.plot_polygon()
plot_polygon_list(Fs_list)
color = ['red', 'blue', 'orange', 'purple', 'fuchsia', 'gold']
for i in range(len(F_alpha_s.vertices)):
    co = color[i]
    v = F_alpha_s.vertices[i]
    plt.scatter(v[0], v[1], zorder=1, c='red', marker='x')
    for sim in range(50):
        #print sim
        wc = random.uniform(wc_lb, wc_ub)
        wc_dot = random.uniform(wcdot_lb, wcdot_ub)
        w = array([wc, wc_dot])
        v_plus = dot(A_k, v) + w
        plt.scatter(v_plus[0], v_plus[1], zorder=1, c='red')
        v = v_plus
# control back-off
KZ = F_alpha_s.affineMap(k_dead_beat)
print('KZ_vertices = ', KZ.vertices)
plt.figure()
KZ.plot_polygon(title='control back-off')
