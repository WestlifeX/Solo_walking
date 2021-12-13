from second_order.plot_utils import plot_2D_covariance
from numpy import array, dot, zeros, tile, exp, sqrt
import matplotlib.pyplot as plt
from math import sqrt, cosh, sinh
from scipy.stats import norm

delta_t = 0.1
g = 9.81
h = 0.8
omega   = sqrt(g/h)
N = 16
A_d = array([[cosh(omega*delta_t)  , (1/omega)*sinh(omega*delta_t)],
                [omega*sinh(omega*delta_t), cosh(omega*delta_t)]])

B_d = array([1 - cosh(omega*delta_t), -omega*sinh(omega*delta_t)])
B_d = B_d.reshape([B_d.shape[0],1])

# dead-beat choice of pre-stabilizing gains
#k = 57.6
k = exp(omega*delta_t)/((exp(omega*delta_t))-1.0)
k_dead_beat = array([[k, k/omega]])
A_K = A_d + dot(B_d, k_dead_beat)
#print A_d, '\n'
#print B_d
cov_x_i = array([[0.0, 0.0], [0.0, 0.0]])
cov_w_i = array([[0.002**2, 0.0],[0.0, 0.02**2]])
for i in range(N):
    cov_x_i_plus_one = dot(dot(A_K, cov_x_i), A_K.T) + cov_w_i
    cov_x_i = cov_x_i_plus_one

    # plot_2D_covariance matrix propagation
    plot_2D_covariance(0, cov_x_i_plus_one)
    # compute back-off
    scale = sqrt(cov_x_i_plus_one[0,0])
    inv_CDF = norm.ppf(0.95, scale=scale) # in case uncertainties are not bounded
    #print scale
    print('eta_j = ', 0.05-inv_CDF, '\n')
plt.show()
