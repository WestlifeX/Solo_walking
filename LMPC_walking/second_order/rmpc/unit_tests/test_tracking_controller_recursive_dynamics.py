from second_order.motion_model import discrete_LIP_dynamics
from numpy import array, zeros, dot
tau = 0.005
h   = 0.80
g   = 9.81
delta_t = 0.1
A_d, B_d = discrete_LIP_dynamics(tau, g, h)
ZY_MPC_plus = zeros((int(delta_t/tau),2))

# first optimal control and state from MPC
# MPC loop 1
#y_0 = array([-0.09, 0.0])
#vy_MPC = -0.10910453235859718

# MPC loop 2
y_0 =  array([-0.08881663,  0.02390866])
vy_MPC =  -0.10478050440242377


#zy_MPC =  [-0.08538778  0.04536784]

for j in range(int(delta_t/tau)):
    ZY_MPC_plus[j,:] = dot(A_d, y_0) + dot(B_d, vy_MPC)
    y_0       = ZY_MPC_plus[j,:]

print('ZY_MPC_plus = ', ZY_MPC_plus)
print('ZY_MPC_plus = ',ZY_MPC_plus[ZY_MPC_plus.shape[0]-1,:])
