from numpy import zeros, repeat, array
#TODO
delta_t        = 0.1                            # sampling time interval
step_time      = 0.8                            # time needed for every step
no_steps_per_T = int(round(step_time/delta_t))
no_desired_steps      = 2
desired_walking_time  = no_desired_steps * no_steps_per_T
N = 16
m = 2

# foot placement matrices
U_c = repeat(array([1.0, 0.0]), N/2, axis=0)
U   = repeat(array([[0.0, 0.0], [1.0, 0.0]]), N/2, axis=0)

u_c_counter = (N//2)-1
u_counter = N-1
for i in range(desired_walking_time):
    print("i = ", i)
    print("u_c_counter = ", u_c_counter)
    print("u_counter = ", u_counter)
    print("U_c = ", U_c)
    print("U = ", U)
    # update foot placement matrices
    if u_c_counter == 0:
        U_c = repeat(array([1.0, 0.0]), N//2, axis=0)
        U   = repeat(array([[0.0, 0.0], [1.0, 0.0]]), N//2, axis=0)
        #reset counters
        u_c_counter = N//2
        u_counter = N
    else:
        U_c[u_c_counter] = 0.0
        U[u_c_counter,0] = 1.0
        U[u_counter, 0]  = 0.0
        U[u_counter, 1]  = 1.0
    u_c_counter = u_c_counter - 1
    u_counter   = u_counter - 1
