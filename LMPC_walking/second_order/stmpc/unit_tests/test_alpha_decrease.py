from numpy import array, dot, exp, sqrt, random, zeros, diag, abs
from second_order.motion_model import discrete_LIP_dynamics
from numpy.linalg import eig, norm, matrix_power
from math import sqrt, cosh, sinh

delta_t = 0.1
g = 9.81
h = 0.8
omega = sqrt(g/h)
k = exp(omega*delta_t)/((exp(omega*delta_t))-1.0)
k_dead_beat = array([[k, k/omega]])
A_d, B_d = discrete_LIP_dynamics(delta_t, g, h)
B_d = B_d.reshape([B_d.shape[0],1])
A_K = A_d + dot(B_d, k_dead_beat)
eig_values, eig_vectors = eig(A_K)
#print('eigen values =  ', eig_values, '\n')
#print('eigen vectors =  ', eig_vectors, '\n')
#print ('norm of eigen values =  ', norm(eig_values), '\n')
schur = False
no_random_matrices = 500
schur_stable_matrices = zeros((no_random_matrices,2,2))

# generate stable matrices that resembles A_K
for i in range(no_random_matrices):
    schur = False
    while(not schur):
        random_matrix =  random.rand(2,2)
        eig_values, eig_vectors = eig(random_matrix)
        # make sure that the matrix is schur stable
        if norm(eig_values) < 1:
            #print(norm(eig_values))
            assert (norm(eig_values) < 1)
            schur_stable_matrices[i,:,:] = random_matrix
            schur = True

wc_ub = 0.002
wcdot_ub =  0.02
sum_of_squares = zeros(no_random_matrices,)
squares_of_sums = zeros(no_random_matrices)
sum_of_squares_plus_one = zeros(no_random_matrices,)
squares_of_sums_plus_one = zeros(no_random_matrices)
alpha_i = zeros(no_random_matrices)
alpha_i_plus_one = zeros(no_random_matrices)

sigma_square_vector = array([(wc_ub/2.0)**2, (wcdot_ub/2.0)**2])
sigma_vector = array([(wc_ub/2.0), (wcdot_ub/2.0)])
sigma_square_vector.reshape([2,1])
sigma_vector.reshape([2,1])
H_j = zeros([1, 2])
H_j[0,0] = 1
#try different values different horizons
N = 16
for k in range(no_random_matrices):
    for j in range(N):
        b_j = dot(H_j, matrix_power(schur_stable_matrices[k,:,:], j))
        #b_j = dot(H_j, matrix_power(A_K, j))
        sum_of_squares[k]+= dot(dot(b_j, diag(b_j.squeeze())),
                                                            sigma_square_vector)
        squares_of_sums[k]+= dot(abs(b_j), sigma_vector)
    squares_of_sums[k] = squares_of_sums[k]**2
    alpha_i[k] = sum_of_squares[k]/ squares_of_sums[k]

for i in range(no_random_matrices):
    for j in range(N+1):
        b_j = dot(H_j, matrix_power(schur_stable_matrices[i,:,:], j))
        #b_j = dot(H_j, matrix_power(A_K, j))
        sum_of_squares_plus_one[i]+= dot(dot(b_j, diag(b_j.squeeze())),
                                                            sigma_square_vector)
        squares_of_sums_plus_one[i]+= dot(abs(b_j), sigma_vector)
    squares_of_sums_plus_one[i] = squares_of_sums_plus_one[i]**2
    alpha_i_plus_one[i] = sum_of_squares_plus_one[i]/squares_of_sums_plus_one[i]

print((' Test that alpha_i decreases along the horizon '.center(60,'*')))
for z in range(no_random_matrices):
    print('Is alpha_i_plus_one < alpha_i  : ',(alpha_i[z]-alpha_i_plus_one[z])>=10**-20)
    if not (alpha_i[z]>alpha_i_plus_one[z]):
        print (alpha_i[z], ' : ', alpha_i_plus_one[z])
