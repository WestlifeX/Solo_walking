# Headers:
# --------
from numpy import dot, ones, zeros, max, identity, array
from numpy.linalg import matrix_power
from scipy.spatial import ConvexHull
from cvxopt import matrix, solvers
from .polyhedron import polyhedron
import math
import cdd

"""
Description:
------------
computation of an invariant outer-epsilon approximation of the minimal Robust
Positively Invariant Set (mRPI), see Rakovic et al., Invariant Approximations
of the Minimal Robust invariant Set.

Parameters:
-----------
epsilon : outer approximation error threshold of the minimal robust positive
          invariant set (mRPI) specified as a priori
W       : polyhedron object disturbance set
A, B    : (mxn), (nx1) numpy arrays
          linear system matrices (x_dot = Ax + Bu)
K       : (1xn) numpy array
          prestabilizing gains

Returns:
--------
F_alpha_s: polyhedron object of outer-epsilon approximation of the mRPI set
Fs_polyhedron_list: list of polyhedron objects [F_0, ....,F_s-1}
"""
def compute_mRPI(epsilon, W, A, B, K):

    #initialization
    alpha      = 0.0        # ideally start with 0
    logicalVar = 1.0
    s          = 0

    # closed-loop dynamics
    A_K = A + dot(B, K)
    if not W.hasHrep:
        W.compute_Hrep()
    f_i   = W.A.T;
    g_i   = W.b;
    I_max = W.A.shape[0]
    n     = W.dim
    alpha_candidate = zeros((I_max))
    while logicalVar == 1.0:
        s = s + 1
        for k in range(I_max):
            a = dot(matrix_power(A_K, s).T, f_i[:,k])
            tmp_extreme, supp = W.compute_extreme(a)
            alpha_candidate[k] = supp/g_i[k]
        alpha = max(alpha_candidate) # (\alpha_0^s, theorem 1)
        e = identity((n))
        sum_vec = zeros((n,2))
        for k in range(s):
            for j in range(n):
                e_j = e[:,j]
                a = dot((matrix_power(A_K, s-1).T), e_j)
                tmp_extreme, supp = W.compute_extreme(a)
                sum_vec[j,0] = sum_vec[j,0] + supp
                tmp_extreme, supp = W.compute_extreme(-a)
                sum_vec[j,1] = sum_vec[j,1] + supp
        Ms = max(max(sum_vec))
        stop = epsilon/(epsilon+Ms)
        # interrupt criterion (error bound condition of theorem 3)
        if alpha <= stop:
            logicalVar = 0

    Fs_polyhedron_list = []
    print('alpha = ', alpha, end=' ')
    print('s = ', s)
    Fs = polyhedron([], [], [], [], array([[0.0, 0.0]]))
    for i in range(s):
        Fs_curr = W.affineMap(matrix_power(A_K, i))
        Fs      = polyhedron.minkowskiSum(Fs, Fs_curr)
        if i > 0: # convex hull can not be constructed with less than 3 points
            Fs.minVrep()
        Fs_polyhedron_list.append(Fs)
    F_alpha_s = Fs.scale(1.0/(1.0-alpha))
    Fs_polyhedron_list.append(F_alpha_s)
    return F_alpha_s, Fs_polyhedron_list
