#    LMPC_walking is a python software implementation of one of the algorithms
#    refered in this paper https://hal.inria.fr/inria-00391408v2/document
#    Copyright (C) 2019 @ahmad gazar

#    LMPC_walking is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    LMPC_walking is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

# I've added This function
# It returns the position of the fron foot based on the hind foot position 
def compute_front_feet(foot_steps, duration, d1, d2, l):
    Foot_steps = np.zeros((foot_steps.shape[0], 4))
    j=0
    for i in range(int(foot_steps.shape[0]/duration)):
        if i%2 == 0:
            Foot_steps[j:j+duration] = np.concatenate((foot_steps[j], foot_steps[j] + l * d1))
        else:
            Foot_steps[j:j+duration] = np.concatenate((foot_steps[j], foot_steps[j] + l * d2))
        j += duration
    return Foot_steps

# Description:
# -----------
# this function implements a desired zig-zag fixed foot step plan located
# in the middle of the robot's foot starting with the right foot

# Parameters:
# ----------
#  foot_step_0 : initial foot step location
#            [foot_step_x0, foot_step_y0].T (2x1 numpy.array)
#  no_steps    : number of desired walking foot steps  (scalar)

# Returns:
# -------
#  Foot_steps = [Foot_steps_x, Foot_steps_y].T   foot steps locations
#                                                (no_steps x 2 numpy.array)

def manual_foot_placement(foot_step_0, fixed_step_x, no_steps):
    Foot_steps   = np.zeros((no_steps, 2))
    for i in range(Foot_steps.shape[0]):
        if i == 0:
            Foot_steps[i,:] = foot_step_0
        else:
            Foot_steps[i,0] = Foot_steps[i-1,0] + fixed_step_x
            Foot_steps[i,1] = -Foot_steps[i-1,1]
    return Foot_steps
    
# I've added This function
# same function as before to work with quadrupeds
def manual_foot_placement_quad(foot_step_zero, fixed_step_x, no_steps, d1, d2, l):

    Foot_steps_tmp = np.zeros((no_steps,2))

    for i in range(Foot_steps_tmp.shape[0]):
        if i == 0:
            Foot_steps_tmp[i, :] = foot_step_zero
        else:
            Foot_steps_tmp[i, 0] = Foot_steps_tmp[i-1,0] + fixed_step_x
            Foot_steps_tmp[i, 1] = -Foot_steps_tmp[i-1, 1]

    Foot_steps = compute_front_feet(Foot_steps_tmp, 1, d1, d2, l)
    return Foot_steps


# Description:
# -----------
# this function computes a CoP reference trajectory based on a desired
# fixed foot step plan, a desired foot step duration and a sampling time

# Parameters:
# ----------
#  no_steps      : number of desired walking foot steps  (scalar)
#  Foot_steps    := [Foot_steps_x, Foot_steps_y]   foot steps locations
#                                                  (no_steps x 2 numpy.array)
# walking_time   : desired walking time duration   (scalar)
# no_steps_per_T : step_duration/T  (scalar)

# Returns:
# -------
# Z_ref  := [z_ref_x_k             , z_ref_y_k              ]   CoP reference trajectory
#                   .              ,    .                       (walking_timex2 numpy.array)
#                   .              ,    .
#           [z_ref_x_k+walking_time, z_ref_y_k+walking_time]

def create_CoP_trajectory(no_steps, Foot_steps, walking_time, no_steps_per_T):
    Z_ref  = np.zeros((walking_time,2))
    j = 0
    for i in range (Foot_steps.shape[0]):
        Z_ref[j:j+no_steps_per_T, :] = Foot_steps[i,:]
        j = j + no_steps_per_T
    return Z_ref

# I've added This function
# same function as before to work with quadrupeds

def create_CoP_trajectory_quad(Foot_steps, walking_time, no_steps_per_T, d1, d2, l):
    Z_ref = np.zeros((walking_time, 2))
    Full_steps =  np.zeros((walking_time, 2))
    D = np.zeros((walking_time, 2))
    j=0

    for i in range(Foot_steps.shape[0]):
        if i%2 == 0:
            Z_ref[j:j+no_steps_per_T] = Foot_steps[i, :2] +  d1 * l/2
            Full_steps[j:j+no_steps_per_T] = Foot_steps[i, :2]
            D[j:j+no_steps_per_T] = d1
        else:
            Z_ref[j:j + no_steps_per_T] = Foot_steps[i, :2] + d2 * l / 2
            Full_steps[j:j + no_steps_per_T] = Foot_steps[i, :2]
            D[j:j + no_steps_per_T] = d2
        j = j + no_steps_per_T
    Full_steps = compute_front_feet(Full_steps, no_steps_per_T, d1, d2, l)
    return Z_ref, Full_steps,D



# Description:
# -----------
# this function computes a CoM rerference trajectory based on a desired
# fixed foot step plan, a desired foot step duration and a sampling time
# assumption: the first two steps are set to be constraint free

# Parameters:
# ----------
# no_steps      : number of desired walking foot steps  (scalar)
# no_steps_per_T : step_duration/T  (scalar)
# walking_time   : desired walking time duration   (scalar)
# max_admissible_com : (scalar)

# Returns:
# -------
# com_ref  := [com_ref_k           ] CoM reference trajectory
#                   .                (walking_timex1 numpy.array)
#                   .                  .
#           [com_ref_k+walking_time]
def create_CoM_trajectory(no_steps, no_steps_per_T, walking_time,
                                                            max_admissible_com):
    com_constraint_ref = np.zeros((walking_time))
    j = 0
    for i in range(no_steps):
        # set the first two steps to some non-active com constraint
        if i<2:
            if i%2 == 0: #assuming walking starts on left foot
                com_constraint_ref[j:j+no_steps_per_T] = -1.0
            else:
                com_constraint_ref[j:j+no_steps_per_T] = 1.0
        else:
            if i%2 == 0: #assuming walking starts on left foot
                com_constraint_ref[j:j+no_steps_per_T] = -max_admissible_com
            else:
                com_constraint_ref[j:j+no_steps_per_T] = max_admissible_com
        j = j + no_steps_per_T
    return com_constraint_ref

#------------------------------------------------------------------------------
# unit test: A.K.A red pill or blue pill
# ------------------------------------------------------------------------------
if __name__=='__main__':
    import numpy.random as random
    print(' red pill or blue pill ! '.center(60,'*'))
    dt = 0.1
    T_step = 0.8
    no_steps_per_T = int(round(T_step/dt))
    #print(no_steps_per_T)
    no_steps = 4
    walking_time = no_steps * no_steps_per_T
    max_admissible_com = 0.05
    com_ref = create_CoM_trajectory(no_steps, no_steps_per_T,
                                               walking_time, max_admissible_com)
    #print(com_ref)
