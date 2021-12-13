from example_robot_data.robots_loader import getModelPath
import pinocchio as pin
import numpy as np
import os

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 100

DATA_FILE_LIPM_MPC = 'talos_walking_mpc_lipm.npz'
DATA_FILE_TSID_MPC = 'talos_walking_mpc_tsid.npz'

DATA_FILE_LIPM_RMPC = 'talos_walking_rmpc_lipm.npz'
DATA_FILE_TSID_RMPC = 'talos_walking_rmpc_tsid.npz'

DATA_FILE_LIPM_SMPC = 'talos_walking_smpc_lipm.npz'
DATA_FILE_TSID_SMPC = 'talos_walking_smpc_tsid.npz'

# Model path
# ----------
filename = str(os.path.dirname(os.path.abspath(__file__)))
urdf = '/talos_data/robots/talos_reduced.urdf'
modelPath = getModelPath(urdf)
urdf = modelPath + urdf
srdf = modelPath + '/talos_data/srdf/talos.srdf'
path = os.path.join(modelPath, '../..')

# robot simplified parameters:
# ----------------------------
h             = 0.88         # fixed CoM height
g             = 9.81         # norm of the gravity vector
omega         = np.sqrt(g/h) # natural frequency
foot_scaling  = 1.
lxp = foot_scaling*0.10      # foot length in positive x direction
lxn = foot_scaling*0.05      # foot length in negative x direction
lyp = foot_scaling*0.05      # foot length in positive y direction
lyn = foot_scaling*0.05      # foot length in negative y direction
lz = 0.0                     # foot sole height with respect to ankle joint

# walking parameters:
# -------------------
foot_step_0   = np.array([0.0, -0.085])   # initial foot step position in x-y
T_step        = 0.8       # time needed for every step
step_length   = 0.2       # fixed step length
step_width    = np.abs(foot_step_0[1])
step_height   = 0.05      # fixed step height
nb_steps      = 8         # number of desired walking steps

# MPC parameters:
# ---------------
alpha          = 10**(-1)  # CoP error squared cost weight
beta           = 10**(-4)  # CoM position error squared cost weight
gamma          = 10**(-4)  # CoM velocity error squared cost weight
com_constraint = 0.04
beta_x         = 0.99
beta_u         = 0.50
dt_mpc         = 0.1       # sampling time interval
N              = 16        # horizon length

# disturbance set
# ----------------
epsilon  = 10.0**(-6)  # outer-approximation error of the mRPI set
wc_lb    = -0.0016
wc_ub    =  0.0016
wcdot_lb = -0.016
wcdot_ub =  0.016

# TSID parameters
# ---------------
# time parameters
dt     = 0.002                              # controller time step
T_pre  = 2.0                                # time before starting to walk
T_post = 2.0                                # time after walking
mu     = 0.3                                # friction coefficient
fMin   = 0.0                                # minimum normal force
fMax   = 1e6                                # maximum normal force
rf_frame_name = "leg_right_sole_fix_joint"  # right foot frame name
lf_frame_name = "leg_left_sole_fix_joint"   # left foot frame name
waist_frame_name = 'base_link'              #'torso_2_link'
contactNormal = np.matrix([0., 0., 1.]).T   # direction of the contact normal

# task weights
w_com = 10e5               # weight of center of mass task
w_foot = 1e0               # weight of the foot motion task
w_contact = 1e2            # weight of the foot in contact
w_waist = 3e0              # weight of the waist task
w_posture = 1e-1           # weight of joint posture task
w_forceRef = 1e-5          # weight of force regularization task
w_torque_bounds = 0.0      # weight of the torque bounds
w_joint_bounds = 0.0
tau_max_scaling = 1.45     # scaling factor of torque bounds
v_max_scaling = 0.8

kp_contact = 10.0          # proportional gain of contact constraint
kp_foot = 10.0             # proportional gain of contact constraint
kp_com = 10.0              # proportional gain of center of mass task
kp_posture = 10.0          # proportional gain of joint posture task
kp_waist = 10.0
nv = 38
masks_posture = np.ones(nv-6)

# plotting parameters
# -------------------
PLOT_LIPM  = True
PLOT_TSID  = False
viewer = pin.visualize.GepettoVisualizer
PRINT_N = 500        # print every PRINT_N time steps
DISPLAY_N = 20       # update robot configuration rate
CAMERA_TRANSFORM = [3.578777551651001, 1.2937744855880737, 0.8885031342506409,
0.4116811454296112, 0.5468055009841919, 0.6109083890914917, 0.3978860676288605]
