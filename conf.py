import numpy as np
import pinocchio as pin

DATA_FILE_LIPM = 'quad_walking_traj_lipm.npz'
DATA_FILE_TSID = 'quad_walking_traj_TSID.npz'

path = "/opt/openrobots/share/example-robot-data/robots/solo_description/robots"
urdf = path + "/solo12.urdf"

# Dog parameters
vector1 = np.array([3.89200000e-01, 1.46950000e-01 * 2])
vector2 = np.array([3.89200000e-01, -1.46950000e-01*2])
d1= vector1/ np.linalg.norm(vector1)
d2= vector2/ np.linalg.norm(vector2)
l = np.sqrt(vector1[0]**2 + vector1[1]**2)
foot_step_zero = np.array([0.0, -1.46950000e-01])


# configuration for LIPM trajectory optimization
# ----------------------------------------------
alpha       = 10**(2)   # CoP error squared cost weight
beta        = 0         # CoM position error squared cost weight
gamma       = 10**(-1)  # CoM velocity error squared cost weight
h           = 0.23       # fixed CoM height
g           = 9.81      # norm of the gravity vector
dt_mpc                = 0.2               # sampling time interval
T_step                = 0.8              # time needed for every step
dt = 0.001


fixed_step_x = 0.05
stride_length = 0.05
step_height = 0.1   # maximum foot trajectory height
nb_steps = 4     # number of steps to perform

nb_dt_per_step = int(round(T_step/dt_mpc))
N  = nb_steps * nb_dt_per_step
N_post = 1000
N_pre = 1000

print(nb_dt_per_step)
T_pre  = 1.5                    # simulation time before starting to walk
T_post = 1.5                    # simulation time after walking


contact_frames = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]

contactNormal = np.matrix([0., 0., 1.]).T
mu = 0.3                            # friction coefficient
fMin = 0.0                          # minimum normal force
fMax = 1e6                       # maximum normal force

kp_contact = 10.0               # proportional gain of contact constraint
kp_foot = 10.0    *0            #             
kp_com = 30.0                   # proportional gain of center of mass task
kp_am = 20.0  *0                 # proportional gain of angular momentum task
kp_posture = 1.0    * 0          # proportional gain of joint posture task

w_contact = 1e5                 # weight of the foot in contact
w_com = 10                     # weight of center of mass task
w_posture = 1e-4                # weight of joint posture task
w_am = 1e0 
w_forceRef = 1e-8
w_foot = 1 *0
w_torque_bounds = 0
#w_cop = 1e-1

tau_max_scaling = 1.4


# PARAMETERS OF REFERENCE SINUSOIDAL TRAJECTORY
amp         = np.array([0.05, 0.0, 0]).T           # amplitude
x0          = np.array([0, 0, 0.12]).T         # offset
phi         = np.array([0.0, 0.0, 0.5*np.pi]).T     # phase
freq        = np.array([0.4, 0.2, 0.4]).T           # frequency (time 2 PI)


T_SIMULATION = 10             # simulation time
dt = 0.001                   # controller time step
ndt = 10 

show_floor = False

randomize_robot_model = 0

simulate_coulomb_friction = 0    # flag specifying whether coulomb friction is simulated
simulation_type = 'timestepping' # either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(6)   # expressed as percentage of torque max


use_viewer = True
simulate_real_time = 1          # flag specifying whether simulation should be real time or as fast as possible
show_floor = False
viewer = pin.visualize.GepettoVisualizer
PRINT_N = 500                   # print every PRINT_N time steps
DISPLAY_N = 1                  # update robot configuration in viwewer every DISPLAY_N time steps
CAMERA_TRANSFORM = [4.0, -0.2, 0.4, 0.5243823528289795, 0.518651008605957, 0.4620114266872406, 0.4925136864185333]

REF_SPHERE_RADIUS = 0.02
REF_SPHERE_COLOR = (1., 0., 0., 1.)