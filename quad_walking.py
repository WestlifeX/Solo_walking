#from pinocchio import libpinocchio_pywrap as pin
import numpy as np
import pybullet as pyb  # Pybullet server
import pybullet_data
from utils.robot_loaders import loadSolo
from utils.robot_wrapper import RobotWrapper
from utils.robot_simulator import RobotSimulator
from numpy import nan
from numpy.linalg import inv, pinv
import os
import gepetto.corbaserver
import time
import matplotlib.pyplot as plt
import sys
import conf as conf



cwd = os.getcwd()
pyb_path = cwd +'/pybullet/'
sys.path.insert(0, pyb_path)
from PyBulletSimulator import PyBulletSimulator

# Simulation integration step
dt = conf.dt

# Import data and trajectories
data = np.load(conf.DATA_FILE_TSID)
N = data['com'].shape[1]
N_pre  = int(conf.T_pre/conf.dt)
N_post = int(conf.T_post/conf.dt)

com_pos = np.empty((3, N+N_post))*np.nan
com_vel = np.empty((3, N+N_post))*np.nan
com_acc = np.empty((3, N+N_post))*np.nan
x_FL   = np.empty((3, N+N_post))*np.nan
dx_FL  = np.empty((3, N+N_post))*np.nan
ddx_FL = np.empty((3, N+N_post))*np.nan
ddx_FL_des = np.empty((3, N+N_post))*np.nan
x_FR   = np.empty((3, N+N_post))*np.nan
dx_FR = np.empty((3, N+N_post))*np.nan
ddx_FR = np.empty((3, N+N_post))*np.nan
ddx_FR_des = np.empty((3, N+N_post))*np.nan
cop_RF = np.zeros((2, N+N_post))
cop_LF = np.zeros((2, N+N_post))

contact_phase = data['contact_phase']
com_pos_ref = np.asarray(data['com'])
com_vel_ref = np.asarray(data['dcom'])
com_acc_ref = np.asarray(data['ddcom'])
x_HRF_ref    = np.asarray(data['x_HRF'])
dx_HRF_ref   = np.asarray(data['dx_HRF'])
ddx_HRF_ref  = np.asarray(data['ddx_HRF'])
x_HLF_ref    = np.asarray(data['x_HLF'])
dx_HLF_ref   = np.asarray(data['dx_HLF'])
ddx_HLF_ref  = np.asarray(data['ddx_HLF'])
x_FRF_ref    = np.asarray(data['x_FRF'])
dx_FRF_ref   = np.asarray(data['dx_FRF'])
ddx_FRF_ref  = np.asarray(data['ddx_FRF'])
x_FLF_ref    = np.asarray(data['x_FLF'])
dx_FLF_ref   = np.asarray(data['dx_FLF'])
ddx_FLF_ref  = np.asarray(data['ddx_FLF'])
cop_ref     = np.asarray(data['cop'])
com_acc_des = np.empty((3, N+N_post))*np.nan # acc_des = acc_ref - Kp*pos_err - Kd*vel_err





r = loadSolo()
robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
simu = RobotSimulator(conf, robot)

# display target position with red sphere in viewer
simu.gui.addSphere('world/target', conf.REF_SPHERE_RADIUS, conf.REF_SPHERE_COLOR)
simu.gui.applyConfiguration('world/target', conf.x0.tolist()+[0.,0.,0.,1.])

frame_id = robot.model.getFrameId(conf.FLF_frame_name)

nx, ndx = 3, 3
N = int(conf.T_SIMULATION/conf.dt)      # number of time steps
tau     = np.empty((robot.na, N))*nan    # joint torques
tau_c   = np.empty((robot.na, N))*nan    # joint Coulomb torques
q       = np.empty((robot.nq, N+1))*nan  # joint angles
v       = np.empty((robot.nv, N+1))*nan  # joint velocities
dv      = np.empty((robot.nv, N+1))*nan  # joint accelerations
x       = np.empty((nx,  N))*nan        # end-effector position
dx      = np.empty((ndx, N))*nan        # end-effector velocity
ddx     = np.empty((ndx, N))*nan        # end effector acceleration
x_ref   = np.empty((nx,  N))*nan        # end-effector reference position
dx_ref  = np.empty((ndx, N))*nan        # end-effector reference velocity
ddx_ref = np.empty((ndx, N))*nan        # end-effector reference acceleration
ddx_des = np.empty((ndx, N))*nan        # end-effector desired acceleration

two_pi_f             = 2*np.pi*conf.freq   # frequency (time 2 PI)
two_pi_f_amp         = two_pi_f*conf.amp
two_pi_f_squared_amp = two_pi_f*two_pi_f_amp

t = 0.0
kp, kd = conf.kp, conf.kd

for i in range(0, N):
    time_start = time.time()
    # set reference trajectory
    x_ref[:,i]  = conf.x0 +  conf.amp*np.sin(two_pi_f*t + conf.phi)
    dx_ref[:,i]  = two_pi_f_amp * np.cos(two_pi_f*t + conf.phi)
    ddx_ref[:,i] = - two_pi_f_squared_amp * np.sin(two_pi_f*t + conf.phi)
    
    simu.gui.applyConfiguration('world/target', x_ref[:,i].tolist()+[0.,0.,0.,1.])
    # read current state from simulator
    v[:,i] = simu.v
    q[:,i] = simu.q
    
    # compute mass matrix M, bias terms h, gravity terms g
    robot.computeAllTerms(q[:,i], v[:,i])
    M = robot.mass(q[:,i], False)
    h = robot.nle(q[:,i], v[:,i], False)
    g = robot.gravity(q[:,i])
    
    J6 = robot.frameJacobian(q[:,i], frame_id, False)
    J = J6[:3,:]            # take first 3 rows of J6
    H = robot.framePlacement(q[:,i], frame_id, False)
    
    x[:,i] = H.translation # take the 3d position of the end-effector
    v_frame = robot.frameVelocity(q[:,i], v[:,i], frame_id, False)
    dx[:,i] = v_frame.linear # take linear part of 6d velocity
#    dx[:,i] = J.dot(v[:,i])
    dJdq = robot.frameAcceleration(q[:,i], v[:,i], None, frame_id, False).linear
    
    ddx_des[:,i] = ddx_ref[:,i] + kp * (x_ref[:,i] - x[:,i]) + kd*(dx_ref[:,i] - dx[:,i]) 
    
    Minv = inv(M)
    Lambda = inv(J @ Minv @ J.T)
    mu = Lambda @ (J @ Minv @ h - dJdq)
    
    #tau_1 = M @ (conf.kp *(conf.q0 - q[:, i]) - kd * v[:,i])
    #tau_0 = (np.identity(6) - J.T @ pinv(J.T)) @ tau_1
    #tau[:, i] = 0.0
    #tau[:,i] = M @ (pinv(J) @ (ddx_des[:,i] - dJdq)) + h
    tau[:, i] = J.T @  Lambda @ ddx_des[:, i] + h
    #tau[:,i] += tau_0

    
    # send joint torques to simulator
    simu.simulate(tau[:,i], conf.dt, conf.ndt)
    tau_c[:,i] = simu.tau_c
        
    
    t += conf.dt
        
    time_spent = time.time() - time_start
    if(conf.simulate_real_time and time_spent < conf.dt): 
        time.sleep(conf.dt-time_spent)
        
'''q_init = np.zeros(12)
#q_init = [ x_HRF_ref[:,0], x_FLF_ref[:,0], x_HLF_ref[:,0], x_FRF_ref[:,0]]
# Instatiate the quadruped
device = PyBulletSimulator()
device.Init(calibrateEncoders=True, q_init=q_init,
            envID=0, use_flat_plane=True, enable_pyb_GUI=True, dt=dt)

q_ref = np.zeros(12)
v_ref = np.zeros(12)
tau_ref = np.zeros(12)
P = 0.5 * np.ones(12)
D = 0.5 * np.ones(12)

t = 0
i= 0
t_max = 10
for i in range(3):
    plt.plot(x_HRF_ref[i,:])
    plt.show()

for i in range(int(1e6)):
    if i<N:
        device.SetDesiredFPosition([x_HRF_ref[:,i], x_FLF_ref[:,i], x_HLF_ref[:,i], x_FRF_ref[:,i]])
        #print (np.array(device.GetContact()))
        device.SendCommand()
    time.sleep(dt)
    i += 1'''