import matplotlib
import numpy as np
import LMPC_walking.second_order.plot_utils as plot_utils
from LMPC_walking.second_order.LIPM_to_whole_body import compute_foot_traj, interpolate_lipm_traj_quad
import matplotlib.pyplot as plt
import conf as conf

#matplotlib.use('TkAgg')

# READ COM-COP TRAJECTORIES COMPUTED WITH LIPM MODEL
data = np.load(conf.DATA_FILE_LIPM)
com_state_x = data['com_state_x']
com_state_y = data['com_state_y']
cop_ref = data['cop_ref']
cop_x = data['cop_x']
cop_y = data['cop_y']
foot_steps = data['foot_steps']
foot_steps_trj = data['foot_steps_trj']

# INTERPOLATE WITH TIME STEP OF CONTROLLER (TSID)
dt_ctrl = conf.dt                       # time step used by TSID
com, dcom, ddcom, cop, contact_phase, foot_steps_ctrl = \
    interpolate_lipm_traj_quad(conf.T_step, conf.nb_steps, conf.dt_mpc, dt_ctrl, conf.h, conf.g, foot_steps_trj,
                             com_state_x, com_state_y, cop_ref, cop_x, cop_y)

# COMPUTE TRAJECTORIES FOR FEET
N  = conf.nb_steps * int(round(conf.T_step/conf.dt_mpc))  # number of time steps for traj-opt
N_ctrl = int((N*conf.dt_mpc)/dt_ctrl)                     # number of time steps for TSID

# Hind feet
foot_steps_HRF = foot_steps[::2,0:2]    # assume first foot step corresponds to right foot
x_HRF, dx_HRF, ddx_HRF = compute_foot_traj(foot_steps_HRF, N_ctrl, dt_ctrl, conf.T_step, conf.step_height, 'stance')
foot_steps_HLF = foot_steps[1::2,0:2]
x_HLF, dx_HLF, ddx_HLF = compute_foot_traj(foot_steps_HLF, N_ctrl, dt_ctrl, conf.T_step, conf.step_height, 'swing')

# Front feet
foot_steps_FRF = foot_steps[1::2,2:4]    # assume first foot step corresponds to right foot
x_FRF, dx_FRF, ddx_FRF = compute_foot_traj(foot_steps_FRF, N_ctrl, dt_ctrl, conf.T_step, conf.step_height, 'swing')
foot_steps_FLF = foot_steps[::2, 2:4]
x_FLF, dx_FLF, ddx_FLF = compute_foot_traj(foot_steps_FLF, N_ctrl, dt_ctrl, conf.T_step, conf.step_height, 'stance')

# SAVE COMPUTED TRAJECTORIES IN NPY FILE FOR TSID
np.savez(conf.DATA_FILE_TSID, com=com, dcom=dcom, ddcom=ddcom,
         x_HRF=x_HRF, dx_HRF=dx_HRF, ddx_HRF=ddx_HRF,
         x_FRF=x_FRF, dx_FRF=dx_FRF, ddx_FRF=ddx_FRF,
         x_HLF=x_HLF, dx_HLF=dx_HLF, ddx_HLF=ddx_HLF,
         x_FLF=x_FLF, dx_FLF=dx_FLF, ddx_FLF=ddx_FLF,
         contact_phase=contact_phase, cop=cop)

# PLOT STUFF
time_ctrl = np.arange(0, round(N_ctrl*dt_ctrl, 2), dt_ctrl)

for i in range(3):
    plt.figure()
    plt.plot(time_ctrl, x_HRF[i,:-1], label='x RF '+str(i))
    plt.plot(time_ctrl, x_HLF[i,:-1], label='x LF '+str(i))
    plt.plot(time_ctrl, x_FRF[i, :-1], label='x FRF ' + str(i))
    plt.plot(time_ctrl, x_FLF[i, :-1], label='x FLF ' + str(i))
    plt.legend()
plt.show()