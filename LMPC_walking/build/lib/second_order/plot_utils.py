#    LMPC_walking is a python software implementation of some of the linear MPC
#    algorithms based presented in:
#    https://groups.csail.mit.edu/robotics-center/public_papers/Wieber15.pdf
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

# headers:
# -------
from numpy import zeros, tile, arange, array, dot, frombuffer
from math import sqrt, atan2, pi, cos, sin
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy.linalg import eig


def plot_x(plot_legend, reference_time_stamp, reference_time, min_admissible_cop,
                                             max_admissible_cop, Z_x, X, Z_ref):
    #ZMP_x_fig = plt.figure()
    if plot_legend:
        plt.rc('text', usetex = True)
        plt.rc('font', family ='serif')
        plt.step(reference_time_stamp, Z_x, 'blue', label = r'$p_x$')
        #plt.step(reference_time_stamp, Z_ref[:,0])
        plt.step(reference_time_stamp, X[:,0], 'lime',
                 label = r'$c_x$')
        plt.step(reference_time_stamp, min_admissible_cop[:, 0],'black',
                    linestyle = '--', linewidth = 1.5, label = r'$\mathcal{U}$')
        plt.step(reference_time_stamp, max_admissible_cop[:, 0],'black',
                                              linestyle = '--', linewidth = 1.5)
        plt.xlabel(r'\textbf{time} (s)', fontsize=14)
        plt.ylabel(r'\textbf{x, z} (m)', fontsize=14)
        plt.legend()
    else:
        plt.step(reference_time_stamp, Z_x, 'blue')
        #plt.step(reference_time_stamp, Z_ref[:,0])
        plt.step(reference_time_stamp, X[:,0], 'lime')

def plot_y(plot_legend, reference_time_stamp, reference_time, min_admissible_cop,
                         max_admissible_cop, Z_y, Y, Z_ref, max_admissible_com):
    #ZMP_y_fig = plt.figure()
    plt.rc('text', usetex = True)
    plt.rc('font', family ='serif')
    if plot_legend:
        plt.step(reference_time_stamp, Z_y, 'blue', label = r'$p^y$')
        plt.step(reference_time_stamp, Y[:,0], 'lime', label = r'$c^y$')
        plt.step(reference_time_stamp, min_admissible_cop[:, 1],'black',
                 linestyle = '-.', linewidth = 1.5, label = r'$\mathcal{U}$')
        plt.step(reference_time_stamp, max_admissible_cop[:, 1],'black',
                 linestyle = '-.', linewidth = 1.5)
        plt.step(reference_time_stamp, max_admissible_com[:],'red',
                linestyle = '-.', linewidth = 1.5, label = r'$\mathcal{X}^c$')
        plt.xlabel(r'\textbf{time} (s)', fontsize=14)
        plt.ylabel(r'\textbf{y, z} (m)', fontsize=14)
        plt.legend()
    else:
        plt.step(reference_time_stamp, Z_y, 'blue')
        plt.step(reference_time_stamp, Y[:,0], 'lime', label = r'\textbf{CoM}')

def plot_y_box(plot_legend, reference_time_stamp, reference_time,
                          min_admissible_cop, max_admissible_cop, Z_y, Y, Z_ref,
                                                         max_admissible_com, N):
    #ZMP_y_fig = plt.figure()
    plt.rc('text', usetex = True)
    plt.rc('font', family ='serif')
    if plot_legend:
        plt.step(reference_time_stamp, Z_y, 'blue', label = r'$p^y$')
        plt.step(reference_time_stamp, Y[:,0], 'lime', label = r'$c^y$')
        plt.step(reference_time_stamp, min_admissible_cop[:, 1],'black',
                 linestyle = '-.', linewidth = 1.5, label = r'$\mathcal{U}$')
        plt.step(reference_time_stamp, max_admissible_cop[:, 1],'black',
                                              linestyle = '-.', linewidth = 1.5)
        plt.step(reference_time_stamp[N:reference_time-N+1],
                                 max_admissible_com[N:reference_time-N+1],'red',
                  linestyle = '-.', linewidth = 1.5, label = r'$\mathcal{X}^c$')
        plt.step(reference_time_stamp[N:reference_time-N+1],
                                -max_admissible_com[N:reference_time-N+1],'red',
                                              linestyle = '-.', linewidth = 1.5)
        plt.xlabel(r'\textbf{time} (s)', fontsize=14)
        plt.ylabel(r'\textbf{y, z} (m)', fontsize=14)
        plt.legend()
    else:
        plt.step(reference_time_stamp, Z_y, 'blue')
        plt.step(reference_time_stamp, Y[:,0], 'lime', label = r'\textbf{CoM}')

def plot_xy(reference_time_stamp, walking_time, foot_length, foot_width, Z_ref,
                                                                Z_x, Z_y, X, Y):
    ZMP_CoP_xy_fig = plt.figure()
    plt.plot(Z_x, Z_y, 'r', label = r'\textbf{computed CoP}')
    plt.plot(X[:,0], Y[:,0], 'lime', label = r'\textbf{CoM}')
    currentAxis    = plt.gca()
    for i in range(walking_time):
        current_foot = patches.Rectangle((Z_ref[i,0]-foot_length/2,
                            Z_ref[i,1]-foot_width/2),foot_length, foot_width,\
                            linewidth = 0.8, linestyle = '-.',  edgecolor = 'b',
                                                             facecolor = 'none')
        currentAxis.add_patch(current_foot)
    currentAxis.autoscale_view()
    plt.xlabel(r'\textbf{x} (m)', fontsize=14)
    plt.ylabel(r'\textbf{y} (m)', fontsize=14)
    plt.legend()
    plt.show()
#-------------------------------------------------------------------------------
#TODO
#def plot_x_robust(plot_legend, reference_time_stamp, reference_time,
#min_admissible_cop, max_admissible_cop, min_admissible_cop_back_off,
#max_admissible_cop_back_off, Z_x, X, Z_ref):
def plot_y_robust_MPC_box(plot_legend, reference_time_stamp, reference_time,
            min_admissible_cop, max_admissible_cop, min_admissible_cop_back_off,
    max_admissible_cop_back_off, Z_y, Y, Z_ref, CoM_constraint, CoM_back_off,N):
    #ZMP_y_fig = plt.figure()
    if plot_legend:
        plt.rc('text', usetex = True)
        plt.rc('font', family ='serif')
        plt.step(reference_time_stamp, Z_y, 'blue',label = r'$p^y$')
        #plt.step(reference_time_stamp, Z_ref[:,1])
        plt.step(reference_time_stamp, Y[:,0], 'lime', label = r'$c^y$')
        plt.step(reference_time_stamp, min_admissible_cop[:, 1],'black',
                 linestyle = '-.', linewidth = 1.5, label = r'$\mathcal{U}$')
        plt.step(reference_time_stamp, max_admissible_cop[:, 1],'black',
                linestyle = '-.', linewidth = 1.5)
        plt.step(reference_time_stamp,min_admissible_cop_back_off[:, 1], 'black',
                                              linestyle = '--', linewidth = 1.0,
                                  label = r'$\mathcal{U} \ominus K\mathcal{W}$')
        plt.step(reference_time_stamp, max_admissible_cop_back_off[:, 1],'black',
                                              linestyle = '--', linewidth = 1.0)
        plt.step(reference_time_stamp[N:reference_time-N+1],
             CoM_constraint[N:reference_time-N+1],'red', linestyle = '-.',
                                    linewidth = 1.5, label = r'$\mathcal{X}^c$')
        plt.step(reference_time_stamp[N:reference_time-N+1],
                  -CoM_constraint[N:reference_time-N+1],'red', linestyle = '-.',
                                                                linewidth = 1.5)
        plt.step(reference_time_stamp[N:reference_time-N+1],
                     CoM_back_off[N:reference_time-N+1],'red', linestyle = '--',
                   linewidth = 1.0, label = r'$\mathcal{X}^c \ominus \Omega^c$')
        plt.step(reference_time_stamp[N:reference_time-N+1],
                    -CoM_back_off[N:reference_time-N+1],'red', linestyle = '--',
                                                                linewidth = 1.0)
        plt.xlabel('time (s)', fontsize=19)
        plt.ylabel('y, z (m)', fontsize=19)
        plt.legend( fontsize=10)
    else:
        plt.step(reference_time_stamp, Z_y, 'blue')
        plt.step(reference_time_stamp, Y[:,0], 'lime')

def plot_y_robust_MPC(plot_legend, reference_time_stamp, reference_time,
            min_admissible_cop, max_admissible_cop, min_admissible_cop_back_off,
      max_admissible_cop_back_off, Z_y, Y, Z_ref, CoM_constraint, CoM_back_off):
    #ZMP_y_fig = plt.figure()
    if plot_legend:
        plt.rc('text', usetex = True)
        plt.rc('font', family ='serif')
        plt.step(reference_time_stamp, Z_y, 'blue',label = r'$p^y$')
        #plt.step(reference_time_stamp, Z_ref[:,1])
        plt.step(reference_time_stamp, Y[:,0], 'lime', label = r'$c^y$')
        plt.step(reference_time_stamp, min_admissible_cop[:, 1],'black',
                 linestyle = '-.', linewidth = 1.5, label = r'$\mathcal{U}$')
        plt.step(reference_time_stamp, max_admissible_cop[:, 1],'black',
                linestyle = '-.', linewidth = 1.5)
        plt.step(reference_time_stamp,min_admissible_cop_back_off[:, 1], 'black',
                                              linestyle = '--', linewidth = 1.0,
                                  label = r'$\mathcal{U} \ominus K\mathcal{W}$')
        plt.step(reference_time_stamp, max_admissible_cop_back_off[:, 1],'black',
                                              linestyle = '--', linewidth = 1.0)
        plt.step(reference_time_stamp, CoM_constraint,'red', linestyle = '-.',
                                    linewidth = 1.5, label = r'$\mathcal{X}^c$')
        plt.step(reference_time_stamp, CoM_back_off,'red', linestyle = '--',
                   linewidth = 1.0, label = r'$\mathcal{X}^c \ominus \Omega^c$')
        plt.xlabel(r'\textbf{time} (s)',fontsize=19)
        plt.ylabel(r'\textbf{y, z} (m)',fontsize=19)
        plt.legend()
    else:
        plt.step(reference_time_stamp, Z_y, 'blue')
        plt.step(reference_time_stamp, Y[:,0], 'lime')
#-------------------------------------------------------------------------------
def plot_y_stochastic_MPC_box(plot_legend, reference_time_stamp, reference_time,
            min_admissible_cop, max_admissible_cop, min_admissible_cop_back_off,
                 max_admissible_cop_back_off, Z_y, Y, Z_ref, CoM_constraint, N):
    #ZMP_y_fig = plt.figure()
    if plot_legend:
        plt.rc('text', usetex = True)
        plt.rc('font', family ='serif')
        plt.step(reference_time_stamp, Z_y, 'blue',label = r'$p^y$')
        #plt.step(reference_time_stamp, Z_ref[:,1])
        plt.step(reference_time_stamp, Y[:,0], 'lime', label = r'$c^y$')
        plt.step(reference_time_stamp, min_admissible_cop[:, 1],'black',
                    linestyle = '-.', linewidth = 1.5, label = r'$\mathcal{U}$')
        plt.step(reference_time_stamp, max_admissible_cop[:, 1],'black',
                                              linestyle = '-.', linewidth = 1.5)
        plt.step(reference_time_stamp,min_admissible_cop_back_off[:, 1],'black',
                                              linestyle = '--', linewidth = 1.0,
                                  label = r'$\mathcal{U} \ominus K\mathcal{W}$')
        plt.step(reference_time_stamp, max_admissible_cop_back_off[:, 1],'black',
                                              linestyle = '--', linewidth = 1.0)
        plt.step(reference_time_stamp[N:reference_time-N+1],
                   CoM_constraint[N:reference_time-N+1],'red', linestyle = '-.',
                                    linewidth = 1.5, label = r'$\mathcal{X}^c$')
        plt.step(reference_time_stamp[N:reference_time-N+1],
                  -CoM_constraint[N:reference_time-N+1],'red', linestyle = '-.',
                                    linewidth = 1.5, label = r'$\mathcal{X}^c$')
        plt.xlabel('{time} (s)', fontsize=19)
        plt.ylabel('{y, z} (m)', fontsize=19)
        plt.legend()
    else:
        plt.step(reference_time_stamp, Z_y, 'blue')
        plt.step(reference_time_stamp, Y[:,0], 'lime')

def plot_y_MPC_vs_SMPC(plot_legend, reference_time_stamp,
                   desired_walking_time, min_admissible_cop, max_admissible_cop,
              min_admissible_cop_back_off, max_admissible_cop_back_off, Z_y_mpc,
                  Y_mpc, Z_y_smpc, Y_smpc, Z_ref, CoM_constraint, CoM_back_off):
    if plot_legend:
        plt.rc('text', usetex = True)
        plt.rc('font', family ='serif')
        plt.step(reference_time_stamp, Z_y_smpc, 'blue',label = r'$p^y_{smpc}$ ')
        plt.step(reference_time_stamp, Z_y_mpc, 'purple',label = r'$p^y_{mpc}$')
        #plt.step(reference_time_stamp, Z_ref[:,1])
        plt.step(reference_time_stamp, Y_smpc[:,0], 'lime',
                                                        label = r'$c^y_{smpc}$')
        plt.step(reference_time_stamp, Y_mpc[:,0], 'orange',
                                                         label = r'$c^y_{mpc}$')
        plt.step(reference_time_stamp, min_admissible_cop[:, 1],'black',
                    linestyle = '-.', linewidth = 1.5, label = r'$\mathcal{U}$')
        plt.step(reference_time_stamp, max_admissible_cop[:, 1],'black',
                                              linestyle = '-.', linewidth = 1.5)
        #plt.step(reference_time_stamp,min_admissible_cop_back_off[:, 1], 'black',
        #linestyle = '--', linewidth = 1.0, label = r'$\mathcal{U} \ominus K\mathcal{W}$')
        #plt.step(reference_time_stamp, max_admissible_cop_back_off[:, 1],'black',
        #    linestyle = '--', linewidth = 1.0)
        plt.step(reference_time_stamp, CoM_constraint,'red', linestyle = '-.',
                                    linewidth = 1.5, label = r'$\mathcal{X}^c$')
        #plt.step(reference_time_stamp, CoM_back_off,'red', linestyle = '--',
        #linewidth = 1.0, label = r'$\mathcal{X}^c \ominus \Omega^c$')
        plt.xlabel(r'\textbf{time} (s)', fontsize=14)
        plt.ylabel(r'\textbf{y, z} (m)', fontsize=14)
        plt.legend()
    else:
        plt.step(reference_time_stamp, Z_y_smpc, 'blue')
        plt.step(reference_time_stamp, Y_smpc[:,0], 'lime')
        plt.step(reference_time_stamp, Z_y_mpc, 'purple')
        plt.step(reference_time_stamp, Y_mpc[:,0], 'orange')

#-------------------------------------------------------------------------------
#                               plots for debugging
# ------------------------------------------------------------------------------
def create_empty_figure(nRows=1, nCols=1, figsize=(7, 7), spinesPos=None,
                                                                   sharex=True):
    f, ax = plt.subplots(nRows, nCols, figsize=figsize, sharex=sharex)
    mngr = plt.get_current_fig_manager()
    # mngr.window.setGeometry(50,50,1080,720)
    if spinesPos is not None:
        if nRows * nCols > 1:
            for axis in ax.reshape(nRows * nCols):
                movePlotSpines(axis, spinesPos)
        else:
            movePlotSpines(ax, spinesPos)
    return f, ax

def plot_horizons(desired_walking_time, N, desired_Z_ref, horizon_data,
                 foot_length, foot_width):
    for i in range(desired_walking_time):
        time_k  = horizon_data[i]['time_k']
        Z_ref_k = horizon_data[i]['zmp_reference']
        X_k     = horizon_data[i]['X_k']
        Y_k     = horizon_data[i]['Y_k']
        Z_x_k   = horizon_data[i]['Z_x_k']
        Z_y_k   = horizon_data[i]['Z_y_k']

        min_admissible_CoP = Z_ref_k - tile([foot_length/2, foot_width/2], (N,1))
        max_admissible_cop = Z_ref_k + tile([foot_length/2, foot_width/2], (N,1))

        #plot_x(time_k, N, min_admissible_CoP, max_admissible_cop, \
        #                  Z_x_k, X_k, Z_ref_k)
        plot_y(time_k, N, min_admissible_CoP, max_admissible_cop, \
                          Z_y_k, Y_k, Z_ref_k)
        #plot_xy(time_k, N, foot_length, foot_width, desired_Z_ref, \
        #                   Z_x_k, Z_y_k, X_k, Y_k)

# plot 2D covariance matrix
def plot_2D_covariance(mu, cov_matrix, color='g', sigmaFactor=1.0):

    [w, v]   = eig(cov_matrix)
    lambda_1  = w[0]
    lambda_2 = w[1]
    if lambda_1 < 0 or lambda_2 < 0:
      print('Non-positive definite matrix:', '\n')
      print(cov_matrix)
      return
    v1 = v[:,0];
    v2 = v[:,1];
    theta = atan2(v1[1],v1[0]);

    a = sigmaFactor*sqrt(lambda_1)
    b = sigmaFactor*sqrt(lambda_2)

    #plot the ellipse
    no_points = 500
    ang =  (2*pi/no_points)*arange(0, no_points+1)
    X   = zeros((2, ang.shape[0]))
    for i in range(ang.shape[0]):
        X[0, i] = a*cos(ang[i])
        X[1, i] = b*sin(ang[i])
    rot_mat = array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    points  = dot(rot_mat, X)
    #figure, ax = plt.subplots()
    #ax.autoscale_view()
    #ax.relim()
    plt.plot(points[0,:], points[1,:])
    #figure.canvas.draw()
    #figure.show()
    #figure.canvas.flush_events()
    #image = frombuffer(figure.canvas.tostring_rgb())

    #return image
