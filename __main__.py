
#####################
#  LOADING MODULES ##
#####################

import time

import pybullet as p
import pinocchio as pin
import numpy as np
from numpy.linalg import inv, pinv

from utils.robot_wrapper import RobotWrapper
from utils.robot_simulator import RobotSimulator
from tsid_quadruped import TsidQuadruped
import conf as conf

# Functions to initialize the simulation and retrieve joints positions/velocities
from utils.initialization_simulation import configure_simulation, getPosVelJoints

####################
#  INITIALIZATION ##
####################

dt = 0.001  # time step of the simulation
t = 0.0
N_SIMULATION = 10000
# If True then we will sleep in the main loop to have a 1:1 ratio of (elapsed real time / elapsed time in the
# simulation)
realTimeSimulation = True
enableGUI = True  # enable PyBullet GUI or not
robotId, r, revoluteJointIndices = configure_simulation(dt, conf.q0[:3], enableGUI)

robot = RobotWrapper(r.model, r.collision_model, r.visual_model)

# TSID implementation for quadrupeds
tsid = TsidQuadruped(conf, )
sampleCom = tsid.trajCom.computeNext()

# Balls to show trajectories
sphereRadius = conf.REF_SPHERE_RADIUS
ref_sphere_viz = p.createVisualShape(p.GEOM_SPHERE,
                            radius = sphereRadius,
                            rgbaColor = conf.REF_SPHERE_COLOR)
ref_sphere1 = p.createMultiBody(
          baseMass = 0,
          baseVisualShapeIndex= ref_sphere_viz, 
          basePosition = conf.x01,
          useMaximalCoordinates=True)
ref_sphere2 = p.createMultiBody(
          baseMass = 0,
          baseVisualShapeIndex= ref_sphere_viz, 
          basePosition = conf.x02,
          useMaximalCoordinates=True)

# Now I'm only tracking a sine wave with the CoM, 
# This will became a trajectory from the trajectory optimization, along with trajectories for feet
offset     = np.array([0.0, 0.0, 0.3])
amp        = np.array([0.05, 0.01, 0.03])
two_pi_f             = 2*np.pi*np.array([1, 1.0, 1])
two_pi_f_amp         = np.multiply(two_pi_f,amp)
two_pi_f_squared_amp = np.multiply(two_pi_f, two_pi_f_amp)


q, qdot = tsid.q, tsid.qdot
samplePosture = tsid.trajPosture.computeNext()
tsid.postureTask.setReference(samplePosture)

###############
#  MAIN LOOP ##
###############

for i in range(N_SIMULATION):  # run the simulation during dt * i_max seconds (simulation time)

    # Time at the start of the loop
    if realTimeSimulation:
        t0 = time.time()

    # Position and velocity of the PyBullet simulator
    q, qdot = getPosVelJoints(robotId, revoluteJointIndices)
    
    
    # Refernce trajectories
    x1_ref = offset + np.multiply(amp, np.sin(two_pi_f*t))
    dx1_ref = np.multiply(two_pi_f_amp, np.cos(two_pi_f*t))
    ddx1_ref = np.multiply(two_pi_f_squared_amp, -np.sin(two_pi_f*t))
    
    # Display the trajectory with a ball 
    p.resetBasePositionAndOrientation(ref_sphere1, x1_ref, p.getQuaternionFromEuler([0,0,0]))

    sampleCom.pos(x1_ref)
    sampleCom.vel(dx1_ref)
    sampleCom.acc(ddx1_ref)
    

    tsid.comTask.setReference(sampleCom)

    
    HQPData = tsid.formulation.computeProblemData(t, q, qdot)
    if i == 0: HQPData.print_all()

    sol = tsid.solver.solve(HQPData)
    if(sol.status!=0):
        print("[%d] QP problem could not be solved! Error code:"%(i), sol.status)
        break
    
    q2dot = tsid.formulation.getAccelerations(sol)
    jointTorques = tsid.formulation.getActuatorForces(sol)
    
    
    # Set control torques for all joints in PyBullet
    p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

    # Compute one step of simulation
    p.stepSimulation()
    t += dt
    # Sleep to get a real time simulation
    if realTimeSimulation:
        t_sleep = dt - (time.time() - t0)
        if t_sleep > 0:
            time.sleep(t_sleep)
            
# Shut down the PyBullet client
p.disconnect()

