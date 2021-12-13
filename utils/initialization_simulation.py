# coding: utf8

import numpy as np  # Numpy library
import pybullet_data
from example_robot_data import loadSolo  # Functions to load the SOLO quadruped

import pybullet as p  # PyBullet simulator


def configure_simulation(dt, x0, enableGUI):
    global jointTorques
    # Load the robot for Pinocchio
    solo = loadSolo(False)
    solo.initDisplay(loadModel=True)

    # Start the client for PyBullet
    if enableGUI:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)  # noqa
    # p.GUI for graphical version
    # p.DIRECT for non-graphical version

    # Set gravity (disabled by default)
    p.setGravity(0, 0, -9.81)

    # Load horizontal plane for PyBullet
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    # Load the robot for PyBullet
    robotStartPos = x0
    robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    p.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/robots/solo_description/robots")
    robotId = p.loadURDF("solo12.urdf", robotStartPos, robotStartOrientation)

    
    # Set time step of the simulation
    # dt = 0.001
    p.setTimeStep(dt)
    # realTimeSimulation = True # If True then we will sleep in the main loop to have a frequency of 1/dt

    # Disable default motor control for revolute joints
    #revoluteJointIndices = [0, 1, 3, 4, 6, 7, 9, 10]
    revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    
    for i in revoluteJointIndices:
        p.resetJointState(robotId, i, np.pi/4)
        
    p.resetJointState(robotId, 0, 0)
    p.resetJointState(robotId, 4, 0)
    p.resetJointState(robotId, 8, 0)
    p.resetJointState(robotId, 12, 0)
    
    p.resetJointState(robotId, 2, -np.pi/2)
    p.resetJointState(robotId, 6, -np.pi/2)
    p.resetJointState(robotId, 10, -np.pi/2)
    p.resetJointState(robotId, 14, -np.pi/2)
    
    p.setJointMotorControlArray(robotId,
                                jointIndices=revoluteJointIndices,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocities=[0.0 for m in revoluteJointIndices],
                                forces=[0.0 for m in revoluteJointIndices])
    
# Get position of feet in world frame with base at (0, 0, 0)
    feetLinksID = [3, 7, 11, 15]
    linkStates = p.getLinkStates(robotId, feetLinksID)

    # Get minimum height of feet (they are in the ground since base is at 0, 0, 0)
    z_min = linkStates[0][4][2]
    i_min = 0
    i = 1
    for link in linkStates[1:]:
        if link[4][2] < z_min:
            z_min = link[4][2]
            i_min = i
        i += 1

    # Set base at (0, 0, -z_min) so that the lowest foot is at z = 0
    p.resetBasePositionAndOrientation(robotId, [0.0, 0.0, -z_min], [0, 0, 0, 1])

    # Progressively raise the base to achieve proper contact (take into account radius of the foot)
    while (p.getClosestPoints(robotId, planeId, distance=0.005,
                                linkIndexA=feetLinksID[i_min]))[0][8] < 0.001:
        z_min -= 0.001
        p.resetBasePositionAndOrientation(robotId, [x0[0], x0[1], -z_min], [0, 0, 0, 1])

    # Enable torque control for revolute joints
    jointTorques = [0.0 for m in revoluteJointIndices]
    p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

    # Compute one step of simulation for initialization
    p.stepSimulation()

    return robotId, solo, revoluteJointIndices


# Function to get the position/velocity of the base and the angular position/velocity of all joints
def getPosVelJoints(robotId, revoluteJointIndices):

    jointStates = p.getJointStates(robotId, revoluteJointIndices)  # State of all joints
    baseState = p.getBasePositionAndOrientation(robotId)  # Position of the free flying base
    baseVel = p.getBaseVelocity(robotId)  # Velocity of the free flying base

    # Reshaping data into q and qdot
    q = np.vstack((np.array([baseState[0]]).transpose(), np.array([baseState[1]]).transpose(),
                   np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).transpose()))
    qdot = np.vstack((np.array([baseVel[0]]).transpose(), np.array([baseVel[1]]).transpose(),
                      np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).transpose()))

    return q, qdot
