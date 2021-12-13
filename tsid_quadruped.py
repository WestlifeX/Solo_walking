#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pinocchio as pin
import tsid
import numpy as np
import numpy.matlib as matlib
import os
import gepetto.corbaserver
import time
import subprocess

class TsidQuadruped:
    def __init__(self, conf, q0, viewer = True):
        self.conf = conf
        self.robot = tsid.RobotWrapper(conf.urdf, [conf.path], pin.JointModelFreeFlyer(),False)
        robot = self.robot
        self.model = model = robot.model()
        self.FL = model.getFrameId(conf.contact_frames[0])
        self.FR = model.getFrameId(conf.contact_frames[1])
        self.HL = model.getFrameId(conf.contact_frames[2])
        self.HR = model.getFrameId(conf.contact_frames[3])
        
        q = q0
        qdot = np.zeros(robot.nv)
        
        formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        formulation.computeProblemData(0.0, q, qdot)
        data = formulation.data()
        
        H_fl_ref = robot.framePosition(data, self.FL)
        H_fr_ref = robot.framePosition(data, self.FR)
        H_hl_ref = robot.framePosition(data, self.HL)
        H_hr_ref = robot.framePosition(data, self.HR)
        
        
        self.com_ref = robot.com(data)
        
        #q[2] -= robot.framePosition(data, model.getFrameId(conf.contact_frames[0])).translation[2]
        robot.computeAllTerms(data, q, qdot)
        
        # Contacts for all the 4 feet
        contacts = 4*[None]
        for i, name in enumerate(conf.contact_frames):
            contacts[i] =tsid.ContactPoint(name, robot, name, conf.contactNormal, conf.mu, conf.fMin, conf.fMax)
            contacts[i].setKp(conf.kp_contact * np.ones(3))
            contacts[i].setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(3))
            H_ref = robot.framePosition(data, robot.model().getFrameId(name))
            contacts[i].setReference(H_ref)
            contacts[i].useLocalFrame(False)
            formulation.addRigidContact(contacts[i], conf.w_forceRef, 1.0, 1)
            
        
        self.FLTask = tsid.TaskSE3Equality('Task_FL', self.robot, self.conf.contact_frames[0])
        self.FLTask.setKp(conf.kp_foot * np.ones(6))
        self.FLTask.setKd(2* np.sqrt(conf.kp_foot) * np.ones(6))
        self.FL_traj = tsid.TrajectorySE3Constant('traj_FL', H_fl_ref)
        formulation.addMotionTask(self.FLTask, conf.w_foot, 1, 0.0)
        
        self.FRTask = tsid.TaskSE3Equality('Task_FR', self.robot, self.conf.contact_frames[1])
        self.FRTask.setKp(conf.kp_foot * np.ones(6))
        self.FRTask.setKd(2* np.sqrt(conf.kp_foot) * np.ones(6))
        self.FR_traj = tsid.TrajectorySE3Constant('traj_FR', H_fr_ref)
        formulation.addMotionTask(self.FRTask, conf.w_foot, 1, 0.0)
        
        self.HLTask = tsid.TaskSE3Equality('Task_HL', self.robot, self.conf.contact_frames[2])
        self.HLTask.setKp(conf.kp_foot * np.ones(6))
        self.HLTask.setKd(2* np.sqrt(conf.kp_foot) * np.ones(6))
        self.HL_traj = tsid.TrajectorySE3Constant('traj_HL', H_hl_ref)
        formulation.addMotionTask(self.HLTask, conf.w_foot, 1, 0.0)
        
        self.HRTask = tsid.TaskSE3Equality('Task_HR', self.robot, self.conf.contact_frames[3])
        self.HRTask.setKp(conf.kp_foot * np.ones(6))
        self.HRTask.setKd(2* np.sqrt(conf.kp_foot) * np.ones(6))
        self.HR_traj = tsid.TrajectorySE3Constant('traj_HR', H_hr_ref)
        formulation.addMotionTask(self.HRTask, conf.w_foot, 1, 0.0)
     
        self.COMTask = tsid.TaskComEquality("task-com", robot)
        self.COMTask.setKp(conf.kp_com * np.ones(3))
        self.COMTask.setKd(2.0 * np.sqrt(conf.kp_com) * np.ones(3))
        self.COM_traj = tsid.TrajectoryEuclidianConstant("traj_com", self.com_ref)
        formulation.addMotionTask(self.COMTask, conf.w_com, 1, 0.0)
        
        self.amTask = tsid.TaskAMEquality("task-am", robot)
        self.amTask.setKp(conf.kp_am * np.array([1., 1., 0.]))
        self.amTask.setKd(2.0 * np.sqrt(conf.kp_am * np.array([1., 1., 0.])))
        formulation.addMotionTask(self.amTask, conf.w_am, 1, 0.)
        self.sampleAM = tsid.TrajectorySample(3)
        self.amTask.setReference(self.sampleAM)
        
        #copTask = tsid.TaskCopEquality("task-cop", robot)
        #formulation.addForceTask(copTask, conf.w_cop, 1, 0.0)
        
        postureTask = tsid.TaskJointPosture("task-posture", robot)
        postureTask.setKp(conf.kp_posture * np.ones(robot.nv-6))
        postureTask.setKd(2.0 * np.sqrt(conf.kp_posture) * np.ones(robot.nv-6))
        formulation.addMotionTask(postureTask, conf.w_posture, 1, 0.0)
        
        q_ref = q[7:]
        self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)
        
        
        self.tau_max = conf.tau_max_scaling*robot.model().effortLimit[-robot.na:]
        self.tau_min = -self.tau_max
        actuationBoundsTask = tsid.TaskActuationBounds("task-actuation-bounds", robot)
        actuationBoundsTask.setBounds(self.tau_min, self.tau_max)
        if conf.w_torque_bounds > 0.0:
            formulation.addActuationTask(actuationBoundsTask, conf.w_torque_bounds, 0, 0.0)
   
        solver = tsid.SolverHQuadProgFast("qp_solver")
        solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)
        
        
        self.contacts = contacts
        
        self.contact_FL = contacts[0]
        self.contact_FR = contacts[1]
        self.contact_HL = contacts[2]
        self.contact_HR = contacts[3]
        
        self.sampleFL = self.FL_traj.computeNext()
        self.sample_FL_pos = self.sampleFL.pos()
        self.sample_FL_vel = self.sampleFL.vel()
        self.sample_FL_acc = self.sampleFL.acc()
        
        self.sampleFR = self.FR_traj.computeNext()
        self.sample_FR_pos = self.sampleFR.pos()
        self.sample_FR_vel = self.sampleFR.vel()
        self.sample_FR_acc = self.sampleFR.acc()
        
        self.sampleHL = self.HL_traj.computeNext()
        self.sample_HL_pos = self.sampleHL.pos()
        self.sample_HL_vel = self.sampleHL.vel()
        self.sample_HL_acc = self.sampleHL.acc()
        
        self.sampleHR = self.HR_traj.computeNext()
        self.sample_HR_pos = self.sampleHR.pos()
        self.sample_HR_vel = self.sampleHR.vel()
        self.sample_HR_acc = self.sampleHR.acc()
        
        self.sampleCOM = self.COM_traj.computeNext()
        self.sample_COM_pos = self.sampleCOM.pos()
        self.sample_COM_vel = self.sampleCOM.vel()
        self.sample_COM_acc = self.sampleCOM.acc()
        
        self.H_fl_ref =  H_fl_ref
        self.H_fr_ref =  H_fr_ref
        self.H_hl_ref =  H_hl_ref
        self.H_hr_ref =  H_hr_ref
        
        
        self.postureTask = postureTask
        
        self.formulation = formulation
        self.data = data
        self.solver = solver
        self.q = q
        self.qdot = qdot
        
        
        if viewer:
            self.robot_display = pin.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path], pin.JointModelFreeFlyer())
            if viewer == pin.visualize.GepettoVisualizer:
                import gepetto.corbaserver
                launched = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
                if int(launched[1]) == 0:
                    os.system('gepetto-gui &')
                time.sleep(1)
                self.viz = viewer(self.robot_display.model, self.robot_display.collision_model,
                                  self.robot_display.visual_model)
                self.viz.initViewer(loadModel=True)
                self.viz.displayCollisions(False)
                self.viz.displayVisuals(True)
                self.viz.display(q)

                self.gui = self.viz.viewer.gui
                # self.gui.setCameraTransform(0, conf.CAMERA_TRANSFORM)
                self.gui.addFloor('world/floor')
                self.gui.setLightingMode('world/floor', 'OFF')
            elif viewer == pin.visualize.MeshcatVisualizer:
                self.viz = viewer(self.robot_display.model, self.robot_display.collision_model,
                                  self.robot_display.visual_model)
                self.viz.initViewer(loadModel=True)
                self.viz.display(q)
                
    def set_com_ref(self, pos, vel, acc):
        self.sample_COM_pos[:3] = pos
        self.sample_COM_vel[:3] = vel
        self.sample_COM_acc[:3] = acc
        
        self.sampleCOM.pos(self.sample_COM_pos)
        self.sampleCOM.vel(self.sample_COM_vel)
        self.sampleCOM.acc(self.sample_COM_acc)
        
        self.COMTask.setReference(self.sampleCOM)
        
    def set_FL_ref(self, pos, vel, acc):
        self.sample_FL_pos[:3] = pos
        self.sample_FL_vel[:3] = vel
        self.sample_FL_acc[:3] = acc
        
        self.sampleFL.pos(self.sample_FL_pos)
        self.sampleFL.vel(self.sample_FL_vel)
        self.sampleFL.acc(self.sample_FL_acc)
        
        self.FLTask.setReference(self.sampleFL)
        
    def set_FR_ref(self, pos, vel, acc):
        self.sample_FR_pos[:3] = pos
        self.sample_FR_vel[:3] = vel
        self.sample_FR_acc[:3] = acc
        
        self.sampleFR.pos(self.sample_FR_pos)
        self.sampleFR.vel(self.sample_FR_vel)
        self.sampleFR.acc(self.sample_FR_acc)
        
        self.FRTask.setReference(self.sampleFR)
        
    def set_HL_ref(self, pos, vel, acc):
        self.sample_HL_pos[:3] = pos
        self.sample_HL_vel[:3] = vel
        self.sample_HL_acc[:3] = acc
        
        self.sampleHL.pos(self.sample_HL_pos)
        self.sampleHL.vel(self.sample_HL_vel)
        self.sampleHL.acc(self.sample_HL_acc)
        
        self.HLTask.setReference(self.sampleHL)
        
    def set_HR_ref(self, pos, vel, acc):
        self.sample_HR_pos[:3] = pos
        self.sample_HR_vel[:3] = vel
        self.sample_HR_acc[:3] = acc
        
        self.sampleHR.pos(self.sample_HR_pos)
        self.sampleHR.vel(self.sample_HR_vel)
        self.sampleHR.acc(self.sample_HR_acc)
        
        self.HRTask.setReference(self.sampleHR)
        
    def get_FL_pos(self):
        data = self.formulation.data()
        H = self.robot.framePosition(data, self.FL)
        
        return H.translation
    
    def get_FR_pos(self):
        data = self.formulation.data()
        H = self.robot.framePosition(data, self.FR)
        
        return H.translation
    
    def get_HL_pos(self):
        data = self.formulation.data()
        H = self.robot.framePosition(data, self.HL)
        
        return H.translation
        
    def get_HR_pos(self):
        data = self.formulation.data()
        H = self.robot.framePosition(data, self.HR)
        
        return H.translation
        
    def add_contact_FL(self, transition_time=0.0):
        H_fl_ref = self.robot.framePosition(self.formulation.data(), self.FL)
        self.contact_FL.setReference(H_fl_ref)
        if self.conf.w_contact >= 0.0:
            self.formulation.addRigidContact(self.contact_FL, self.conf.w_forceRef, self.conf.w_contact, 1)
        else:
            self.formulation.addRigidContact(self.contact_FL, self.conf.w_forceRef)
    
        self.contact_FL_active = True
    
    def add_contact_FR(self, transition_time=0.0):
        H_fr_ref = self.robot.framePosition(self.formulation.data(), self.FR)
        self.contact_FR.setReference(H_fr_ref)
        if self.conf.w_contact >= 0.0:
            self.formulation.addRigidContact(self.contact_FR, self.conf.w_forceRef, self.conf.w_contact, 1)
        else:
            self.formulation.addRigidContact(self.contact_FR, self.conf.w_forceRef)
    
        self.contact_FR_active = True
        
    def add_contact_HL(self, transition_time=0.0):
        H_hl_ref = self.robot.framePosition(self.formulation.data(), self.HL)
        self.contact_HL.setReference(H_hl_ref)
        if self.conf.w_contact >= 0.0:
            self.formulation.addRigidContact(self.contact_HL, self.conf.w_forceRef, self.conf.w_contact, 1)
        else:
            self.formulation.addRigidContact(self.contact_HL, self.conf.w_forceRef)
    
        self.contact_HL_active = True
        
    def add_contact_HR(self, transition_time=0.0):
        H_hr_ref = self.robot.framePosition(self.formulation.data(), self.HR)
        self.contact_HR.setReference(H_hr_ref)
        if self.conf.w_contact >= 0.0:
            self.formulation.addRigidContact(self.contact_HR, self.conf.w_forceRef, self.conf.w_contact, 1)
        else:
            self.formulation.addRigidContact(self.contact_HR, self.conf.w_forceRef)
    
        self.contact_HR_active = True
        
    def remove_contact_FL(self, transition_time = 0.0):
        H_fl_ref = self.robot.framePosition(self.formulation.data(), self.FL)
        self.FL_traj.setReference(H_fl_ref)
        self.FLTask.setReference(self.FL_traj.computeNext())
        
        self.formulation.removeRigidContact(self.contacts[0].name, transition_time)
        self.contact_FL_active = False
        
    def remove_contact_FR(self, transition_time = 0.0):
        H_fr_ref = self.robot.framePosition(self.formulation.data(), self.FR)
        self.FR_traj.setReference(H_fr_ref)
        self.FRTask.setReference(self.FR_traj.computeNext())
        
        self.formulation.removeRigidContact(self.contacts[1].name, transition_time)
        self.contact_FR_active = False
        
    def remove_contact_HL(self, transition_time = 0.0):
        H_hl_ref = self.robot.framePosition(self.formulation.data(), self.HL)
        self.HL_traj.setReference(H_hl_ref)
        self.HLTask.setReference(self.HL_traj.computeNext())
        
        self.formulation.removeRigidContact(self.contacts[2].name, transition_time)
        self.contact_HL_active = False
        
    def remove_contact_HR(self, transition_time = 0.0):
        H_hr_ref = self.robot.framePosition(self.formulation.data(), self.HR)
        self.HR_traj.setReference(H_hr_ref)
        self.HRTask.setReference(self.HR_traj.computeNext())
        
        self.formulation.removeRigidContact(self.contacts[3].name, transition_time)
        self.contact_HR_active = False
        
    def integrate_dv(self, q, v, dv, dt):
        v_mean = v + 0.5 * dt * dv
        v += dt * dv
        q = pin.integrate(self.model, q, dt * v_mean)
        return q, v
    
    def display(self, q):
        if hasattr(self, 'viz'):
            self.viz.display(q)
            
            
            
