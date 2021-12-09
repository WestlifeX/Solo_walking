#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 21:17:49 2021

@author: alessandro
"""
import pinocchio as pin
import tsid
import numpy as np
import numpy.matlib as matlib
import os
import gepetto.corbaserver
import time
import subprocess

class TsidQuadruped:
    def __init__(self, conf, viewer = True):
        self.conf = conf
        self.robot = tsid.RobotWrapper(conf.urdf, [conf.path], pin.JointModelFreeFlyer(),False)
        robot = self.robot
        self.model = model = robot.model()
        q = conf.q0
        qdot = np.zeros(robot.nv)
        
        formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        formulation.computeProblemData(0.0, q, qdot)
        data = formulation.data()
        
        #q[2] -= robot.framePosition(data, model.getFrameId(conf.contact_frames[0])).translation[2]
        robot.computeAllTerms(data, q, qdot)
               
        contacts = 4*[None]
        for i, name in enumerate(conf.contact_frames):
            contacts[i] =tsid.ContactPoint(name, robot, name, conf.contactNormal, conf.mu, conf.fMin, conf.fMax)
            contacts[i].setKp(conf.kp_contact * np.ones(3))
            contacts[i].setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(3))
            H_ref = robot.framePosition(data, robot.model().getFrameId(name))
            contacts[i].setReference(H_ref)
            contacts[i].useLocalFrame(False)
            formulation.addRigidContact(contacts[i], conf.w_forceRef, 1.0, 1)
     
        comTask = tsid.TaskComEquality("task-com", robot)
        comTask.setKp(conf.kp_com * np.ones(3))
        comTask.setKd(2.0 * np.sqrt(conf.kp_com) * np.ones(3))
        formulation.addMotionTask(comTask, conf.w_com, 1, 0.0)
        
        postureTask = tsid.TaskJointPosture("task-posture", robot)
        postureTask.setKp(conf.kp_posture * np.ones(robot.nv-6))
        postureTask.setKd(2.0 * np.sqrt(conf.kp_posture) * np.ones(robot.nv-6))
        formulation.addMotionTask(postureTask, conf.w_posture, 1, 0.0)
        
        q_ref = q
        self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)
        
        com_ref = robot.com(data)
        self.trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)
        self.sample_com = self.trajCom.computeNext()
        
   
        solver = tsid.SolverHQuadProgFast("qp_solver")
        solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)
        
        self.contact_FL = contacts[0]
        self.contact_FR = contacts[1]
        self.contact_HL = contacts[2]
        self.contact_HR = contacts[3]
        
        self.comTask = comTask
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
        
    def integrate_dv(self, q, v, dv, dt):
        v_mean = v + 0.5 * dt * dv
        v += dt * dv
        q = pin.integrate(self.model, q, dt * v_mean)
        return q, v
    
    def display(self, q):
        if hasattr(self, 'viz'):
            self.viz.display(q)
            
            
            
