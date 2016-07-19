#!/usr/bin/env python

import rosbag
import numpy as np
from scipy import io
import sys
import os
import glob
import cv2
import argparse
import multiprocessing as mp
import time
import itertools


def extractBagAndSave(args):
    filename,que,pID = args
    filepath = filename[:-4]
    bag = rosbag.Bag(filename)
    
    numTotalMsgs = 0
    numMsgs = {}
    
    for topic, msg, t in bag.read_messages():
        # Send update through queue
        if not numTotalMsgs%100:
            que.put((pID,0))
        
        numTotalMsgs+=1
        
        if topic in numMsgs.keys():
            numMsgs[topic] += 1
        else:
            numMsgs[topic] = 1
            if topic == '/markerImage':
                if msg.encoding == 'mono8':
                    channels = 1
                elif msg.encoding == 'rgb8' or msg.encoding == 'bgr8':
                    channels = 3
                cols = msg.step/channels
                rows = len(msg.data)/(channels*cols)
            if ((topic == '/output') or (topic == '/ugv0/output') or (topic == '/ugv1/output')):
                thetaLen = len(msg.thetaHat) if hasattr(msg,'thetaHat') else 0
    
    # Initialize output data arrays
    if '/output' in numMsgs.keys():
        numOutputMsgs = numMsgs['/output']
        outputTime = np.zeros(numOutputMsgs)
        y = np.zeros((numOutputMsgs,3))
        yhat = np.zeros((numOutputMsgs,3))
        error = np.zeros((numOutputMsgs,3))
        XYZ = np.zeros((numOutputMsgs,3))
        XYZhat = np.zeros((numOutputMsgs,3))
        XYZerror = np.zeros((numOutputMsgs,3))
        q = np.zeros((numOutputMsgs,4))
        qhat = np.zeros((numOutputMsgs,4))
        qError = np.zeros((numOutputMsgs,4))
        phi = np.zeros((numOutputMsgs,7))
        phiHat = np.zeros((numOutputMsgs,7))
        thetaHat = np.zeros((numOutputMsgs,thetaLen))
        estimatorOn = np.zeros(numOutputMsgs)
        usePredictor = np.zeros(numOutputMsgs)
        deadReckoning = np.zeros(numOutputMsgs)
        normalizedKinematics = np.zeros(numOutputMsgs)
        artificialSwitching = np.zeros(numOutputMsgs)
        useVelocityMap = np.zeros(numOutputMsgs)
        streets = np.zeros(numOutputMsgs)
        multiBot = np.zeros(numOutputMsgs)
        delTon = np.zeros(numOutputMsgs)
        delToff = np.zeros(numOutputMsgs)
    
    # Initialize output data arrays
    if '/exp/output' in numMsgs.keys():
        numOutputMsgs = numMsgs['/exp/output']
        outputTimeExp = np.zeros(numOutputMsgs)
        yExp = np.zeros((numOutputMsgs,3))
        yhatExp = np.zeros((numOutputMsgs,3))
        errorExp = np.zeros((numOutputMsgs,3))
        XYZExp = np.zeros((numOutputMsgs,3))
        XYZhatExp = np.zeros((numOutputMsgs,3))
        XYZerrorExp = np.zeros((numOutputMsgs,3))
        estimatorOnExp = np.zeros(numOutputMsgs)
        usePredictorExp = np.zeros(numOutputMsgs)
        deadReckoningExp = np.zeros(numOutputMsgs)
        normalizedKinematicsExp = np.zeros(numOutputMsgs)
        artificialSwitchingExp = np.zeros(numOutputMsgs)
        useVelocityMapExp = np.zeros(numOutputMsgs)
        delTonExp = np.zeros(numOutputMsgs)
        delToffExp = np.zeros(numOutputMsgs)
    
    # Initialize output data arrays
    if '/ekf/output' in numMsgs.keys():
        numOutputMsgs = numMsgs['/ekf/output']
        outputTimeEkf = np.zeros(numOutputMsgs)
        yEkf = np.zeros((numOutputMsgs,3))
        yhatEkf = np.zeros((numOutputMsgs,3))
        errorEkf = np.zeros((numOutputMsgs,3))
        XYZEkf = np.zeros((numOutputMsgs,3))
        XYZhatEkf = np.zeros((numOutputMsgs,3))
        XYZerrorEkf = np.zeros((numOutputMsgs,3))
        estimatorOnEkf = np.zeros(numOutputMsgs)
        usePredictorEkf = np.zeros(numOutputMsgs)
        deadReckoningEkf = np.zeros(numOutputMsgs)
        normalizedKinematicsEkf = np.zeros(numOutputMsgs)
        artificialSwitchingEkf = np.zeros(numOutputMsgs)
        useVelocityMapEkf = np.zeros(numOutputMsgs)
        delTonEkf = np.zeros(numOutputMsgs)
        delToffEkf = np.zeros(numOutputMsgs)
    
    # Initialize output data arrays
    if '/ugv0/output' in numMsgs.keys():
        numOutputMsgs = numMsgs['/ugv0/output']
        outputTimeUGV0 = np.zeros(numOutputMsgs)
        yUGV0 = np.zeros((numOutputMsgs,3))
        yhatUGV0 = np.zeros((numOutputMsgs,3))
        errorUGV0 = np.zeros((numOutputMsgs,3))
        XYZUGV0 = np.zeros((numOutputMsgs,3))
        XYZhatUGV0 = np.zeros((numOutputMsgs,3))
        XYZerrorUGV0 = np.zeros((numOutputMsgs,3))
        qUGV0 = np.zeros((numOutputMsgs,4))
        qhatUGV0 = np.zeros((numOutputMsgs,4))
        qErrorUGV0 = np.zeros((numOutputMsgs,4))
        phiUGV0 = np.zeros((numOutputMsgs,7))
        phiHatUGV0 = np.zeros((numOutputMsgs,7))
        thetaHatUGV0 = np.zeros((numOutputMsgs,thetaLen))
        estimatorOnUGV0 = np.zeros(numOutputMsgs)
        usePredictorUGV0 = np.zeros(numOutputMsgs)
        deadReckoningUGV0 = np.zeros(numOutputMsgs)
        normalizedKinematicsUGV0 = np.zeros(numOutputMsgs)
        artificialSwitchingUGV0 = np.zeros(numOutputMsgs)
        useVelocityMapUGV0 = np.zeros(numOutputMsgs)
        streetsUGV0 = np.zeros(numOutputMsgs)
        multiBotUGV0 = np.zeros(numOutputMsgs)
        delTonUGV0 = np.zeros(numOutputMsgs)
        delToffUGV0 = np.zeros(numOutputMsgs)
    
    # Initialize output data arrays
    if '/ugv1/output' in numMsgs.keys():
        numOutputMsgs = numMsgs['/ugv1/output']
        outputTimeUGV1 = np.zeros(numOutputMsgs)
        yUGV1 = np.zeros((numOutputMsgs,3))
        yhatUGV1 = np.zeros((numOutputMsgs,3))
        errorUGV1 = np.zeros((numOutputMsgs,3))
        XYZUGV1 = np.zeros((numOutputMsgs,3))
        XYZhatUGV1 = np.zeros((numOutputMsgs,3))
        XYZerrorUGV1 = np.zeros((numOutputMsgs,3))
        qUGV1 = np.zeros((numOutputMsgs,4))
        qhatUGV1 = np.zeros((numOutputMsgs,4))
        qErrorUGV1 = np.zeros((numOutputMsgs,4))
        phiUGV1 = np.zeros((numOutputMsgs,7))
        phiHatUGV1 = np.zeros((numOutputMsgs,7))
        thetaHatUGV1 = np.zeros((numOutputMsgs,thetaLen))
        estimatorOnUGV1 = np.zeros(numOutputMsgs)
        usePredictorUGV1 = np.zeros(numOutputMsgs)
        deadReckoningUGV1 = np.zeros(numOutputMsgs)
        normalizedKinematicsUGV1 = np.zeros(numOutputMsgs)
        artificialSwitchingUGV1 = np.zeros(numOutputMsgs)
        useVelocityMapUGV1 = np.zeros(numOutputMsgs)
        streetsUGV1 = np.zeros(numOutputMsgs)
        multiBotUGV1 = np.zeros(numOutputMsgs)
        delTonUGV1 = np.zeros(numOutputMsgs)
        delToffUGV1 = np.zeros(numOutputMsgs)
    
    # Initialized camera pose data arrays
    if '/image/pose' in numMsgs.keys():
        numCamPoseMsgs = numMsgs['/image/pose']
        camPoseTime = np.zeros(numCamPoseMsgs)
        camPoseT = np.zeros((numCamPoseMsgs,3))
        camPoseQ = np.zeros((numCamPoseMsgs,4))
    
    # Initialized camera velocity data arrays
    if '/image/body_vel' in numMsgs.keys():
        numCamVelMsgs = numMsgs['/image/body_vel']
        camVelTime = np.zeros(numCamVelMsgs)
        camVelLinear = np.zeros((numCamVelMsgs,3))
        camVelAngular = np.zeros((numCamVelMsgs,3))
    
    # Initialized camera pose data arrays
    if '/bebop_image/pose' in numMsgs.keys():
        numCamPoseMsgs = numMsgs['/bebop_image/pose']
        camPoseTime = np.zeros(numCamPoseMsgs)
        camPoseT = np.zeros((numCamPoseMsgs,3))
        camPoseQ = np.zeros((numCamPoseMsgs,4))
    
    # Initialized camera velocity data arrays
    if '/bebop_image/body_vel' in numMsgs.keys():
        numCamVelMsgs = numMsgs['/bebop_image/body_vel']
        camVelTime = np.zeros(numCamVelMsgs)
        camVelLinear = np.zeros((numCamVelMsgs,3))
        camVelAngular = np.zeros((numCamVelMsgs,3))
    
    # Initialized target pose data arrays
    if '/ugv0/pose' in numMsgs.keys():
        numTargetPoseMsgs = numMsgs['/ugv0/pose']
        targetPoseTime = np.zeros(numTargetPoseMsgs)
        targetPoseT = np.zeros((numTargetPoseMsgs,3))
        targetPoseQ = np.zeros((numTargetPoseMsgs,4))
    
    # Initialized target velocity data arrays
    if '/ugv0/body_vel' in numMsgs.keys(): # Change this back to '/ugv0/odom' for predictor/ZOH experiments
        numTargetVelMsgs = numMsgs['/ugv0/body_vel']  # Change this back to '/ugv0/odom' for predictor/ZOH experiments
        targetVelTime = np.zeros(numTargetVelMsgs)
        targetVelLinear = np.zeros((numTargetVelMsgs,3))
        targetVelAngular = np.zeros((numTargetVelMsgs,3))
    
    # Initialized target pose data arrays
    if '/ugv1/pose' in numMsgs.keys():
        numTargetPoseMsgs = numMsgs['/ugv1/pose']
        target2PoseTime = np.zeros(numTargetPoseMsgs)
        target2PoseT = np.zeros((numTargetPoseMsgs,3))
        target2PoseQ = np.zeros((numTargetPoseMsgs,4))
    
    # Initialized target velocity data arrays
    if '/ugv1/body_vel' in numMsgs.keys(): # Change this back to '/ugv0/odom' for predictor/ZOH experiments
        numTargetVelMsgs = numMsgs['/ugv1/body_vel']  # Change this back to '/ugv0/odom' for predictor/ZOH experiments
        target2VelTime = np.zeros(numTargetVelMsgs)
        target2VelLinear = np.zeros((numTargetVelMsgs,3))
        target2VelAngular = np.zeros((numTargetVelMsgs,3))
    
    # Initialized target pose measurement (pose from ArUco) data arrays
    if '/markers' in numMsgs.keys():
        numTargetPoseMeasMsgs = numMsgs['/markers']
        targetPoseMeasTime = np.zeros(numTargetPoseMeasMsgs)
        targetPoseMeasT = np.zeros((numTargetPoseMeasMsgs,3))
        targetPoseMeasQ = np.zeros((numTargetPoseMeasMsgs,4))
    
    #video writer object
    # MPEG1 only supports certain frame rates. Use dummy value. Actual timestamps of each frame are saved to account for variable frame rate
    if '/markerImage' in numMsgs.keys():
        numMarkerImageMsgs = numMsgs['/markerImage']
        
        # OpenCV 2
        #vidWriter = cv2.VideoWriter(filename=filepath'.mpg',fourcc=cv2.cv.CV_FOURCC('P','I','M','1'),fps=30,frameSize=(cols,rows))
        
        # OpenCV 3
        vidWriter = cv2.VideoWriter(filename=filepath+'.avi',fourcc=cv2.VideoWriter_fourcc(*'XVID'),fps=30,frameSize=(cols,rows))
        
        vidTime = np.zeros(numMarkerImageMsgs)
    
    when = max(int(round(0.01*numTotalMsgs)),1) # send progress report at this interval
    ind = 0
    indices = dict.fromkeys(numMsgs.keys(),0)
    for topic, msg, t in bag.read_messages():
        if not ind%when:
            #sys.stdout.write('Percent done: '+str(round(float(ind)/float(numTotalMsgs)*100)) + '%\r')
            #sys.stdout.flush()
            
            # Communicate status to parent
            percent = float(ind)/float(numTotalMsgs)*100
            que.put((pID,percent))
        
        if topic == '/output':
            # Save timestamp
            outputTime[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            y[indices[topic],:] = np.array(msg.y)
            yhat[indices[topic],:] = np.array(msg.yhat)
            error[indices[topic],:] = np.array(msg.error)
            XYZ[indices[topic],:] = np.array(msg.XYZ)
            XYZhat[indices[topic],:] = np.array(msg.XYZhat)
            XYZerror[indices[topic],:] = np.array(msg.XYZerror)
            q[indices[topic],:] = np.array([msg.q[3],msg.q[0],msg.q[1],msg.q[2]]) if hasattr(msg,'q') else np.zeros(4)
            qhat[indices[topic],:] = np.array([msg.qhat[3],msg.qhat[0],msg.qhat[1],msg.qhat[2]]) if hasattr(msg,'qhat') else np.zeros(4)
            qError[indices[topic],:] = np.array([msg.qError[3],msg.qError[0],msg.qError[1],msg.qError[2]]) if hasattr(msg,'qError') else np.zeros(4)
            phi[indices[topic],:] = np.array(msg.phi) if hasattr(msg,'phi') else np.zeros(7)
            phiHat[indices[topic],:] = np.array(msg.phiHat) if hasattr(msg,'phiHat') else np.zeros(7)
            thetaHat[indices[topic],:] = np.array(msg.thetaHat) if hasattr(msg,'thetaHat') else np.zeros(thetaLen)
            estimatorOn[indices[topic]] = np.array(1.0*msg.estimatorOn) # convert boolean to numeric array due to scipy bug in savemat
            usePredictor[indices[topic]] = np.array(1.0*msg.usePredictor)
            deadReckoning[indices[topic]] = np.array(1.0*msg.deadReckoning) if hasattr(msg,'deadReckoning') else 0.0
            normalizedKinematics[indices[topic]] = np.array(1.0*msg.normalizedKinematics) if hasattr(msg,'normalizedKinematics') else 0.0
            artificialSwitching[indices[topic]] = np.array(1.0*msg.artificialSwitching) if hasattr(msg,'artificialSwitching') else 0.0
            useVelocityMap[indices[topic]] = np.array(1.0*msg.useVelocityMap) if hasattr(msg,'useVelocityMap') else 0.0
            streets[indices[topic]] = np.array(1.0*msg.streets) if hasattr(msg,'streets') else 0.0
            multiBot[indices[topic]] = np.array(1.0*msg.multiBot) if hasattr(msg,'multiBot') else 0.0
            delTon[indices[topic]] = np.array(msg.delTon) if hasattr(msg,'delTon') else 0.0
            delToff[indices[topic]] = np.array(msg.delToff) if hasattr(msg,'delToff') else 0.0
        
        elif topic == '/exp/output':
            # Save timestamp
            outputTimeExp[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            yExp[indices[topic],:] = np.array(msg.y)
            yhatExp[indices[topic],:] = np.array(msg.yhat)
            errorExp[indices[topic],:] = np.array(msg.error)
            XYZExp[indices[topic],:] = np.array(msg.XYZ)
            XYZhatExp[indices[topic],:] = np.array(msg.XYZhat)
            XYZerrorExp[indices[topic],:] = np.array(msg.XYZerror)
            estimatorOnExp[indices[topic]] = np.array(1.0*msg.estimatorOn) # convert boolean to numeric array due to scipy bug in savemat
            usePredictorExp[indices[topic]] = np.array(1.0*msg.usePredictor)
            deadReckoningExp[indices[topic]] = np.array(1.0*msg.deadReckoning) if hasattr(msg,'deadReckoning') else 0.0
            normalizedKinematicsExp[indices[topic]] = np.array(1.0*msg.normalizedKinematics) if hasattr(msg,'normalizedKinematics') else 0.0
            artificialSwitchingExp[indices[topic]] = np.array(1.0*msg.artificialSwitching) if hasattr(msg,'artificialSwitching') else 0.0
            useVelocityMapExp[indices[topic]] = np.array(1.0*msg.useVelocityMap) if hasattr(msg,'useVelocityMap') else 0.0
            delTonExp[indices[topic]] = np.array(msg.delTon) if hasattr(msg,'delTon') else 0.0
            delToffExp[indices[topic]] = np.array(msg.delToff) if hasattr(msg,'delToff') else 0.0
        
        elif topic == '/ekf/output':
            # Save timestamp
            outputTimeEkf[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            yEkf[indices[topic],:] = np.array(msg.y)
            yhatEkf[indices[topic],:] = np.array(msg.yhat)
            errorEkf[indices[topic],:] = np.array(msg.error)
            XYZEkf[indices[topic],:] = np.array(msg.XYZ)
            XYZhatEkf[indices[topic],:] = np.array(msg.XYZhat)
            XYZerrorEkf[indices[topic],:] = np.array(msg.XYZerror)
            estimatorOnEkf[indices[topic]] = np.array(1.0*msg.estimatorOn) # convert boolean to numeric array due to scipy bug in savemat
            usePredictorEkf[indices[topic]] = np.array(1.0*msg.usePredictor)
            deadReckoningEkf[indices[topic]] = np.array(1.0*msg.deadReckoning) if hasattr(msg,'deadReckoning') else 0.0
            normalizedKinematicsEkf[indices[topic]] = np.array(1.0*msg.normalizedKinematics) if hasattr(msg,'normalizedKinematics') else 0.0
            artificialSwitchingEkf[indices[topic]] = np.array(1.0*msg.artificialSwitching) if hasattr(msg,'artificialSwitching') else 0.0
            useVelocityMapEkf[indices[topic]] = np.array(1.0*msg.useVelocityMap) if hasattr(msg,'useVelocityMap') else 0.0
            delTonEkf[indices[topic]] = np.array(msg.delTon) if hasattr(msg,'delTon') else 0.0
            delToffEkf[indices[topic]] = np.array(msg.delToff) if hasattr(msg,'delToff') else 0.0
        
        elif topic == '/ugv0/output':
            # Save timestamp
            outputTimeUGV0[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            yUGV0[indices[topic],:] = np.array(msg.y)
            yhatUGV0[indices[topic],:] = np.array(msg.yhat)
            errorUGV0[indices[topic],:] = np.array(msg.error)
            XYZUGV0[indices[topic],:] = np.array(msg.XYZ)
            XYZhatUGV0[indices[topic],:] = np.array(msg.XYZhat)
            XYZerrorUGV0[indices[topic],:] = np.array(msg.XYZerror)
            qUGV0[indices[topic],:] = np.array([msg.q[3],msg.q[0],msg.q[1],msg.q[2]]) if hasattr(msg,'q') else np.zeros(4)
            qhatUGV0[indices[topic],:] = np.array([msg.qhat[3],msg.qhat[0],msg.qhat[1],msg.qhat[2]]) if hasattr(msg,'qhat') else np.zeros(4)
            qErrorUGV0[indices[topic],:] = np.array([msg.qError[3],msg.qError[0],msg.qError[1],msg.qError[2]]) if hasattr(msg,'qError') else np.zeros(4)
            phiUGV0[indices[topic],:] = np.array(msg.phi) if hasattr(msg,'phi') else np.zeros(7)
            phiHatUGV0[indices[topic],:] = np.array(msg.phiHat) if hasattr(msg,'phiHat') else np.zeros(7)
            thetaHatUGV0[indices[topic],:] = np.array(msg.thetaHat) if hasattr(msg,'thetaHat') else np.zeros(thetaLen)
            estimatorOnUGV0[indices[topic]] = np.array(1.0*msg.estimatorOn) # convert boolean to numeric array due to scipy bug in savemat
            usePredictorUGV0[indices[topic]] = np.array(1.0*msg.usePredictor)
            deadReckoningUGV0[indices[topic]] = np.array(1.0*msg.deadReckoning) if hasattr(msg,'deadReckoning') else 0.0
            normalizedKinematicsUGV0[indices[topic]] = np.array(1.0*msg.normalizedKinematics) if hasattr(msg,'normalizedKinematics') else 0.0
            artificialSwitchingUGV0[indices[topic]] = np.array(1.0*msg.artificialSwitching) if hasattr(msg,'artificialSwitching') else 0.0
            useVelocityMapUGV0[indices[topic]] = np.array(1.0*msg.useVelocityMap) if hasattr(msg,'useVelocityMap') else 0.0
            streetsUGV0[indices[topic]] = np.array(1.0*msg.streets) if hasattr(msg,'streets') else 0.0
            multiBotUGV0[indices[topic]] = np.array(1.0*msg.multiBot) if hasattr(msg,'multiBot') else 0.0
            delTonUGV0[indices[topic]] = np.array(msg.delTon) if hasattr(msg,'delTon') else 0.0
            delToffUGV0[indices[topic]] = np.array(msg.delToff) if hasattr(msg,'delToff') else 0.0
        
        elif topic == '/ugv1/output':
            # Save timestamp
            outputTimeUGV1[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            yUGV1[indices[topic],:] = np.array(msg.y)
            yhatUGV1[indices[topic],:] = np.array(msg.yhat)
            errorUGV1[indices[topic],:] = np.array(msg.error)
            XYZUGV1[indices[topic],:] = np.array(msg.XYZ)
            XYZhatUGV1[indices[topic],:] = np.array(msg.XYZhat)
            XYZerrorUGV1[indices[topic],:] = np.array(msg.XYZerror)
            qUGV1[indices[topic],:] = np.array([msg.q[3],msg.q[0],msg.q[1],msg.q[2]]) if hasattr(msg,'q') else np.zeros(4)
            qhatUGV1[indices[topic],:] = np.array([msg.qhat[3],msg.qhat[0],msg.qhat[1],msg.qhat[2]]) if hasattr(msg,'qhat') else np.zeros(4)
            qErrorUGV1[indices[topic],:] = np.array([msg.qError[3],msg.qError[0],msg.qError[1],msg.qError[2]]) if hasattr(msg,'qError') else np.zeros(4)
            phiUGV1[indices[topic],:] = np.array(msg.phi) if hasattr(msg,'phi') else np.zeros(7)
            phiHatUGV1[indices[topic],:] = np.array(msg.phiHat) if hasattr(msg,'phiHat') else np.zeros(7)
            thetaHatUGV1[indices[topic],:] = np.array(msg.thetaHat) if hasattr(msg,'thetaHat') else np.zeros(thetaLen)
            estimatorOnUGV1[indices[topic]] = np.array(1.0*msg.estimatorOn) # convert boolean to numeric array due to scipy bug in savemat
            usePredictorUGV1[indices[topic]] = np.array(1.0*msg.usePredictor)
            deadReckoningUGV1[indices[topic]] = np.array(1.0*msg.deadReckoning) if hasattr(msg,'deadReckoning') else 0.0
            normalizedKinematicsUGV1[indices[topic]] = np.array(1.0*msg.normalizedKinematics) if hasattr(msg,'normalizedKinematics') else 0.0
            artificialSwitchingUGV1[indices[topic]] = np.array(1.0*msg.artificialSwitching) if hasattr(msg,'artificialSwitching') else 0.0
            useVelocityMapUGV1[indices[topic]] = np.array(1.0*msg.useVelocityMap) if hasattr(msg,'useVelocityMap') else 0.0
            streetsUGV1[indices[topic]] = np.array(1.0*msg.streets) if hasattr(msg,'streets') else 0.0
            multiBotUGV1[indices[topic]] = np.array(1.0*msg.multiBot) if hasattr(msg,'multiBot') else 0.0
            delTonUGV1[indices[topic]] = np.array(msg.delTon) if hasattr(msg,'delTon') else 0.0
            delToffUGV1[indices[topic]] = np.array(msg.delToff) if hasattr(msg,'delToff') else 0.0
        
        elif topic == '/image/pose':
            # Save timestamp
            camPoseTime[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            pose = msg.pose
            camPoseT[indices[topic],:] = np.array([pose.position.x,pose.position.y,pose.position.z])
            camPoseQ[indices[topic],:] = np.array([pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z])
            
        elif topic == '/image/body_vel':
            # Save timestamp
            camVelTime[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            twist = msg.twist
            camVelLinear[indices[topic],:] = np.array([twist.linear.x,twist.linear.y,twist.linear.z])
            camVelAngular[indices[topic],:] = np.array([twist.angular.x,twist.angular.y,twist.angular.z])
        
        elif topic == '/bebop_image/pose':
            # Save timestamp
            camPoseTime[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            pose = msg.pose
            camPoseT[indices[topic],:] = np.array([pose.position.x,pose.position.y,pose.position.z])
            camPoseQ[indices[topic],:] = np.array([pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z])
            
        elif topic == '/bebop_image/body_vel':
            # Save timestamp
            camVelTime[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            twist = msg.twist
            camVelLinear[indices[topic],:] = np.array([twist.linear.x,twist.linear.y,twist.linear.z])
            camVelAngular[indices[topic],:] = np.array([twist.angular.x,twist.angular.y,twist.angular.z])
            
        elif topic == '/ugv0/pose':
            # Save timestamp
            targetPoseTime[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            pose = msg.pose
            targetPoseT[indices[topic],:] = np.array([pose.position.x,pose.position.y,pose.position.z])
            targetPoseQ[indices[topic],:] = np.array([pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z])
            
        elif topic == '/ugv0/body_vel':# Change this back to '/ugv0/odom' for predictor/ZOH experiments
            # Save timestamp
            targetVelTime[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            twist = msg.twist # Change this back to msg.twist.twist for predictor/ZOH experiments
            targetVelLinear[indices[topic],:] = np.array([twist.linear.x,twist.linear.y,twist.linear.z])
            targetVelAngular[indices[topic],:] = np.array([twist.angular.x,twist.angular.y,twist.angular.z])
        
        elif topic == '/ugv1/pose':
            # Save timestamp
            target2PoseTime[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            pose = msg.pose
            target2PoseT[indices[topic],:] = np.array([pose.position.x,pose.position.y,pose.position.z])
            target2PoseQ[indices[topic],:] = np.array([pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z])
            
        elif topic == '/ugv1/body_vel':# Change this back to '/ugv0/odom' for predictor/ZOH experiments
            # Save timestamp
            target2VelTime[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            twist = msg.twist # Change this back to msg.twist.twist for predictor/ZOH experiments
            target2VelLinear[indices[topic],:] = np.array([twist.linear.x,twist.linear.y,twist.linear.z])
            target2VelAngular[indices[topic],:] = np.array([twist.angular.x,twist.angular.y,twist.angular.z])
        
        elif topic == '/markers':
            # Save timestamp
            targetPoseMeasTime[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Save data
            pose = msg.pose
            targetPoseMeasT[indices[topic],:] = np.array([pose.position.x,pose.position.y,pose.position.z])
            targetPoseMeasQ[indices[topic],:] = np.array([pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z])
        
        elif topic == '/markerImage':
            # Save timestamp
            vidTime[indices[topic]] = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/float(1e9)
            
            # Write frames to video
            image = np.fromstring(msg.data,dtype=np.uint8)
            image.shape = (rows,cols,channels)
            cv2.putText(image,"t = "+str(np.round(vidTime[indices[topic]]-vidTime[0],3)),(1,rows),cv2.FONT_HERSHEY_PLAIN,1.5,(255,255,255),thickness=2)
            if msg.encoding == 'mono8':
                image = cv2.cvtColor(image,cv2.cv.CV_GRAY2RGB)
            elif msg.encoding == 'bgr8':
                #image = cv2.cvtColor(image,cv2.cv.CV_BGR2RGB)
                pass
            
            vidWriter.write(image)
        
        # Increment indices
        ind+=1
        indices[topic] +=1
    
    
    # release vidwriter object
    if '/markerImage' in numMsgs.keys():
        vidWriter.release()
    
    stuffToSave = {}
    if '/output' in numMsgs.keys():
        idx = np.argsort(outputTime) # Sort data to deal with slight out of order issues
        stuffToSave['outputTime'] = outputTime[idx]
        stuffToSave['y'] = y[idx,:]
        stuffToSave['yhat'] = yhat[idx,:]
        stuffToSave['error'] = error[idx,:]
        stuffToSave['XYZ'] = XYZ[idx,:]
        stuffToSave['XYZhat'] = XYZhat[idx,:]
        stuffToSave['XYZerror'] = XYZerror[idx,:]
        stuffToSave['q'] = q[idx,:]
        stuffToSave['qhat'] = qhat[idx,:]
        stuffToSave['qError'] = qError[idx,:]
        stuffToSave['phi'] = phi[idx,:]
        stuffToSave['phiHat'] = phiHat[idx,:]
        stuffToSave['thetaHat'] = thetaHat[idx,:]
        stuffToSave['estimatorOn'] = estimatorOn[idx]
        stuffToSave['usePredictor'] = usePredictor[idx]
        stuffToSave['deadReckoning'] = deadReckoning[idx]
        stuffToSave['normalizedKinematics'] = normalizedKinematics[idx]
        stuffToSave['artificialSwitching'] = artificialSwitching[idx]
        stuffToSave['useVelocityMap'] = useVelocityMap[idx]
        stuffToSave['streets'] = streets[idx]
        stuffToSave['multiBot'] = multiBot[idx]
        stuffToSave['delTon'] = delTon[idx]
        stuffToSave['delToff'] = delToff[idx]
    
    if '/exp/output' in numMsgs.keys():
        idx = np.argsort(outputTimeExp) # Sort data to deal with slight out of order issues
        stuffToSave['outputTimeExp'] = outputTimeExp[idx]
        stuffToSave['yExp'] = yExp[idx,:]
        stuffToSave['yhatExp'] = yhatExp[idx,:]
        stuffToSave['errorExp'] = errorExp[idx,:]
        stuffToSave['XYZExp'] = XYZExp[idx,:]
        stuffToSave['XYZhatExp'] = XYZhatExp[idx,:]
        stuffToSave['XYZerrorExp'] = XYZerrorExp[idx,:]
        stuffToSave['estimatorOnExp'] = estimatorOnExp[idx]
        stuffToSave['usePredictorExp'] = usePredictorExp[idx]
        stuffToSave['deadReckoningExp'] = deadReckoningExp[idx]
        stuffToSave['normalizedKinematicsExp'] = normalizedKinematicsExp[idx]
        stuffToSave['artificialSwitchingExp'] = artificialSwitchingExp[idx]
        stuffToSave['useVelocityMapExp'] = useVelocityMapExp[idx]
        stuffToSave['delTonExp'] = delTonExp[idx]
        stuffToSave['delToffExp'] = delToffExp[idx]
    
    if '/ekf/output' in numMsgs.keys():
        idx = np.argsort(outputTimeEkf) # Sort data to deal with slight out of order issues
        stuffToSave['outputTimeEkf'] = outputTimeEkf[idx]
        stuffToSave['yEkf'] = yEkf[idx,:]
        stuffToSave['yhatEkf'] = yhatEkf[idx,:]
        stuffToSave['errorEkf'] = errorEkf[idx,:]
        stuffToSave['XYZEkf'] = XYZEkf[idx,:]
        stuffToSave['XYZhatEkf'] = XYZhatEkf[idx,:]
        stuffToSave['XYZerrorEkf'] = XYZerrorEkf[idx,:]
        stuffToSave['estimatorOnEkf'] = estimatorOnEkf[idx]
        stuffToSave['usePredictorEkf'] = usePredictorEkf[idx]
        stuffToSave['deadReckoningEkf'] = deadReckoningEkf[idx]
        stuffToSave['normalizedKinematicsEkf'] = normalizedKinematicsEkf[idx]
        stuffToSave['artificialSwitchingEkf'] = artificialSwitchingEkf[idx]
        stuffToSave['useVelocityMapEkf'] = useVelocityMapEkf[idx]
        stuffToSave['delTonEkf'] = delTonEkf[idx]
        stuffToSave['delToffEkf'] = delToffEkf[idx]
    
    if '/ugv0/output' in numMsgs.keys():
        idx = np.argsort(outputTimeUGV0) # Sort data to deal with slight out of order issues
        stuffToSave['outputTimeUGV0'] = outputTimeUGV0[idx]
        stuffToSave['yUGV0'] = yUGV0[idx,:]
        stuffToSave['yhatUGV0'] = yhatUGV0[idx,:]
        stuffToSave['errorUGV0'] = errorUGV0[idx,:]
        stuffToSave['XYZUGV0'] = XYZUGV0[idx,:]
        stuffToSave['XYZhatUGV0'] = XYZhatUGV0[idx,:]
        stuffToSave['XYZerrorUGV0'] = XYZerrorUGV0[idx,:]
        stuffToSave['qUGV0'] = qUGV0[idx,:]
        stuffToSave['qhatUGV0'] = qhatUGV0[idx,:]
        stuffToSave['qErrorUGV0'] = qErrorUGV0[idx,:]
        stuffToSave['phiUGV0'] = phiUGV0[idx,:]
        stuffToSave['phiHatUGV0'] = phiHatUGV0[idx,:]
        stuffToSave['thetaHatUGV0'] = thetaHatUGV0[idx,:]
        stuffToSave['estimatorOnUGV0'] = estimatorOnUGV0[idx]
        stuffToSave['usePredictorUGV0'] = usePredictorUGV0[idx]
        stuffToSave['deadReckoningUGV0'] = deadReckoningUGV0[idx]
        stuffToSave['normalizedKinematicsUGV0'] = normalizedKinematicsUGV0[idx]
        stuffToSave['artificialSwitchingUGV0'] = artificialSwitchingUGV0[idx]
        stuffToSave['useVelocityMapUGV0'] = useVelocityMapUGV0[idx]
        stuffToSave['streetsUGV0'] = streetsUGV0[idx]
        stuffToSave['multiBotUGV0'] = multiBotUGV0[idx]
        stuffToSave['delTonUGV0'] = delTonUGV0[idx]
        stuffToSave['delToffUGV0'] = delToffUGV0[idx]
    
    if '/ugv1/output' in numMsgs.keys():
        idx = np.argsort(outputTimeUGV1) # Sort data to deal with slight out of order issues
        stuffToSave['outputTimeUGV1'] = outputTimeUGV1[idx]
        stuffToSave['yUGV1'] = yUGV1[idx,:]
        stuffToSave['yhatUGV1'] = yhatUGV1[idx,:]
        stuffToSave['errorUGV1'] = errorUGV1[idx,:]
        stuffToSave['XYZUGV1'] = XYZUGV1[idx,:]
        stuffToSave['XYZhatUGV1'] = XYZhatUGV1[idx,:]
        stuffToSave['XYZerrorUGV1'] = XYZerrorUGV1[idx,:]
        stuffToSave['qUGV1'] = qUGV1[idx,:]
        stuffToSave['qhatUGV1'] = qhatUGV1[idx,:]
        stuffToSave['qErrorUGV1'] = qErrorUGV1[idx,:]
        stuffToSave['phiUGV1'] = phiUGV1[idx,:]
        stuffToSave['phiHatUGV1'] = phiHatUGV1[idx,:]
        stuffToSave['thetaHatUGV1'] = thetaHatUGV1[idx,:]
        stuffToSave['estimatorOnUGV1'] = estimatorOnUGV1[idx]
        stuffToSave['usePredictorUGV1'] = usePredictorUGV1[idx]
        stuffToSave['deadReckoningUGV1'] = deadReckoningUGV1[idx]
        stuffToSave['normalizedKinematicsUGV1'] = normalizedKinematicsUGV1[idx]
        stuffToSave['artificialSwitchingUGV1'] = artificialSwitchingUGV1[idx]
        stuffToSave['useVelocityMapUGV1'] = useVelocityMapUGV1[idx]
        stuffToSave['streetsUGV1'] = streetsUGV1[idx]
        stuffToSave['multiBotUGV1'] = multiBotUGV1[idx]
        stuffToSave['delTonUGV1'] = delTonUGV1[idx]
        stuffToSave['delToffUGV1'] = delToffUGV1[idx]
    
    if '/image/pose' in numMsgs.keys():
        idx = np.argsort(camPoseTime) # Sort data to deal with slight out of order issues
        stuffToSave['camPoseTime'] = camPoseTime[idx]
        stuffToSave['camPoseT'] = camPoseT[idx,:]
        stuffToSave['camPoseQ'] = camPoseQ[idx,:]
    
    if '/image/body_vel' in numMsgs.keys():
        idx = np.argsort(camVelTime) # Sort data to deal with slight out of order issues
        stuffToSave['camVelTime'] = camVelTime[idx]
        stuffToSave['camVelLinear'] = camVelLinear[idx,:]
        stuffToSave['camVelAngular'] = camVelAngular[idx,:]
    
    if '/bebop_image/pose' in numMsgs.keys():
        idx = np.argsort(camPoseTime) # Sort data to deal with slight out of order issues
        stuffToSave['camPoseTime'] = camPoseTime[idx]
        stuffToSave['camPoseT'] = camPoseT[idx,:]
        stuffToSave['camPoseQ'] = camPoseQ[idx,:]
    
    if '/bebop_image/body_vel' in numMsgs.keys():
        idx = np.argsort(camVelTime) # Sort data to deal with slight out of order issues
        stuffToSave['camVelTime'] = camVelTime[idx]
        stuffToSave['camVelLinear'] = camVelLinear[idx,:]
        stuffToSave['camVelAngular'] = camVelAngular[idx,:]
    
    if '/ugv0/pose' in numMsgs.keys():
        idx = np.argsort(targetPoseTime) # Sort data to deal with slight out of order issues
        stuffToSave['targetPoseTime'] = targetPoseTime[idx]
        stuffToSave['targetPoseT'] = targetPoseT[idx,:]
        stuffToSave['targetPoseQ'] = targetPoseQ[idx,:]
    
    if '/ugv0/body_vel' in numMsgs.keys(): # Change this back to '/ugv0/odom' for predictor/ZOH experiments
        idx = np.argsort(targetVelTime) # Sort data to deal with slight out of order issues
        stuffToSave['targetVelTime'] = targetVelTime[idx]
        stuffToSave['targetVelLinear'] = targetVelLinear[idx,:]
        stuffToSave['targetVelAngular'] = targetVelAngular[idx,:]
    
    if '/ugv1/pose' in numMsgs.keys():
        idx = np.argsort(target2PoseTime) # Sort data to deal with slight out of order issues
        stuffToSave['target2PoseTime'] = target2PoseTime[idx]
        stuffToSave['target2PoseT'] = target2PoseT[idx,:]
        stuffToSave['target2PoseQ'] = target2PoseQ[idx,:]
    
    if '/ugv1/body_vel' in numMsgs.keys(): # Change this back to '/ugv0/odom' for predictor/ZOH experiments
        idx = np.argsort(target2VelTime) # Sort data to deal with slight out of order issues
        stuffToSave['target2VelTime'] = target2VelTime[idx]
        stuffToSave['target2VelLinear'] = target2VelLinear[idx,:]
        stuffToSave['target2VelAngular'] = target2VelAngular[idx,:]
    
    if '/markers' in numMsgs.keys():
        idx = np.argsort(targetPoseMeasTime) # Sort data to deal with slight out of order issues
        stuffToSave['targetPoseMeasTime'] = targetPoseMeasTime[idx]
        stuffToSave['targetPoseMeasT'] = targetPoseMeasT[idx,:]
        stuffToSave['targetPoseMeasQ'] = targetPoseMeasQ[idx,:]
    
    if '/markerImage' in numMsgs.keys():
        vidWriter.release()
        stuffToSave['vidTime'] = vidTime
    
    io.savemat(filepath+'.mat',stuffToSave,oned_as='column')
    
    que.put((pID,100))


# End extractBagAndSave

if __name__ == '__main__':
    #m = mp.Manager()
    #q = m.Queue()
    #extractBagAndSave(('model_learning/random_2016-06-17-14-30-24.bag',q,1))
    
    #print "done"
    
    #quit()
    
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unpack rosbags to mat file and video. Default will unpack newest file in current folder')
    parser.add_argument('--file','--f',help='Specify file to unpack') # Specify file
    parser.add_argument('--all','--a',nargs='?',const='',default=None,help='Unpack all files in current folder, or specified folder, which haven\'t already been unpacked (mat file already exists)') # Extract all in current folder, or specified folder
    parser.add_argument('--force',action='store_true',help='Force unpacking all bag files') # Extract all in current folder, or specified folder
    args = parser.parse_args()
    
    if args.all is not None: # Do all in folder
        fileList = sorted(glob.glob(os.path.join(args.all,'*.bag')))
        if not args.force:
            fileList = [fileName for fileName in fileList if not os.path.isfile(os.path.splitext(fileName)[0] + '.mat')]
    elif args.file is not None: # Specified file
        fileList = [args.file]
    else: # Most recent bag
        filepath = max(glob.iglob(folder+'/*.bag'),key=os.path.getctime)
        fileList = [filepath]
    nameLen = max(max([len(f) for f in fileList]),len('File'))
    numFiles = len(fileList)
    
    # Instantiate pool and queue for communication
    pool = mp.Pool(processes=4)
    m = mp.Manager()
    q = m.Queue()
    
    # Start processes
    result = pool.map_async(extractBagAndSave,itertools.izip(fileList,itertools.repeat(q),range(numFiles)))
    
    # Show progress in terminal
    status = [False]*numFiles
    while not result.ready():
        if not q.empty():
            # Get status updates from queue
            while not q.empty():
                update = q.get()
                status[update[0]] = update[1]
        
        # Write to screen
        os.system('clear')
        string = 'File'+(' '*(nameLen-2))+'Status'
        print string
        
        for state,f in zip(status,fileList):
            string = f+' '*(nameLen+2-len(f))
            if state is False: # Extraction not started yet
                string += 'Waiting...'
            elif state == 100: # Extraction completed
                string += 'Done'
            elif state == 0: # Skimming bag for number of messages
                string += 'Skimming...'
            else: # Percent completed
                numBars = np.floor(state/5.0)
                string += '<'+'='*numBars+' '*(20-numBars)+'> '+str(int(state))+'%'
            print string
        time.sleep(0.5)
    
    pool.close()
    pool.join()
    os.system('clear')
    print 'Done'


