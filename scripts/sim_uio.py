#!/usr/bin/env python

import rospy
import tf
import numpy as np
import itertools
import cv2
from aruco_ros.msg import Center
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import CameraInfo

qMult = tf.transformations.quaternion_multiply # quaternion multiplication function handle
qInv = tf.transformations.quaternion_inverse # quaternion inverse function handle
q2m = tf.transformations.quaternion_matrix # quaternion to 4x4 transformation matrix

markerID = 100;
#camMat = np.array([[558.953280,0.000000,365.566775],[0.000000,557.877582,240.157184],[0,0,1]])
#distCoeffs = np.array([-0.385156,0.163233,0.000539,0.000302,0.000000])
camMat = np.eye(3)
distCoeffs = np.zeros(5)
intRate = 50 # integration rate
velRate = 100 # velocity data publish rate
frameRate = 50 # marker publish rate

def sim():
    global t, pose, camInfoMsg
    global centerPub, velPub, camInfoPub, br, tfl
    
    rospy.init_node("sim")
    
    centerPub = rospy.Publisher("markerCenters",Center,queue_size=10)
    velPub = rospy.Publisher("image/local_vel",TwistStamped,queue_size=10)
    cameraName = rospy.get_param(rospy.get_name()+"/camera","camera")
    camInfoPub = rospy.Publisher(cameraName+"/camera_info",CameraInfo,queue_size=1)
    br = tf.TransformBroadcaster()
    tfl = tf.TransformListener()
    
    # Camera parameters
    camInfoMsg = CameraInfo()
    camInfoMsg.D = distCoeffs.tolist()
    camInfoMsg.K = camMat.reshape((-1)).tolist()
    
    # Wait for node to get cam info
    while (velPub.get_num_connections() == 0) and (not rospy.is_shutdown()):
        # publish camera parameters
        camInfoPub.publish(camInfoMsg)
        rospy.sleep(0.5)
    
    # Publishers
    rospy.Timer(rospy.Duration(1.0/velRate),velPubCB)
    rospy.Timer(rospy.Duration(1.0/frameRate),imagePubCB)
    rospy.Timer(rospy.Duration(0.5),camInfoPubCB)
    
    # Initial conditions
    startTime = rospy.get_time()
    camPos = np.array([0,-1,1.5])
    camOrient = np.array([-1*np.sqrt(2)/2,0,0,np.sqrt(2)/2])
    targetPos = np.array([-1.5,1.5,0])
    targetOrient = np.array([0,0,0,1])
    pose = np.concatenate((camPos,camOrient,targetPos,targetOrient))
    
    r = rospy.Rate(intRate)
    h = 1.0/intRate
    while not rospy.is_shutdown():
        t = np.array(rospy.get_time() - startTime)
        poseDot = poseDyn(t,pose)
        pose = pose + poseDot*h
        
        r.sleep()


def velPubCB(event):
    # Camera pose broadcaster
    camPos = pose[0:3]
    camOrient = pose[3:7]
    br.sendTransform(camPos,camOrient,rospy.Time.now(),"image","world")
    
    # Velocity publisher
    (vc,wc,vp,wp) = velocities(t)
    twistMsg = TwistStamped()
    twistMsg.header.stamp = rospy.Time.from_sec(t)
    twistMsg.header.frame_id = "image_local"
    twistMsg.twist.linear.x,twistMsg.twist.linear.y,twistMsg.twist.linear.z = tuple(vc.tolist())
    twistMsg.twist.angular.x,twistMsg.twist.angular.y,twistMsg.twist.angular.z = tuple(wc.tolist())
    velPub.publish(twistMsg)


def imagePubCB(event):
    
    camPos = pose[0:3]
    camOrient = pose[3:7]
    targetPos = pose[7:10]
    targetOrient = pose[10:]
    
    # Transform
    br.sendTransform(targetPos,targetOrient,rospy.Time.now(),"ugv0","world")
    
    # Inverse camera pose
    coiInv = qInv(camOrient)
    cpiInv = rotateVec(-camPos,coiInv)
    (rvec,jac) = cv2.Rodrigues(q2m(coiInv)[0:3,0:3])
    
    # Image Projections
    (imagePoint,jac) = cv2.projectPoints(np.array([[targetPos]]),rvec,cpiInv,camMat,distCoeffs)
    imagePoint = np.squeeze(imagePoint)
    (bearing,jac) = cv2.projectPoints(np.array([[targetPos]]),rvec,cpiInv,np.eye(3),np.zeros(5))
    bearing = np.squeeze(bearing)
    
    # Publish image point
    centMsg = Center()
    centMsg.header.stamp = rospy.Time.now()
    centMsg.header.frame_id = str(markerID)
    centMsg.x = imagePoint[0]
    centMsg.y = imagePoint[1]
    centMsg.x1 = bearing[0]
    centMsg.x2 = bearing[1]
    centerPub.publish(centMsg)


def camInfoPubCB(event):
    # publish camera parameters
    camInfoPub.publish(camInfoMsg)


def poseDyn(t,pose):
    camPos = pose[0:3]
    camOrient = pose[3:7]
    targetPos = pose[7:10]
    targetOrient = pose[10:]
    
    (vc,wc,vp,wp) = velocities(t) # velocities in camera coordinates
    vpg = rotateVec(vp,targetOrient)
    vcg = rotateVec(vc,camOrient)
    
    camPosDot = vcg
    camOrientDot = 0.5*qMult(np.append(wc,0),camOrient)
    targetPosDot = vpg
    targetOrientDot = 0.5*qMult(np.append(wp,0),targetOrient)
    
    poseDot = np.concatenate((camPosDot,camOrientDot,targetPosDot,targetOrientDot));
    
    return poseDot

def velocities(t):
    lenT = t.size
    
    # camera velocities/accelerations, expressed in camera coordinates
    vc = np.array([0, 0.5*np.sin(2*t), 0])
    wc = np.array([0, 0, 0])
    ac = np.array([0, np.cos(0.5*t), 0])
    
    # target velocities/accelerations, expressed in target coordinates
    vp = np.array([1*np.sin(0.5*t), 0, 0])
    wp = np.array([0, 0, 0])
    
    # target velocities/accelerations, expressed in target coordinates
    #vp = np.squeeze(np.concatenate((1*np.sin(0.5*t).reshape(-1), np.zeros(lenT), np.zeros(lenT))).reshape((-1,lenT)).T)
    #wp = np.squeeze(np.concatenate((np.zeros(lenT), np.zeros(lenT), 0.2*np.sin(0.3*t).reshape(-1))).reshape((-1,lenT)).T)
    #ap = np.squeeze(np.concatenate((1*0.5*np.cos(0.5*t).reshape(-1), np.zeros(lenT), np.zeros(lenT))).reshape((-1,lenT)).T)
    
    return (vc,wc,vp,wp)


def rotateVec(p,q):
    return qMult(q,qMult(np.append(p,0),qInv(q)))[0:3]


if __name__ == '__main__':
    sim()

