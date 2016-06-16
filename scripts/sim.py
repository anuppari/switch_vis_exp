#!/usr/bin/env python

import rospy
import tf
import threading
import numpy as np
import itertools
import cv2
from aruco_ros.msg import Center
from geometry_msgs.msg import TwistStamped, PoseStamped, Pose
from sensor_msgs.msg import CameraInfo, Joy
from switch_vis_exp.srv import MapVel

qMult = tf.transformations.quaternion_multiply # quaternion multiplication function handle
qInv = tf.transformations.quaternion_inverse # quaternion inverse function handle
q2m = tf.transformations.quaternion_matrix # quaternion to 4x4 transformation matrix

markerID = 100;
camMat = np.array([[558.953280,0.000000,365.566775],[0.000000,557.877582,240.157184],[0,0,1]])
#distCoeffs = np.array([-0.385156,0.163233,0.000539,0.000302,0.000000])
#camMat = np.eye(3)
distCoeffs = np.zeros(5)
intRate = 200 # [hz] integration rate
velRate = 200 # [hz] velocity data publish rate
frameRate = 60 # [hz] marker publish rate
onDuration = np.array([3,5])
offDuration = np.array([1,4])
lock = threading.Lock()

def sim():
    global t, pose, camInfoMsg
    global centerPub, targetVelPub, camVelPub, camInfoPub, br, tfl, posePub, getMapVel
    global switchTimer, imageTimer, estimatorOn
    
    rospy.init_node("sim")
    estimatorOn = True
    
    centerPub = rospy.Publisher("markerCenters",Center,queue_size=10)
    posePub = rospy.Publisher("relPose",PoseStamped,queue_size=10)
    targetVelPub = rospy.Publisher("ugv0/body_vel",TwistStamped,queue_size=10)
    camVelPub = rospy.Publisher("image/body_vel",TwistStamped,queue_size=10)
    cameraName = rospy.get_param(rospy.get_name()+"/camera","camera")
    #camInfoPub = rospy.Publisher(cameraName+"/camera_info",CameraInfo,queue_size=1)
    joySub = rospy.Subscriber("joy", Joy, joyCB, queue_size=1)
    br = tf.TransformBroadcaster()
    tfl = tf.TransformListener()
    
    # Camera parameters
    camInfoMsg = CameraInfo()
    camInfoMsg.D = distCoeffs.tolist()
    camInfoMsg.K = camMat.reshape((-1)).tolist()
    
    ## Wait for node to get cam info
    #while (camVelPub.get_num_connections() == 0) and (not rospy.is_shutdown()):
        ## publish camera parameters
        #camInfoPub.publish(camInfoMsg)
        #rospy.sleep(0.5)
    
    # Target velocity service handle
    rospy.wait_for_service('get_velocity')
    getMapVel = rospy.ServiceProxy('get_velocity', MapVel)
    
    # Publishers
    rospy.Timer(rospy.Duration(1.0/velRate),velPubCB)
    imageTimer = rospy.Timer(rospy.Duration(1.0/frameRate),imagePubCB)
    #rospy.Timer(rospy.Duration(0.5),camInfoPubCB)
    switchTimer = rospy.Timer(rospy.Duration(60.0),switchCB,oneshot=True)
    
    # Initial conditions
    startTime = rospy.get_time()
    camPos = np.array([0,-1,1.5])
    camOrient = np.array([-1*np.sqrt(2)/2,0,0,np.sqrt(2)/2])
    #camPos = np.array([0,0,0])
    #camOrient = np.array([0,0,0,1])
    targetPos = np.array([1,.1,0])
    targetOrient = np.array([0,0,0,1])
    pose = np.concatenate((camPos,camOrient,targetPos,targetOrient))
    
    r = rospy.Rate(intRate)
    h = 1.0/intRate
    lastTime = rospy.get_time()
    while not rospy.is_shutdown():
        # Time
        timeNow = rospy.get_time()
        delT = timeNow - lastTime
        lastTime = timeNow
        t = np.array(timeNow - startTime)
        
        poseDot = poseDyn(t,pose)
        pose = pose + poseDot*delT
        
        # Send Transform
        camPos = pose[0:3]
        camOrient = pose[3:7]
        targetPos = pose[7:10]
        targetOrient = pose[10:]
        br.sendTransform(targetPos,targetOrient,rospy.Time.now(),"ugv0","world")
        br.sendTransform(camPos,camOrient,rospy.Time.now(),"image","world")
        
        r.sleep()


def switchCB(event):
    global estimatorOn, switchTimer, imageTimer
    
    if estimatorOn:
        imageTimer.shutdown()
        switchTimer = rospy.Timer(rospy.Duration((offDuration[1]-offDuration[0])*np.random.rand(1)+offDuration[0]),switchCB,oneshot=True)
        estimatorOn = False
    else:
        imageTimer = rospy.Timer(rospy.Duration(1.0/frameRate),imagePubCB)
        switchTimer = rospy.Timer(rospy.Duration((onDuration[1]-onDuration[0])*np.random.rand(1)+onDuration[0]),switchCB,oneshot=True)
        estimatorOn = True


def velPubCB(event):
    
    # Velocity publisher
    (vc,wc,vp,wp) = velocities(t)
    velMsg = TwistStamped()
    velMsg.header.stamp = rospy.Time.now()
    velMsg.twist.linear.x = vp[0]
    velMsg.twist.linear.y = vp[1]
    velMsg.twist.linear.z = vp[2]
    velMsg.twist.angular.x = wp[0]
    velMsg.twist.angular.y = wp[1]
    velMsg.twist.angular.z = wp[2]
    targetVelPub.publish(velMsg)
    
    velMsg = TwistStamped()
    velMsg.header.stamp = rospy.Time.now()
    velMsg.twist.linear.x = vc[0]
    velMsg.twist.linear.y = vc[1]
    velMsg.twist.linear.z = vc[2]
    velMsg.twist.angular.x = wc[0]
    velMsg.twist.angular.y = wc[1]
    velMsg.twist.angular.z = wc[2]
    camVelPub.publish(velMsg)


def imagePubCB(event):
    camPos = pose[0:3]
    camOrient = pose[3:7]
    targetPos = pose[7:10]
    targetOrient = pose[10:]
    
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
    centerPub.publish(centMsg)
    
    # Publish pose
    relPos = rotateVec(targetPos - camPos,qInv(camOrient))
    relOrient = qMult(qInv(camOrient),targetOrient)
    poseMsg = PoseStamped()
    poseMsg.header.stamp = centMsg.header.stamp
    poseMsg.header.frame_id = str(markerID)
    poseMsg.pose.position.x = relPos[0]
    poseMsg.pose.position.y = relPos[1]
    poseMsg.pose.position.z = relPos[2]
    poseMsg.pose.orientation.x = relOrient[0]
    poseMsg.pose.orientation.y = relOrient[1]
    poseMsg.pose.orientation.z = relOrient[2]
    poseMsg.pose.orientation.w = relOrient[3]
    posePub.publish(poseMsg)


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
    camOrientDot = 0.5*np.dot(differentialMat(camOrient),wc)
    targetPosDot = vpg
    targetOrientDot = 0.5*np.dot(differentialMat(targetOrient),wp)
    
    poseDot = np.concatenate((camPosDot,camOrientDot,targetPosDot,targetOrientDot));
    
    return poseDot


def joyCB(joyData):
    global latestJoyMsg
    latestJoyMsg = joyData


def velocities(t):
    global pose
    
    lenT = t.size
    
    # Target velocity from map
    camPos = pose[0:3]
    camOrient = pose[3:7]
    targetPos = pose[7:10]
    targetOrient = pose[10:]
    poseMsg = Pose()
    poseMsg.position.x = targetPos[0]
    poseMsg.position.y = targetPos[1]
    poseMsg.position.z = targetPos[2]
    poseMsg.orientation.x = targetOrient[0]
    poseMsg.orientation.y = targetOrient[1]
    poseMsg.orientation.z = targetOrient[2]
    poseMsg.orientation.w = targetOrient[3]
    lock.acquire()
    try:
        resp = getMapVel([poseMsg])
    finally:
        lock.release()
    twistMsg = resp.twist[0]
    vp = rotateVec(np.array([twistMsg.linear.x,twistMsg.linear.y,twistMsg.linear.z]),qInv(targetOrient))
    vp[1:] = 0
    wp = rotateVec(np.array([twistMsg.angular.x,twistMsg.angular.y,twistMsg.angular.z]),qInv(targetOrient))
    vc = np.array([0,0,0])
    wc = np.array([0,0,0])
    
    #vc = np.array([0,0,0])
    #wc = np.array([0,0,0])
    #vp = np.array([joyData.axes[1],0,0])
    #wp = np.array([0,0,joyData.axes[0]])
    
    ## camera velocities, expressed in camera coordinates
    #vc = 0.3*np.array([np.sin(3*t),np.cos(4*t),0])
    #wc = 0.4*np.array([0.5*np.cos(2*t),0.5*np.sin(t),0.0*np.cos(3*t)])
    
    ## target velocities, expressed in target coordinates
    #vp = np.array([0.5,0,0])
    #wp = np.array([0,0,0.5])
    
    # target velocities, expressed in target coordinates
    #vp = np.squeeze(np.concatenate((1*np.sin(0.5*t).reshape(-1), np.zeros(lenT), np.zeros(lenT))).reshape((-1,lenT)).T)
    #wp = np.squeeze(np.concatenate((np.zeros(lenT), np.zeros(lenT), 0.2*np.sin(0.3*t).reshape(-1))).reshape((-1,lenT)).T)

    return (vc,wc,vp,wp)


def rotateVec(p,q):
    return qMult(q,qMult(np.append(p,0),qInv(q)))[0:3]


# qDot = 1/2*B*omega => omega = 2*B^T*qDot 
# See strapdown naviation book. If quaternion is orientation of frame 
# B w.r.t N in the sense that nP = q*bP*q', omega is ang. vel of frame B w.r.t. N, i.e. N_w_B,
# expressed in the B coordinate system
def differentialMat(q): 
    b0 = q[3] 
    b1 = q[0] 
    b2 = q[1] 
    b3 = q[2] 
    B = np.array([[b0, -b3, b2],[b3, b0, -b1],[-b2, b1, b0],[-b1, -b2, -b3]])
     
    return B


if __name__ == '__main__':
    sim()

