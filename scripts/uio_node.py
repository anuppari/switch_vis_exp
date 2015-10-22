#!/usr/bin/env python

import rospy
import tf
import numpy as np
import cv2
import collections
import threading
from aruco_ros.msg import Center
from geometry_msgs.msg import PointStamped, TwistStamped
from sensor_msgs.msg import CameraInfo
from switch_vis_exp.msg import Output
from std_msgs.msg import Empty

qMult = tf.transformations.quaternion_multiply # quaternion multiplication function handle
qInv = tf.transformations.quaternion_inverse # quaternion inverse function handle
q2m = tf.transformations.quaternion_matrix # quaternion to 4x4 transformation matrix

# Globals
markerID = 100
lock = threading.Lock()

def uio_node():
    global camMat, camSub, tfl, pub, pointPub, testPub
    global lastImageTime, z
    global A, E, L, M, N
    global velTimeBuff, vcBuff, wBuff
    
    # Init node
    rospy.init_node('uio_node')
    tfl = tf.TransformListener()
    
    # Gains
    A = np.array([[0,0,0],[0,0,1],[0,0,0]])
    C = np.array([[1,0,0],[0,1,0]])
    D = np.array([1,0,0])
    M = np.array([[0,0.71,0.68],[0,0.755,0.655],[0,0.276,0.163]])
    E = np.dot(np.dot((M-np.eye(3)),C.T),np.linalg.inv(np.dot(C,C.T)))
    K = np.array([[14.7, 5.20],[5.04, 15.55],[64.45, 71.06]])
    N = np.dot(M,A) - np.dot(K,C)
    L = np.dot(K,(np.eye(2)+np.dot(C,E))) - np.dot(np.dot(M,A),E)
    
    print "E: "+str(E)
    print "N: "+str(N)
    print "L: "+str(L)
    
    # Initialize parameters
    camMat = None
    lastImageTime = rospy.get_time()
    xhat0 = np.array([0,0,2.0])
    y0 = np.dot(C,xhat0)
    z = xhat0 + np.dot(E,y0)
    
    # Velocity buffers
    velTimeBuff = collections.deque(maxlen=70)
    vcBuff = collections.deque(maxlen=70)
    wBuff = collections.deque(maxlen=70)
    
    # Camera info
    cameraName = rospy.get_param("~camera","camera")
    camSub = rospy.Subscriber(cameraName+"/camera_info",CameraInfo,camInfoCB)
    
    # wait until camera info received
    while (camMat is None) and (not rospy.is_shutdown()):
        rospy.loginfo("Waiting for camera parameters...")
        rospy.sleep(1)
    
    # Output publishers
    pub = rospy.Publisher("output",Output,queue_size=10)
    pointPub = rospy.Publisher("outputPoint",PointStamped,queue_size=10)
    testPub = rospy.Publisher("observerVelTest",Empty,queue_size=1)
    
    # Velocity topic
    posSub = rospy.Subscriber("image/local_vel",TwistStamped,velCB)
    
    # Wait for velocity data
    while (len(velTimeBuff) == 0) and (not rospy.is_shutdown()):
        rospy.loginfo("Waiting for velocity data...")
        rospy.sleep(1)
    rospy.loginfo("Got Velocity data")
    
    rospy.sleep(10)
    
    # Marker center topic
    featureSub = rospy.Subscriber("markerCenters", Center, featureCB)
    
    # Wait until shutdown
    rospy.spin()


# Callback for getting image frame velocity
def velCB(velData):
    global velTimeBuff, vcBuff, wBuff
    
    # Time
    timeStamp = velData.header.stamp
    timeNow = float(timeStamp.secs) + float(timeStamp.nsecs)/float(1e9)
    # Camera velocity, expressed in local image coordinate system
    vc = np.array([velData.twist.linear.x,velData.twist.linear.y,velData.twist.linear.z])
    w = np.array([velData.twist.angular.x,velData.twist.angular.y,velData.twist.angular.z])
    velTimeBuff.append(timeNow)
    
    lock.acquire()
    try:
        # Add to buffer
        vcBuff.append(vc)
        wBuff.append(w)
        
        # Debug
        testPub.publish(Empty())
    finally:
        lock.release()


def featureCB(data):
    global lastImageTime, z
    
    # Disregard erroneous tag tracks
    if not (data.header.frame_id == str(markerID)):
        return
    
    # time
    timeStamp = data.header.stamp
    timeNow = float(timeStamp.secs) + float(timeStamp.nsecs)/float(1e9)
    delT = timeNow - lastImageTime
    lastImageTime = timeNow
    
    # Current image frame velocity, expressed in image coordinate system
    lock.acquire()
    try:
        vc = np.array([np.interp(timeNow,velTimeBuff,np.array(vcBuff)[:,0]),np.interp(timeNow,velTimeBuff,np.array(vcBuff)[:,1]),np.interp(timeNow,velTimeBuff,np.array(vcBuff)[:,2])])
        w = np.array([np.interp(timeNow,velTimeBuff,np.array(wBuff)[:,0]),np.interp(timeNow,velTimeBuff,np.array(wBuff)[:,1]),np.interp(timeNow,velTimeBuff,np.array(wBuff)[:,2])])
    finally:
        lock.release()
    
    # Ground truth
    tfl.waitForTransform("image","ugv0",timeStamp,rospy.Duration(0.1))
    (trans,quat) = tfl.lookupTransform("image","ugv0",timeStamp)
    x = np.array([trans[0]/trans[2],trans[1]/trans[2],1.0/trans[2]])
    
    # Undistort image coordinates. RETURNS NORMALIZED EUCLIDEAN COORDINATES
    undistPoints = np.squeeze(cv2.undistortPoints(np.array([[[data.x,data.y]]],dtype=np.float32),camMat,distCoeffs))
    y = np.array(undistPoints)
    
    ## Estimator
    # State
    xhat = z - np.dot(E,y)
    XYZhat = np.array([xhat[0]/xhat[2], xhat[1]/xhat[2], 1.0/xhat[2]])
    
    # Signals
    fbar = fFunc(xhat,vc,w) - np.dot(A,xhat)
    g = gFunc(xhat,w)
    
    # Update
    print "fbar: "+str(fbar)
    print "g: "+str(g)
    print "np.dot(N,z): "+str(np.dot(N,z))
    print "np.dot(L,y): "+str(np.dot(L,y))
    print "np.dot(M,fbar): "+str(np.dot(M,fbar))
    print "np.dot(M,g): "+str(np.dot(M,g))
    print "xhat: "+str(xhat)
    
    zDot = np.dot(N,z) + np.dot(L,y) + np.dot(M,fbar) + np.dot(M,g)
    
    # Integrate
    z = z + zDot*delT
    
    # Publish output
    outMsg = Output()
    outMsg.header.stamp = timeStamp
    outMsg.y = x.tolist()
    outMsg.yhat = xhat.tolist()
    outMsg.error = (x-xhat).tolist()
    outMsg.XYZ = trans
    outMsg.XYZhat = XYZhat.tolist()
    outMsg.XYZerror = [trans[0]-XYZhat[0],trans[1]-XYZhat[1],trans[2]-XYZhat[2]]
    outMsg.estimatorOn = True
    #outMsg.usePredictor = usePredictor
    pub.publish(outMsg)
    
    # Publish point
    pntMsg = PointStamped()
    pntMsg.header.stamp = timeStamp
    pntMsg.header.frame_id = 'image'
    pntMsg.point.x = xhat[0]/xhat[2]
    pntMsg.point.y = xhat[1]/xhat[2]
    pntMsg.point.z = 1.0/xhat[2]
    pointPub.publish(pntMsg)


def fFunc(x,vc,w):
    x1,x2,x3 = tuple(x.tolist())
    vcx,vcy,vcz = tuple(vc.tolist())
    w1,w2,w3 = tuple(w.tolist())
    
    f1 = (vcx - x1*vcz)*x3
    f2 = (vcy - x2*vcz)*x3
    f3 = -vcz*np.power(x3,2) - (w1*x2 - w2*x1)*x3
    
    return np.array([f1,f2,f3])


def gFunc(x,w):
    x1,x2,x3 = tuple(x.tolist())
    w1,w2,w3 = tuple(w.tolist())
    
    Omega1 = -x1*x2*w1 + w2 + np.power(x1,2)*w2 - x2*w3
    Omega2 = -w1 - np.power(x2,2)*w1 + x1*x2*w2 + x1*w3
    
    return np.array([Omega1,Omega2,0])


# Callback for getting camera intrinsic parameters and then unsubscribing
def camInfoCB(camInfo):
    global distCoeffs, camMat
    
    distCoeffs = np.array(camInfo.D,dtype=np.float32)
    camMat = np.array(camInfo.K,dtype=np.float32).reshape((3,3))
    
    print distCoeffs
    print camMat
    rospy.loginfo("Got camera parameters!")
    camSub.unregister()


def rotateVec(p,q):
    return qMult(q,qMult(np.append(p,0),qInv(q)))[0:3]


if __name__ == '__main__':
    uio_node()

