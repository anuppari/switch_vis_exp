#!/usr/bin/env python

import rospy
import tf
import collections
import numpy as np
import numpy.matlib as ml
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import CameraInfo #DEBUG

qMult = tf.transformations.quaternion_multiply # quaternion multiplication function handle
qInv = tf.transformations.quaternion_inverse # quaternion inverse function handle
q2m = tf.transformations.quaternion_matrix # quaternion to 4x4 transformation matrix

class DataHandler:
    def __init__(self, visibilityTimeout, tfl):
        self.tfl = tfl
        self.vCc = ml.zeros((3,1))
        self.wGCc = ml.zeros((3,1))
        self.vTt = ml.zeros((3,1)) #DEBUG
        self.wGTt = ml.zeros((3,1)) #DEBUG
        self.watchdogTimer = rospy.Timer(rospy.Duration.from_sec(5),self.timeout,oneshot=True) # large initial duration for slow startup
        self.visibilityTimeout = visibilityTimeout
        self.estimatorOn = False
        self.tBuff = collections.deque()
        self.etaBuff = collections.deque()
        self.sigmaBuff = collections.deque()
        self.fBuff = collections.deque()
    
    def timeout(self, event):
        self.estimatorOn = False
    
    def camVelCB(self, velData):
        self.vCc = np.matrix([[velData.twist.linear.x],[velData.twist.linear.y],[velData.twist.linear.z]])
        self.wGCc = np.matrix([[velData.twist.angular.x],[velData.twist.angular.y],[velData.twist.angular.z]])
    
    def targetVelCB(self, velData): #DEBUG
        self.vTt = np.matrix([[velData.twist.linear.x],[velData.twist.linear.y],[velData.twist.linear.z]])
        self.wGTt = np.matrix([[velData.twist.angular.x],[velData.twist.angular.y],[velData.twist.angular.z]])
    
    def poseCB(self, poseData):
        # Feature in FOV
        self.watchdogTimer.shutdown()
        
        # Data
        #self.poseT
        x = np.matrix([[poseData.pose.position.x],[poseData.pose.position.y],[poseData.pose.position.z]])
        q = np.matrix([[poseData.pose.orientation.x],[poseData.pose.orientation.y],[poseData.pose.orientation.z],[poseData.pose.orientation.w]])
        self.eta = np.append(x,q,axis=0)
        
        try:
            self.tfl.waitForTransform("image", "world", poseData.header.stamp, rospy.Duration(0.01))
            (self.xCam,self.qCam) = self.tfl.lookupTransform("image", "world", poseData.header.stamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as err:
            print err
            return
        
        # Update integration buffers
        self.tBuff.append(poseData.header.stamp.to_sec())
        self.etaBuff.append(self.eta.T)
        self.fBuff.append(f(self.eta,self.vCc,self.wGCc).T)
        self.sigmaBuff.append(sigma(self.eta,self.xCam,self.qCam).T)
        
        # Integrate
        npTbuff = np.array(self.tBuff)
        self.scriptEta = np.matrix(np.trapz(np.array(self.etaBuff),x=npTbuff,axis=0))
        self.scriptF = np.matrix(np.trapz(np.array(self.fBuff),x=npTbuff,axis=0))
        self.scriptY = np.matrix(np.trapz(np.array(self.sigmaBuff),x=npTbuff,axis=0))
        
        # Set timer to check if feature left FOV
        self.watchdogTimer = rospy.Timer(rospy.Duration.from_sec(self.visibilityTimeout),self.timeout,oneshot=True)
        self.estimatorOn = True

#DEBUG
def camInfoCB(dat):
    pass


def model_learning():
    
    rospy.init_node("model_learning")
    br = tf.TransformBroadcaster()
    tfl = tf.TransformListener()
    
    # Node parameters
    nodeName = rospy.get_name()
    k1 = 1
    k2 = 1
    kCL = 1
    visibilityTimout = rospy.get_param(nodeName+"/visibilityTimout",0.2)
    
    # Initialize object that handles all callbacks. Also initialize watchdog for feature visibility
    callbacks = DataHandler(visibilityTimout,tfl)
    
    # Subscribers
    camVelSub = rospy.Subscriber("image/body_vel", TwistStamped, callbacks.camVelCB, queue_size=1)
    targetVelSub = rospy.Subscriber("ugv0/body_vel", TwistStamped, callbacks.targetVelCB, queue_size=1)
    poseSub = rospy.Subscriber("relPose", PoseStamped, callbacks.poseCB, queue_size = 1)
    
    # DEBUG
    camSub = rospy.Subscriber("camera/camera_info",CameraInfo,camInfoCB)
    rospy.sleep(1)
    camSub.unregister()
    
    # Wait for initial data
    while not callbacks.estimatorOn:
        rospy.sleep(0.2)
    
    # Main loop
    lastTime = rospy.Time.now().to_sec()
    etaHat = ml.zeros((7,1))
    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        
        # Time
        timeNow = rospy.Time.now().to_sec()
        delT = timeNow - lastTime
        lastTime = timeNow
        
        # Setup
        eta = callbacks.eta
        x = eta[0:3,:]
        q = eta[3:,:]
        etaTilde = eta-etaHat
        vCc = callbacks.vCc
        wGCc = callbacks.wGCc
        vTt = callbacks.vTt #DEBUG
        wGTt = callbacks.wGTt #DEBUG
        phi = np.append(rotateVec(vTt,q),0.5*differentialMat(q)*wGTt,axis=0) #DEBUG
        
        # Estimation
        if callbacks.estimatorOn: # estimator
            #etaHatDot = np.dot(W.T(),sigma(eta,xCam,qCam)) + f(eta,vCc,wGCc) + k1*etaTilde + k2*np.sign(etaTilde).asmatrix()
            #Wdot = Gamma*sigma(eta,xCam,qCam)*etaTilde.T() + kCL*Gamma*
            etaHatDot = phi + f(eta,vCc,wGCc) + k1*etaTilde + k2*np.sign(etaTilde) #DEBUG
        else: # predictor
            #etaHatDot = np.dot(W.T(),sigma(etaHat,xCam,qCam)) + f(etaHat,vCc,wGCc)
            etaHatDot = phi + f(etaHat,vCc,wGCc) #DEBUG
        
        etaHat += etaHatDot*delT
        
        # Publish
        br.sendTransform(x,q,rospy.Time.now(),"ugv0estimate","image")
        
        r.sleep()


def sigma(eta,xCam,qCam):
    return np.zeros([5,5])


def f(eta,vCc,wGCc):
    x = eta[0:3,:]
    q = eta[3:,:]
    
    f1 = vCc + np.cross(wGCc,x,axis=0)
    f2 = 0.5*differentialMat(q)*rotateVec(wGCc,qInv(np.squeeze(q.A)))
    
    return -1*np.append(f1,f2,axis=0)


# qDot = 1/2*B*omega => omega = 2*B^T*qDot 
# See strapdown naviation book. If quaternion is orientation of frame 
# B w.r.t N in the sense that nP = q*bP*q', omega is ang. vel of frame B w.r.t. N, i.e. N_w_B,
# expressed in the B coordinate system
def differentialMat(q):
    b0 = q[3,0] 
    b1 = q[0,0] 
    b2 = q[1,0] 
    b3 = q[2,0] 
    B = np.matrix([[b0, -b3, b2],[b3, b0, -b1],[-b2, b1, b0],[-b1, -b2, -b3]])
     
    return B


def rotateVec(p,q):
    return np.asmatrix(qMult(np.squeeze(np.asarray(q)),qMult(np.append(np.squeeze(np.asarray(p)),0),qInv(np.squeeze(np.asarray(q)))))[0:3]).T


if __name__ == '__main__':
    model_learning()
