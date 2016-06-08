#!/usr/bin/env python

import rospy
import tf
import collections
import numpy as np
import numpy.matlib as ml
from geometry_msgs.msg import PoseStamped, TwistStamped, Pose, Position, Quaternion
from sensor_msgs.msg import CameraInfo #DEBUG

qMult = tf.transformations.quaternion_multiply # quaternion multiplication function handle
qInv = tf.transformations.quaternion_inverse # quaternion inverse function handle
q2m = tf.transformations.quaternion_matrix # quaternion to 4x4 transformation matrix

class DataHandler:
    def __init__(self, intWindow, visibilityTimeout, tfl):
        self.tfl = tfl
        self.intWindow = intWindow
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
        while tBuff[-1]-tBuff[0] > self.intWindow: # more accurate then setting maxlen
            tBuff.popleft()
            etaBuff.popleft()
            fBuff.popleft()
            sigmaBuff.popleft()
        
        # Integrate
        npTbuff = np.array(self.tBuff-self.tBuff[0])
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
    intWindow = 1
    visibilityTimout = rospy.get_param(nodeName+"/visibilityTimout",0.2)
    
    # Initialize object that handles all callbacks. Also initialize watchdog for feature visibility
    callbacks = DataHandler(intWindow, visibilityTimout, tfl)
    
    # Subscribers
    camVelSub = rospy.Subscriber("image/body_vel", TwistStamped, callbacks.camVelCB, queue_size=1)
    targetVelSub = rospy.Subscriber("ugv0/body_vel", TwistStamped, callbacks.targetVelCB, queue_size=1)
    poseSub = rospy.Subscriber("relPose", PoseStamped, callbacks.poseCB, queue_size = 1)
    
    # DEBUG
    camSub = rospy.Subscriber("camera/camera_info",CameraInfo,camInfoCB)
    rospy.sleep(1)
    camSub.unregister()
    
    # Setup Neural Network
    a = 1
    b = 1
    mapWidth = 1.2*a
    mapHeight = 1.2*b
    x0 = 0
    y0 = 0
    numKernalWidth = 4
    numKernalHeight = 4
    cov = 0.3*ml.eye(2)
    muX = np.linspace(x0-mapWidth,x0+mapWidth,numKernalWidth)
    muY = np.linspace(y0-mapHeight,y0+mapHeight,numKernalHeight)
    (muX,muY) = np.meshgrid(muX,muY)
    mu = np.asmatrix(np.vstack((muX.flatten(),muY.flatten())).T) # Nx2
    sigma = lambda eta, xCam, qCam : sigmaGen(eta,xCam,qCam,mu,cov)
    
    # Generate pre-seed data
    rospy.wait_for_service('get_velocity')
    getMapVel = rospy.ServiceProxy('get_velocity', MapVel)
    numPts = 200
    eta = np.vstack((2*mapWidth*(np.random.random(numPts)-0.5),2*mapHeight*(np.random.random(numPts)-0.5)+y0,np.zeros((5,numPts))))
    poseMsgs = [Pose(position=Point(x=etai[0,0],y=etai[0,1],z=etai[0,2]),orientation=Quaternion(x=etai[0,3],y=etai[0,4],z=etai[0,5],w=etai[0,6])) for etai in eta.T]
    resp = getMapVel(poseMsgs)
    bMat = np.array([np.append(np.matrix([twistMsg.linear.x,twistMsg.linear.y,twistMsg.linear.z]).T,0.5*differentialMat(etai[0,3:7].T)*np.matrix([twistMsg.angular.x,twistMsg.angular.y,twistMsg.angular.z]).T) for twistMsg, etai in zip(resp.twist,eta.T)]).transpose(2,1,0).flatten()
    twistMsg = resp.twist[0]
    vpg = rotateVec(np.array([twistMsg.linear.x,twistMsg.linear.y,twistMsg.linear.z]),targetOrient)
    
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


def sigmaGen(eta,xCam,qCam,mu,cov):
    # mu: Nxdim. dim = 2 or 3
    # eta: 7xM
    # out: NxM
    
    dim = 2
    
    # Convert to global coordinates and setup query points
    pts = np.zeros([eta.shape[1],1,dim) # 1 x dim x M
    for ind,etai in enumerate(eta.T):
        x = etai[:,0:3].T
        q = etai[:,3:].T
        
        xg = rotateVec(x,qCam) + xCam
        qg = qCam*q
        yaw = tf.transformations.euler_from_quaternion(qg)[2]
        
        pts[ind,0,:] = xg[0:2] # add yaw here if dim=3
    
    dev = (mu.A-pts).transpose(0,2,1) 
    return np.exp(-0.5*np.sum(dev*np.dot(np.linalg.inv(cov).A,dev).transpose(1,0,2),axis=1).T)


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
