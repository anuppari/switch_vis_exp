#!/usr/bin/env python

import rospy
import tf
import collections
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import numpy.matlib as ml
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import PoseStamped, TwistStamped, Pose, Point, Quaternion
from sensor_msgs.msg import CameraInfo #DEBUG
from switch_vis_exp.srv import MapVel

qMult = tf.transformations.quaternion_multiply # quaternion multiplication function handle
qInv = tf.transformations.quaternion_inverse # quaternion inverse function handle
q2m = tf.transformations.quaternion_matrix # quaternion to 4x4 transformation matrix

class DataHandler:
    def __init__(self, intWindow, visibilityTimeout, tfl, sigma):
        self.tfl = tfl # MAYBE JUST INITIALIZE INSTEAD OF PASSING IN
        self.sigma = sigma
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
        #print "estimatorOff"
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
        #if not self.estimatorOn: print "estimatorOn"
        
        # Data
        #self.poseT
        x = np.matrix([[poseData.pose.position.x],[poseData.pose.position.y],[poseData.pose.position.z]])
        q = np.matrix([[poseData.pose.orientation.x],[poseData.pose.orientation.y],[poseData.pose.orientation.z],[poseData.pose.orientation.w]])
        self.eta = np.append(x,q,axis=0)
        
        ## XCAM, QCAM WON'T GET UPDATED WHEN MEASUREMENTS ARN'T AVAILABLE!!!!!!!!!
        try:
            self.tfl.waitForTransform("world", "image", poseData.header.stamp, rospy.Duration(0.01))
            (trans,rot) = self.tfl.lookupTransform("world", "image", poseData.header.stamp)
            self.xCam = np.matrix(trans).T
            self.qCam = np.matrix(rot).T
        except:
            return
        
        ## Update integration buffers
        #self.tBuff.append(poseData.header.stamp.to_sec())
        #self.etaBuff.append(self.eta.T)
        #self.fBuff.append(f(self.eta,self.vCc,self.wGCc).T)
        #self.sigmaBuff.append(self.sigma(self.eta,self.xCam,self.qCam).T)
        #while self.tBuff[-1]-self.tBuff[0] > self.intWindow: # more accurate then setting maxlen
            #self.tBuff.popleft()
            #self.etaBuff.popleft()
            #self.fBuff.popleft()
            #self.sigmaBuff.popleft()
        
        ## Integrate
        #npTbuff = np.array([ti-self.tBuff[0] for ti in self.tBuff])
        #self.scriptEta = np.matrix(np.trapz(np.array(self.etaBuff),x=npTbuff,axis=0))
        #self.scriptF = np.matrix(np.trapz(np.array(self.fBuff),x=npTbuff,axis=0))
        #self.scriptY = np.matrix(np.trapz(np.array(self.sigmaBuff),x=npTbuff,axis=0))
        
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
    np.set_printoptions(precision=5,suppress=True)
    
    # Node parameters
    nodeName = rospy.get_name()
    k1 = 2
    k2 = 0.01
    kCL = 1
    intWindow = 1
    visibilityTimout = rospy.get_param(nodeName+"/visibilityTimout",0.2)
    
    # Setup Neural Network
    a = rospy.get_param(nodeName+"/a",1.0)
    b = rospy.get_param(nodeName+"/b",1.0)
    x0 = rospy.get_param(nodeName+"/x0",0.0)
    y0 = rospy.get_param(nodeName+"/y0",0.0)
    mapWidth = 2*a
    mapHeight = 2*b
    numKernalWidth = 11
    numKernalHeight = 11
    cov = 0.3*ml.eye(2)
    muX = np.linspace(x0-mapWidth,x0+mapWidth,numKernalWidth)
    muY = np.linspace(y0-mapHeight,y0+mapHeight,numKernalHeight)
    (muX,muY) = np.meshgrid(muX,muY)
    mu = np.asmatrix(np.vstack((muX.flatten(),muY.flatten())).T) # Nx2
    numKernal = mu.shape[0]
    sigma = lambda eta, xCam, qCam : sigmaGen(eta,xCam,qCam,mu,cov)
    
    # Initialize object that handles all callbacks. Also initialize watchdog for feature visibility
    callbacks = DataHandler(intWindow, visibilityTimout, tfl, sigma) # DONT PASS IN TFL?
    
    # Subscribers
    camVelSub = rospy.Subscriber("image/body_vel", TwistStamped, callbacks.camVelCB, queue_size=1)
    targetVelSub = rospy.Subscriber("ugv0/body_vel", TwistStamped, callbacks.targetVelCB, queue_size=1)
    poseSub = rospy.Subscriber("relPose", PoseStamped, callbacks.poseCB, queue_size = 1)
    
    # DEBUG
    camSub = rospy.Subscriber("camera/camera_info",CameraInfo,camInfoCB)
    rospy.sleep(1)
    camSub.unregister()
    
    # Generate pre-seed data
    print "here1"
    rospy.wait_for_service('get_velocity')
    getMapVel = rospy.ServiceProxy('get_velocity', MapVel)
    print "here2"
    numPts = 2000
    eta = np.asmatrix(np.vstack((2*mapWidth*(np.random.random(numPts)-0.5)+x0,2*mapHeight*(np.random.random(numPts)-0.5)+y0,np.zeros((5,numPts))))) # 7xM
    print "here3"
    Y = sigma(eta,ml.zeros((3,1)),np.matrix([0,0,0,1]).T) # 7MxN
    print "here4"
    #A = scipy.linalg.block_diag(*[sig.T for i in range(7)])
    A = Y
    #A = scipy.sparse.block_diag([sig.T for i in range(7)])
    print "here5"
    poseMsgs = [Pose(position=Point(x=etai[0,0],y=etai[0,1],z=etai[0,2]),orientation=Quaternion(x=etai[0,3],y=etai[0,4],z=etai[0,5],w=etai[0,6])) for etai in eta.T]
    print "here6"
    resp = getMapVel(poseMsgs)
    print "here7"
    #bMat = np.ascontiguousarray(np.array([np.append(np.matrix([twistMsg.linear.x,twistMsg.linear.y,twistMsg.linear.z]).T,0.5*differentialMat(etai[0,3:7].T)*np.matrix([twistMsg.angular.x,twistMsg.angular.y,twistMsg.angular.z]).T,axis=0) for twistMsg, etai in zip(resp.twist,eta.T)]).transpose(2,1,0).flatten())
    bMat = np.ascontiguousarray(np.array([np.append(np.matrix([twistMsg.linear.x,twistMsg.linear.y,twistMsg.linear.z]).T,0.5*differentialMat(etai[0,3:7].T)*np.matrix([twistMsg.angular.x,twistMsg.angular.y,twistMsg.angular.z]).T,axis=0) for twistMsg, etai in zip(resp.twist,eta.T)]).flatten())
    print "here8"
    #WvecIdeal = np.asmatrix(np.linalg.lstsq(A,bMat)[0])
    thetaIdeal = np.asmatrix(np.linalg.lstsq(A,bMat)[0]).T
    #WvecIdeal = np.asmatrix(scipy.sparse.linalg.lsqr(A,bMat)[0])
    print "here9"
    #Wideal = np.reshape(WvecIdeal,(7,numKernal)).T
    
    
    print "here"
    
    ## Check least squares fit
    #plotWidth = 200
    #plotHeight = 200
    #plotX = np.linspace(x0-mapWidth,x0+mapWidth,plotWidth)
    #plotY = np.linspace(y0-mapHeight,y0+mapHeight,plotHeight)
    #(plotX,plotY) = np.meshgrid(plotX,plotY)
    #print "here1"
    #plotEta = np.asmatrix(np.vstack((plotX.flatten(),plotY.flatten(),np.zeros((5,plotWidth*plotHeight))))) # FIX THIS BACK TO ZEROS
    ##plotEta = np.asmatrix(np.vstack((plotX.flatten(),plotY.flatten(),np.zeros((1,plotWidth*plotHeight)),np.random.random([4,plotWidth*plotHeight]))))
    ##sig = sigma(plotEta,ml.zeros((3,1)),np.matrix([0,0,0,1]).T) # NxM
    #Y = sigma(plotEta,ml.zeros((3,1)),np.matrix([0,0,0,1]).T) # 7MxN
    ##sig = sigma(plotEta,np.asmatrix(np.random.random((3,1))),np.asmatrix(np.random.random((4,1)))) # NxM
    #print "here2"
    ##A = np.linalg.block_diag([sig.T for i in range(7)])
    #poseMsgs = [Pose(position=Point(x=etai[0,0],y=etai[0,1],z=etai[0,2]),orientation=Quaternion(x=etai[0,3],y=etai[0,4],z=etai[0,5],w=etai[0,6])) for etai in plotEta.T]
    #print "here3"
    #resp = getMapVel(poseMsgs)
    #bMat = np.asmatrix(np.array([np.append(np.matrix([twistMsg.linear.x,twistMsg.linear.y,twistMsg.linear.z]).T,0.5*differentialMat(etai[0,3:7].T)*np.matrix([twistMsg.angular.x,twistMsg.angular.y,twistMsg.angular.z]).T,axis=0) for twistMsg, etai in zip(resp.twist,plotEta.T)]).transpose(2,1,0))
    #print "here4"
    ##bHat = Wideal.T*sig
    #bHat = (Y*thetaIdeal).reshape(-1,7).T
    #error = bMat - bHat
    #print "here5"
    
    ## Plot
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(plotX,plotY,np.reshape(np.minimum(np.sum(np.square(error),axis=0),10),(plotWidth,plotHeight)),rstride=4,cstride=4,cmap=cm.coolwarm)
    #ax.scatter(eta[0,:].A.flatten(),eta[1,:].A.flatten(),np.zeros(eta[0,:].A.size),c='r',marker='o')
    #ax.set_zlim(-0.1,1.01)
    #plt.show()
    
    #return
    
    # Wait for initial data
    while not callbacks.estimatorOn:
        rospy.sleep(0.2)
    
    # Main loop
    lastTime = rospy.Time.now().to_sec()
    etaHat = ml.zeros((7,1))
    r = rospy.Rate(20)
    while not rospy.is_shutdown():
        
        # Time
        timeNow = rospy.Time.now().to_sec()
        delT = timeNow - lastTime
        lastTime = timeNow
        
        # Get latest data
        eta = callbacks.eta
        x = eta[0:3,:]
        q = eta[3:,:]
        etaTilde = eta-etaHat
        vCc = callbacks.vCc
        wGCc = callbacks.wGCc
        xCam = callbacks.xCam
        qCam = callbacks.qCam
        vTt = callbacks.vTt #DEBUG
        wGTt = callbacks.wGTt #DEBUG
        phi = np.append(rotateVec(vTt,q),0.5*differentialMat(q)*wGTt,axis=0) #DEBUG
        
        # Estimation
        if callbacks.estimatorOn: # estimator
            print "W: " + str((sigma(eta,xCam,qCam)*thetaIdeal).T[0,0:3])
            print "phi: " + str(phi.T[0,0:3])
            
            #etaHatDot = Wideal.T*sigma(eta,xCam,qCam) + f(eta,vCc,wGCc) + k1*etaTilde + k2*np.sign(etaTilde)
            etaHatDot = sigma(eta,xCam,qCam)*thetaIdeal + f(eta,vCc,wGCc) + k1*etaTilde + k2*np.sign(etaTilde)
            #Wdot = Gamma*sigma(eta,xCam,qCam)*etaTilde.T() + kCL*Gamma*
            #etaHatDot = phi + f(eta,vCc,wGCc) + k1*etaTilde + k2*np.sign(etaTilde) #DEBUG
        else: # predictor
            print "W: " + str((sigma(etaHat,xCam,qCam)*thetaIdeal).T[0,0:3])
            print "phi: " + str(phi.T[0,0:3])
            
            #print Wideal.T*sigma(etaHat,xCam,qCam), phi
            #etaHatDot = Wideal.T*sigma(etaHat,xCam,qCam) + f(etaHat,vCc,wGCc)
            etaHatDot = sigma(etaHat,xCam,qCam)*thetaIdeal + f(etaHat,vCc,wGCc)
            #etaHatDot = phi + f(etaHat,vCc,wGCc) #DEBUG
        
        etaHat += etaHatDot*delT
        
        #print "error: " + str((Wideal.T*sigma(eta,xCam,qCam) - phi).T)
        
        # Publish
        xHat = etaHat[0:3,:]
        qHat = etaHat[3:,:]
        br.sendTransform(xHat,qHat,rospy.Time.now(),"ugv0estimate","image")
        
        r.sleep()


def sigmaGen(eta,xCam,qCam,mu,cov):
    # ALL MUST BE MATRIX, NOT ARRAY
    # mu: Nxdim. dim = 2 or 3
    # eta: 7xM
    # out: NxM
    
    dim = 2
    M = eta.shape[1]
    N = mu.shape[0]
    
    x = eta[0:3,:]
    q = eta[3:,:]
    #print "qCam: " + str(qCam.T)
    #print "q: " + str(q.T)
    qg = qMultArray(qCam,q)
    #print "qCam: " + str(qCam.T)
    xg = rotateVecsArray(x,qCam) + xCam
    #print "qCam: " + str(qCam.T)
    #xgTrue = np.zeros(xg.shape)
    #qgTrue = np.zeros(qg.shape)
    pts = xg[0:2,:].T.A.reshape(-1,1,dim) # numpy Mx1xdim. add yaw here if dim=3
    #print "pts: " + str(pts.shape)
    #print pts
    
    #print "xg: " + str(xg.T)
    #print "qg: " + str(qg.T)
    
    ## Convert to global coordinates and setup query points
    #pts = np.zeros([eta.shape[1],1,dim]) # 1 x dim x M
    #for ind,etai in enumerate(eta.T):
        #xi = etai[:,0:3].T
        #qi = etai[:,3:].T
        
        #print "qCam: " + str(qCam.T)
        #xgi = rotateVec(xi,qCam) + xCam
        #print "qCam: " + str(qCam.T)
        #qgi = qMult(qCam.A.flatten(),qi.A.flatten())
        #yaw = tf.transformations.euler_from_quaternion(qgi)[2]
        
        #print "qCam: " + str(qCam.A.flatten())
        #print "qi: " + str(qi.A.flatten())
        
        #xgTrue[:,ind] = xgi.A.flatten()
        #qgTrue[:,ind] = qgi.flatten()
        
        #pts[ind,0,:] = xgi[0:2,:].T.A # add yaw here if dim=3
    
    #print np.allclose(xg,xgTrue)
    #print np.allclose(qg,qgTrue)
    
    #print "xgTrue: " + str(xgTrue.T)
    #print "qgTrue: " + str(qgTrue.T) 
    
    dev = (mu.A-pts).transpose(0,2,1) # numpy M x dim x N
    #return np.asmatrix(np.exp(-0.5*np.sum(dev*np.dot(np.linalg.inv(cov).A,dev).transpose(1,0,2),axis=1).T)) 
    #                                                                             ^ extra transpose before sum is necessary due to how numpy broadcasts for np.dot
    
    
    sigma = np.exp(-0.5*np.sum(dev*np.dot(np.linalg.inv(cov).A,dev).transpose(1,0,2),axis=1)) # numpy M x N
    #                                                                 ^ extra transpose before sum is necessary due to how numpy broadcasts for np.dot
    
    Q = rotMat(np.asmatrix(qInvArray(qCam)))
    Bq = differentialMat(q.A)
    Bqt = differentialMat(qg.A)
    Y = np.zeros((M,7,7*N))
    for i in range(M):
        T = scipy.linalg.block_diag(Q,np.dot(Bq[i,:,:],Bqt[i,:,:].T))
        Y[i,:,:] = np.kron(T,sigma[i,:])
    
    return np.asmatrix(Y.reshape(7*M,7*N))


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
    # q: 4xN
    # B: numpy Nx4x3
    N = q.shape[1]
    b0 = q[3,:].reshape(N,1,1)
    b1 = q[0,:].reshape(N,1,1)
    b2 = q[1,:].reshape(N,1,1)
    b3 = q[2,:].reshape(N,1,1)
    #B = np.matrix([[b0, -b3, b2],[b3, b0, -b1],[-b2, b1, b0],[-b1, -b2, -b3]])
    B = np.concatenate((np.concatenate((b0, -b3, b2),axis=-1),np.concatenate((b3, b0, -b1),axis=-1),np.concatenate((-b2, b1, b0),axis=-1),np.concatenate((-b1, -b2, -b3),axis=-1)),axis=-2)
     
    return B


def rotateVec(p,q):
    return np.asmatrix(qMult(np.squeeze(np.asarray(q)),qMult(np.append(np.squeeze(np.asarray(p)),0),qInv(np.squeeze(np.asarray(q)))))[0:3]).T


def rotateVecsArray(p,q):
    # p: 3xN or 3x1
    # q: 4xN or 4x1
    # (any combination)
    # out: 3xN
    return qMultArray(q,qMultArray(np.vstack((p,np.zeros(p.shape[1]))),qInvArray(q)))[0:3,:]


def qMultArray(q1, q0):
    # q0: 4xN or 4x1
    # q1: 4xN or 4x1
    # (any combination)
    # out: 4xN
    x0 = q0[0,:]
    y0 = q0[1,:]
    z0 = q0[2,:]
    w0 = q0[3,:]
    x1 = q1[0,:]
    y1 = q1[1,:]
    z1 = q1[2,:]
    w1 = q1[3,:]
    
    return np.vstack((
         np.multiply(x1,w0) + np.multiply(y1,z0) - np.multiply(z1,y0) + np.multiply(w1,x0),
        np.multiply(-x1,z0) + np.multiply(y1,w0) + np.multiply(z1,x0) + np.multiply(w1,y0),
         np.multiply(x1,y0) - np.multiply(y1,x0) + np.multiply(z1,w0) + np.multiply(w1,z0),
        np.multiply(-x1,x0) - np.multiply(y1,y0) - np.multiply(z1,z0) + np.multiply(w1,w0)))


def qInvArray(q):
    # q: 4xN
    q = np.copy(q) # must not change input
    q[0:3,:] *= -1
    return np.divide(q,np.sum(np.square(q),axis=0))


def rotMat(q):
    # q: 4x1
    q0 = q[3,0]
    qv = q[0:3,0]
    qvs = skew(qv)
    return q0**2*ml.identity(3) + 2*q0*qvs + qv*qv.T + qvs**2


def skew(v):
    # v: 4x1
    v1 = v[0,0]
    v2 = v[1,0]
    v3 = v[2,0]
    return np.matrix([[0,-v3,v2],[v3,0,-v1],[-v2,v1,0]])


if __name__ == '__main__':
    model_learning()
