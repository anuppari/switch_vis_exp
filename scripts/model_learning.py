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

# Note that a single layer NN, W.T*sigma, can be represented as kron(I,sigma.T)*vec(W),
# where sigma represent basis functions, W represents a matrix of ideal weights, and 
# I is the identity matrix of size col(W).
# Also note that this implementation is based on the assumption that the target follows a 
# vector field fixed to the ground frame, i.e., vTg = phi1(xTGg,qTG), wTGt = phi2(xTGg,qTG),
# where xTGg is the target position with respect to the ground, vTGg is the target velocity with
# respect to the ground, both expressed in ground coordinates. qTG is the target orientation
# with respect to the ground, and wTGt is the angular velocity of the target expressed in 
# the coordinate system on fixed to the target. Hence a NN should be designed as
# [vTGg.T, wTGt.T].T = kron(I(7),sigma(xTGg,qTG).T)*vec(W)
# However, eta (i.e., x and q) represents the pose of the target with respect to the camera.
# The ground coordinates can be generated as
# xTGg = Q(qCG)*x + xCGg and qTG = qCG*q, where xCGg is the camera position with respect to
# ground expressed in ground coordinates and qCG is the camera orientation with respect to 
# the ground. This transformation is performed in the basis function, hence
# [vTGg.T, wTGt.T].T = kron(I(7),sigma(eta,xCGg,qCG).T)*vec(W).
# Further, the velocity signals needed in the estimator are vTGc and 0.5*B(q)*wGTt, where
# vTGc is the target velocity with respect to ground, expressed in camera coordinates, and B
# is the quaternion differential matrix. vTGc can be generated from target velocities as
# vTGc = Q(qCG^-1)*vTGg, leading to [vTGc.T,(0.5*B(q)*wGTt).T].T = D(q,qCG)*[vTGg.T,wGTt.T].T, where 
# D(q,qCG) = block_diag(Q(qCG^-1),0.5*B(q)) = [Q(qCG^-1),    0   ]
#                                             [   0,     0.5*B(q)]
# Finally, this all means that the NN is given by Y*theta, where
# Y = D(q,qCG)*kron(I(7),sigma(eta,xCGg,qCG).T) = kron(D(q,qCG),sigma(eta,xCGg,qCG).T)
# and theta = vec(W).


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
        
        # Update integration buffers
        self.tBuff.append(poseData.header.stamp.to_sec())
        self.etaBuff.append(self.eta)
        self.fBuff.append(f(self.eta,self.vCc,self.wGCc))
        self.sigmaBuff.append(self.sigma(self.eta,self.xCam,self.qCam))
        while self.tBuff[-1]-self.tBuff[0] > self.intWindow: # more accurate then setting maxlen
            self.tBuff.popleft()
            self.etaBuff.popleft()
            self.fBuff.popleft()
            self.sigmaBuff.popleft()
        
        # Integrate
        npTbuff = np.array([ti-self.tBuff[0] for ti in self.tBuff])
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
    np.set_printoptions(precision=5,suppress=True,threshold=20000)
    
    # Node parameters
    nodeName = rospy.get_name()
    k1 = 3
    k2 = 0.1
    kCL = 1
    intWindow = 1
    CLstackSize = 500
    stackFill = 0
    visibilityTimout = rospy.get_param(nodeName+"/visibilityTimout",0.2)
    
    # Setup Neural Network
    a = rospy.get_param(nodeName+"/a",1.0)
    b = rospy.get_param(nodeName+"/b",1.0)
    x0 = rospy.get_param(nodeName+"/x0",0.0)
    y0 = rospy.get_param(nodeName+"/y0",0.0)
    mapWidth = 2*a
    mapHeight = 2*b
    numKernalWidth = 4
    numKernalHeight = 4
    cov = 0.3*ml.eye(2)
    muX = np.linspace(x0-mapWidth,x0+mapWidth,numKernalWidth)
    muY = np.linspace(y0-mapHeight,y0+mapHeight,numKernalHeight)
    (muX,muY) = np.meshgrid(muX,muY)
    mu = np.asmatrix(np.vstack((muX.flatten(),muY.flatten())).T) # Nx2
    numKernal = mu.shape[0]
    sigma = lambda eta, xCam, qCam : sigmaGen(eta,xCam,qCam,mu,cov)
    
    # Initialize integral concurrent learning history stacks
    etaStack = np.zeros((CLstackSize,7,1))
    scriptFstack = np.zeros((CLstackSize,7,1))
    scriptYstack = np.zeros((CLstackSize,7,6*numKernal))
    Gamma = np.matrix(np.identity(6*numKernal))
    
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
    xCam = ml.zeros((3,1))
    qCam = np.matrix([0,0,0,1]).T
    print "here2"
    numPts = 2000
    numPreSeed = CLstackSize
    stackFill = numPreSeed
    eta = np.asmatrix(np.vstack((2*mapWidth*(np.random.random(numPts)-0.5)+x0,2*mapHeight*(np.random.random(numPts)-0.5)+y0,np.zeros((5,numPts))))) # 7xM
    print "here3"
    Y = sigma(eta,xCam,qCam) # 7Mx6N
    scriptYstack[0:numPreSeed,:,:] = Y.A.reshape(numPts,7,6*numKernal)[0:numPreSeed,:,:]
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
    Q = rotMat(np.asmatrix(qInvArray(qCam)))
    bMat = np.ascontiguousarray(np.array([np.append(Q*np.matrix([twistMsg.linear.x,twistMsg.linear.y,twistMsg.linear.z]).T,0.5*differentialMat(etai[0,3:7].T)*np.matrix([twistMsg.angular.x,twistMsg.angular.y,twistMsg.angular.z]).T,axis=0) for twistMsg, etai in zip(resp.twist,eta.T)]).flatten())
    etaStack[0:numPreSeed,:,:] = bMat.reshape(-1,7,1)[0:numPreSeed,:,:]
    print "here8"
    #WvecIdeal = np.asmatrix(np.linalg.lstsq(A,bMat)[0])
    thetaIdeal = np.asmatrix(np.linalg.lstsq(A,bMat)[0]).T
    thetaHat = thetaIdeal
    print thetaIdeal.shape
    #WvecIdeal = np.asmatrix(scipy.sparse.linalg.lsqr(A,bMat)[0])
    print "here9"
    #Wideal = np.reshape(WvecIdeal,(7,numKernal)).T
    bHat = bHat = (Y*thetaIdeal).reshape(-1,7)
    error = bMat.reshape(-1,7)[:,0:3] - bHat[:,0:3]
    #print error
    #print np.mean(np.matrix(np.linalg.norm(bHat[:,0:3],axis=1)).T>np.matrix(np.linalg.norm(bMat.reshape(-1,7)[:,0:3],axis=1)).T)
    
    
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
    thetaHat = ml.zeros((6*numKernal,1))
    r = rospy.Rate(200)
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
        scriptEta = callbacks.scriptEta
        scriptF = callbacks.scriptF
        scriptY = callbacks.scriptY
        vTt = callbacks.vTt #DEBUG
        wGTt = callbacks.wGTt #DEBUG
        phi = np.append(rotateVec(vTt,q),0.5*differentialMat(q)*wGTt,axis=0) #DEBUG
        
        # Estimation
        if callbacks.estimatorOn: # estimator
            #print "W: " + str((sigma(eta,xCam,qCam)*thetaIdeal).T[0,0:3])
            #print "phi: " + str(phi.T[0,0:3])
            #print "ON: " + str(np.linalg.norm((sigma(eta,xCam,qCam)*thetaIdeal).T[0,0:3]) > np.linalg.norm(phi.T[0,0:3]))
            #print f(eta,vCc,wGCc)
            #etaHatDot = Wideal.T*sigma(eta,xCam,qCam) + f(eta,vCc,wGCc) + k1*etaTilde + k2*np.sign(etaTilde)
            etaHatDot = sigma(eta,xCam,qCam)*thetaIdeal + f(eta,vCc,wGCc) + k1*etaTilde + k2*np.sign(etaTilde)
            #thetaHatDot1 = Gamma*sigma(eta,xCam,qCam).T*etaTilde + kCL*Gamma*np.sum([np.dot(scriptYstack[i,:,:].T,(etaStack[i,:,:] - scriptFstack[i,:,:] - scriptYstack[i,:,:]*thetaHat)) for i in range(CLstackSize)])
            thetaHatDot = Gamma*sigma(eta,xCam,qCam).T*etaTilde + kCL*Gamma*np.sum(multArray(scriptYstack.transpose(0,2,1),(etaStack - scriptFstack - np.dot(scriptYstack,thetaHat.A).reshape(-1,7,1))),axis=0)
            
            #Wdot = Gamma*sigma(eta,xCam,qCam)*etaTilde.T + kCL*Gamma*
            #etaHatDot = phi + f(eta,vCc,wGCc) + k1*etaTilde + k2*np.sign(etaTilde) #DEBUG
            
            #print scriptYstack
            
            # Update History Stack
            if stackFill < CLstackSize: # initially, always add data to stack
                etaStack[stackFill,:,:] = scriptEta
                scriptFstack[stackFill,:,:] = scriptF
                scriptYstack[stackFill,:,:] = scriptY
                stackFill += 1
            else:
                YtY = multArray(scriptYstack.transpose(0,2,1),scriptYstack) # all Y.T*Y
                stackSum = np.sum(YtY,axis=0) # sum of Y.T*Y
                currEig = np.amin(np.linalg.eigvals(stackSum)) # min eigenvalue of current stack
                stackSum += scriptY.T*scriptY # add new data to sum
                minEigs = np.amin(np.squeeze(np.linalg.eigvals(np.split(stackSum - YtY,CLstackSize,axis=0))),axis=1) # min eig val for each replacement
                if np.amax(minEigs) > currEig: # replace data if eigenvalue increase
                    ind = np.argmax(minEigs)
                    etaStack[ind,:,:] = scriptEta
                    scriptFstack[ind,:,:] = scriptF
                    scriptYstack[ind,:,:] = scriptY
                print currEig
                print stackSum
            
        else: # predictor
            #print "W: " + str((sigma(etaHat,xCam,qCam)*thetaIdeal).T[0,0:3])
            #print "phi: " + str(phi.T[0,0:3])
            #print "OFF: " + str(np.linalg.norm((sigma(etaHat,xCam,qCam)*thetaIdeal).T[0,0:3]) > np.linalg.norm(phi.T[0,0:3]))
            #print f(etaHat,vCc,wGCc)
            #print Wideal.T*sigma(etaHat,xCam,qCam), phi
            #etaHatDot = Wideal.T*sigma(etaHat,xCam,qCam) + f(etaHat,vCc,wGCc)
            etaHatDot = sigma(etaHat,xCam,qCam)*thetaIdeal + f(etaHat,vCc,wGCc)
            #thetaHatDot = kCL*Gamma*np.sum([np.dot(scriptYstack[i,:,:].T,(etaStack[i,:,:] - scriptFstack[i,:,:] - scriptYstack[i,:,:]*thetaHat)) for i in range(CLstackSize)])
            thetaHatDot = kCL*Gamma*np.sum(multArray(scriptYstack.transpose(0,2,1),(etaStack - scriptFstack - np.dot(scriptYstack,thetaHat.A).reshape(-1,7,1))),axis=0)
            #etaHatDot = phi + f(etaHat,vCc,wGCc) #DEBUG
        
        etaHat += etaHatDot*delT
        #thetaHat += thetaHatDot*delT
        #print "delT: " + str(delT)
        #print "error: " + str((Wideal.T*sigma(eta,xCam,qCam) - phi).T)
        
        # Publish
        xHat = etaHat[0:3,:]
        qHat = etaHat[3:,:]
        br.sendTransform(xHat,qHat,rospy.Time.now(),"ugv0estimate","image")
        
        #r.sleep()


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
    #Bqt = differentialMat(qg.A)
    #Y = np.zeros((M,7,6*N))
    T = np.concatenate((np.concatenate((np.repeat(Q[None,:,:].A,M,axis=0),np.zeros((M,3,3))),axis=2),np.concatenate((np.zeros((M,4,3)),0.5*Bq),axis=2)),axis=1)
    Y = multArray(T,np.kron(np.identity(6),sigma.reshape(M,1,N)))
    #for i in range(M):
        ##T = scipy.linalg.block_diag(Q,np.dot(Bq[i,:,:],Bqt[i,:,:].T))
        #T = scipy.linalg.block_diag(Q,0.5*Bq[i,:,:])
        #Y[i,:,:] = np.kron(T,sigma[i,:])
    
    return np.asmatrix(Y.reshape(7*M,6*N))


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


def multArray(A,B):
    # A: MxNxD or MxNx1
    # B: NxKxD or NxKx1
    # any combination
    Adepth = (A.ndim is 3) and (A.shape[0] > 1)
    Bdepth = (B.ndim is 3) and (B.shape[0] > 1)
    
    if Adepth and Bdepth:
        return np.sum(np.transpose(A,(0,2,1)).reshape(-1,A.shape[2],A.shape[1],1)*B.reshape(-1,B.shape[1],1,B.shape[2]),-3)
    elif Bdepth:
        if A.ndim is 3:
            return np.dot(B.transpose(0,2,1),A.transpose(0,2,1)).transpose(0,2,1)
        else:
            return np.dot(B.transpose(0,2,1),A.T).transpose(0,2,1)
    else:
        return np.dot(A,B)


if __name__ == '__main__':
    model_learning()
