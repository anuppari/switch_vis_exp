#!/usr/bin/env python

# Source Author: Anup Parikh
#
# Circle follower for turtlebot

import rospy
import numpy as np
import tf
from geometry_msgs.msg import Twist, PoseStamped
from gazebo_msgs.msg import ModelStates

qMult = tf.transformations.quaternion_multiply # quaternion multiplication function handle
qInv = tf.transformations.quaternion_inverse # quaternion inverse function handle
q2m = tf.transformations.quaternion_matrix # quaternion to 4x4 transformation matrix
q2e = tf.transformations.euler_from_quaternion # euler angles from quaternion

kpw = 1
kph = 0.5
kpv = 0.2
center = np.array([0,0]) # w.r.t. world (mocap)
desiredRadius = 1  # [m]
nominalVel = 0.5 # [m/s]
radiusThresh = 0.5

def turtlebot_control(): # Main node definition
    global velPub, tfListener, agentID
    global lastLinCmd, mode
    
    rospy.init_node('turtlebot_control') #initialize node
    
    tfListener = tf.TransformListener() # get transform listener
    
    #sub = rospy.Subscriber('pose',PoseStamped,controlCB,queue_size=1)
    sub = rospy.Subscriber('/gazebo/model_states',ModelStates,controlCB,queue_size=1)
    
    velPub = rospy.Publisher('cmd_vel_mux/input/navi',Twist,queue_size=1) # command velocity
    
    rospy.spin()


def control2(data):
    position = data.pose[1].position
    orientation = data.pose[1].orientation
    trans = np.array([position.x,position.y,position.z])
    quat = np.array([orientation.x,orientation.y,orientation.z,orientation.w])
    
    relPos = trans[0:2] - center
    actualRadius = np.sqrt(np.sum(np.power(relPos,2)))
    s = np.arctan2(relPos[1],relPos[0])
    
    theta = q2e(quat,axes='rzyx')[0]
    thetaS = wrapAngle(s+np.pi/2)
    
    cs = 1/desiredRadius
    dcds = 0
    d = desiredRadius - actualRadius
    thetaE = theta - thetaS
    
    sDot = (v/(1-d*cs))*np.cos(thetaE)
    dDot = v*np.sin(thetaE)
    z1 = s
    z2 = d
    z3 = (1-d*cs)*np.tan(thetaE)
    v1 = sDot
    v2 = -k2*v1*z2 - k3*np.absolute(v1)*z3
    thetaEdot = (v2 + (dDot*cs + d*dcds*sDot)*np.tan(thetaE))/((1-d*cs)*(1+np.power(np.tan(thetaE),2)))
    omega = thetaEdot + sDot*cs
    
    twistMsg = Twist()
    twistMsg.linear.x = v
    twistMsg.angular.z = omega
    velPub.publish(twistMsg)


def controlCB(data):
    position = data.pose[1].position
    orientation = data.pose[1].orientation
    trans = np.array([position.x,position.y,position.z])
    quat = np.array([orientation.x,orientation.y,orientation.z,orientation.w])
    
    relPos = trans[0:2] - center
    actualRadius = np.sqrt(np.sum(np.power(relPos,2)))
    actualTheta = np.arctan2(relPos[1],relPos[0])
    
    actualHeading = q2e(quat,axes='rzyx')[0]
    nominalDesHeading = wrapAngle(actualTheta+np.pi/2)
    
    radiusError = desiredRadius - actualRadius
    if radiusError < -1*radiusThresh:
        desiredHeading = wrapAngle(actualTheta+np.pi)
    elif radiusError > radiusThresh:
        desiredHeading = actualTheta
    else:
        desiredHeading = wrapAngle(nominalDesHeading - kph*radiusError)
    
    print "desiredHeading: "+str(desiredHeading)
    print "actualHeading: "+str(actualHeading)
    
    headingError = desiredHeading - actualHeading
    omega = kpw*headingError
    v = np.maximum(nominalVel - kpv*np.absolute(headingError),0)
    
    twistMsg = Twist()
    twistMsg.linear.x = v
    twistMsg.angular.z = omega
    velPub.publish(twistMsg)


def wrapAngle(angle):
    return (angle+np.pi) % (2*np.pi) - np.pi

def rotateVec(p,q):
    return qMult(q,qMult(np.append(p,0),qInv(q)))[0:3]


if __name__ == '__main__':
    try:
        turtlebot_control()
    except rospy.ROSInterruptException:
        pass
