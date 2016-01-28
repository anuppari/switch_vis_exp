#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <aruco_ros/Center.h>

#include <Eigen/Dense>

class Sim
{
    ros::NodeHandle nh;
    ros::Publisher targetVelPub;
    ros::Publisher camVelPub;
    ros::Publisher featurePub;
    ros::Publisher camInfoPub;
    tf::TransformBroadcaster tfbr;
    ros::Timer integrateTimer;
    ros::Timer velPubTimer;
    ros::Timer imagePubTimer;
    
    // Parameters
    double intTime = 1.0/300.0;
    
    // States
    Eigen::Vector3d camPos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond camOrient = Eigen::Quaterniond::Identity();
    Eigen::Vector3d targetPos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond targetOrient = Eigen::Quaterniond::Identity();
public:
    Sim()
    {
        cameraName = "camera";
        
        // Publishers
        camVelPub = nh.advertise<geometry_msgs::TwistStamped>("image/body_vel",10);
        targetVelPub = nh.advertise<nav_msgs::Odometry>("ugv0/odom",10);
        featurePub = nh.advertise<aruco_ros::Center>("markerCenters",10);
        camInfoPub = nh.advertise<sensor_msgs::CameraInfo>(cameraName+"/camera_info",10,true); // latched
        
        // Publish camera info
        sensor_msgs::CameraInfo camInfoMsg;
        camInfoMsg.K = [1,0,0,0,1,0,0,0,1];
        camInfoMsg.D = [0,0,0,0,0];
        camInfoPub.publish(camInfoMsg); // latched
        
        // Integrator
        integrateTimer = nh.createTimer(ros::Duration(intTime),&EKF::integrateCB,this,true);
        
        // Other publishers
        velPubTimer = nh.createTimer(ros::Duration(1.0/300.0),&EKF::velPubCB,this,true);
        imagePubTimer = nh.createTimer(ros::Duration(1.0/30.0),&EKF::imagePubCB,this,true);
    }
    
    void integrateCB(const ros::TimerEvent& event)
    {
        camPos += camLinVel*intTime;
        camOrient += 0.5*diffMat(camOrient)*camAngVel*intTime;
        targetPos += targetLinVel*intTime;
        targetOrient += 0.5*diffMat(targetOrient)*targetAngVel*intTime;
    }
    
    void velPubCB(const ros::TimerEvent& event)
    {
        geometry_msgs::TwistStamped msg;
        msg.header.stamp = ros::Time::now();
        
        msg.twist.linear = camLinVel.data();
        msg.twist.angular = camAngVel.data();
        camVelPub.publish(msg);
        
        msg.twist.linear = targetLinVel.data();
        msg.twist.angular = targetAngVel.data();
        targetVelPub.publish(msg);
    }
    
    void imagePubCB(const ros::TimerEvent& event)
    {
        
        aruco_ros::Center msg;
        msg.header.stamp = ros::Time::now();
    }
} // End Sim class

// Calculate differential matrix for relationship between quaternion derivative and angular velocity.
// qDot = 1/2*B*omega 
// See strapdown inertial book. If quaternion is orientation of frame 
// B w.r.t N in the sense that nP = q*bP*q', omega is ang. vel of frame B w.r.t. N,
// i.e. N_w_B, expressed in the B coordinate system
Matrix<double,4,3> diffMat(const Vector4d q)
{
    Matrix<double,4,3> B;
    B << q(3), -q(2), q(1), q(2), q(3), -q(0), -q(1), q(0), q(3), -q(0), -q(1), -q(2);
    return B;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sim_node");
    
    Sim obj;
    
    ros::spin();
    return 0;
}
