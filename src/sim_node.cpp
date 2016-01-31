#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/TwistStamped.h>
#include <image_geometry/pinhole_camera_model.h>
#include <nav_msgs/Odometry.h>
#include <aruco_ros/Center.h>
#include <sensor_msgs/Joy.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>

Eigen::Matrix<double,4,3> diffMat(const Eigen::Quaterniond);

class Sim
{
    ros::NodeHandle nh;
    ros::Publisher targetVelPub;
    ros::Publisher camVelPub;
    ros::Publisher featurePub;
    ros::Publisher camInfoPub;
    ros::Subscriber joySub;
    tf::TransformBroadcaster tfbr;
    ros::Timer integrateTimer;
    ros::Timer velPubTimer;
    ros::Timer imagePubTimer;
    
    // Parameters
    double intTime;
    int markerID;
    cv::Mat camMat;
    cv::Mat distCoeffs;
    double radius;
    double period;
    
    // States
    Eigen::Vector3d camPos;
    Eigen::Quaterniond camOrient;
    Eigen::Vector3d targetPos;
    Eigen::Quaterniond targetOrient;
    Eigen::Vector3d camLinVel;
    Eigen::Vector3d camAngVel;
    Eigen::Vector3d targetLinVel;
    Eigen::Vector3d targetAngVel;
    
public:
    Sim()
    {
        std::string cameraName = "camera";
        
        // Publishers
        camVelPub = nh.advertise<geometry_msgs::TwistStamped>("image/body_vel",10);
        targetVelPub = nh.advertise<nav_msgs::Odometry>("ugv0/odom",10);
        featurePub = nh.advertise<aruco_ros::Center>("markerCenters",10);
        camInfoPub = nh.advertise<sensor_msgs::CameraInfo>(cameraName+"/camera_info",10,true); // latched
        joySub = nh.subscribe<sensor_msgs::Joy>("/joy",1,&Sim::joyCB,this);
        
        // Initialize states
        camPos = Eigen::Vector3d::Zero();
        camOrient = Eigen::Quaterniond::Identity();
        targetPos = Eigen::Vector3d::Zero();
        targetOrient = Eigen::Quaterniond::Identity();
        camLinVel = Eigen::Vector3d::Zero();
        camAngVel = Eigen::Vector3d::Zero();
        targetLinVel = Eigen::Vector3d::Zero();
        targetAngVel = Eigen::Vector3d::Zero();
        
        // Initialize Parameters
        intTime = 1.0/300.0;
        markerID = 100;
        radius = 2;
        period = 30;
        
        // Initialize and Publish camera info
        sensor_msgs::CameraInfo camInfoMsg;
        double K[] = {1,0,0,0,1,0,0,0,1};
        double D[] = {0,0,0,0,0};
        for (int i = 0; i < 9; i++)
        {
            camInfoMsg.K[i] = K[i];
            if (i<5) camInfoMsg.D[i] = D[i];
        }
        image_geometry::PinholeCameraModel cam_model;
        cam_model.fromCameraInfo(camInfoMsg);
        camMat = cv::Mat(cam_model.fullIntrinsicMatrix());
        camMat.convertTo(camMat,CV_32FC1);
        cam_model.distortionCoeffs().convertTo(distCoeffs,CV_32FC1);
        camInfoPub.publish(camInfoMsg); // latched
        
        // Integrator
        integrateTimer = nh.createTimer(ros::Duration(intTime),&Sim::integrateCB,this,true);
        
        // Other publishers
        velPubTimer = nh.createTimer(ros::Duration(1.0/300.0),&Sim::velPubCB,this,true);
        imagePubTimer = nh.createTimer(ros::Duration(1.0/30.0),&Sim::imagePubCB,this,true);
    }
    
    void integrateCB(const ros::TimerEvent& event)
    {
        // Integrate camera pose
        camPos += camLinVel*intTime;
        Eigen::Vector4d camOrientTemp(camOrient.w(),camOrient.x(),camOrient.y(),camOrient.z());
        camOrientTemp += 0.5*diffMat(camOrient)*camAngVel*intTime;
        camOrient = Eigen::Quaterniond(camOrientTemp(0),camOrientTemp(1),camOrientTemp(2),camOrientTemp(3));
        camOrient.normalize();
        
        // Integrate target pose
        targetPos += targetLinVel*intTime;
        Eigen::Vector4d targetOrientTemp(targetOrient.w(),targetOrient.x(),targetOrient.y(),targetOrient.z());
        targetOrientTemp += 0.5*diffMat(targetOrient)*targetAngVel*intTime;
        targetOrient = Eigen::Quaterniond(targetOrientTemp(0),targetOrientTemp(1),targetOrientTemp(2),targetOrientTemp(3));
        targetOrient.normalize();
        
        // Publish camera tf
        tf::Transform camTransform;
        camTransform.setOrigin(tf::Vector3(camPos(0),camPos(1),camPos(2)));
        camTransform.setRotation(tf::Quaternion(camOrient.x(),camOrient.y(),camOrient.z(),camOrient.w()));
        tfbr.sendTransform(tf::StampedTransform(camTransform,ros::Time::now(),"world","camera"));
        
        // Publish target tf
        tf::Transform targetTransform;
        targetTransform.setOrigin(tf::Vector3(targetPos(0),targetPos(1),targetPos(2)));
        targetTransform.setRotation(tf::Quaternion(targetOrient.x(),targetOrient.y(),targetOrient.z(),targetOrient.w()));
        tfbr.sendTransform(tf::StampedTransform(targetTransform,ros::Time::now(),"world","ugv0"));
    }
    
    void velPubCB(const ros::TimerEvent& event)
    {
        geometry_msgs::TwistStamped msg;
        msg.header.stamp = ros::Time::now();
        
        msg.twist.linear.x = camLinVel(0);
        msg.twist.linear.y = camLinVel(1);
        msg.twist.linear.z = camLinVel(2);
        msg.twist.angular.x = camAngVel(0);
        msg.twist.angular.y = camAngVel(0);
        msg.twist.angular.z = camAngVel(0);
        camVelPub.publish(msg);
        
        msg.twist.linear.x = targetLinVel(0);
        msg.twist.linear.y = targetLinVel(1);
        msg.twist.linear.z = targetLinVel(2);
        msg.twist.angular.x = targetAngVel(0);
        msg.twist.angular.y = targetAngVel(0);
        msg.twist.angular.z = targetAngVel(0);
        targetVelPub.publish(msg);
    }
    
    void imagePubCB(const ros::TimerEvent& event)
    {
        // Target w.r.t. camera
        Eigen::Vector3d t2cPos = camOrient.inverse()*(targetPos - camPos);
        
        // Convert to OpenCV
        cv::Mat t2cPosCV;
        cv::eigen2cv(t2cPos,t2cPosCV);
        
        // Project points to determine pixel coordinates
        cv::Mat imagePoint;
        cv::projectPoints(t2cPosCV,cv::Mat::zeros(1,3,CV_8UC1),cv::Mat::zeros(1,3,CV_8UC1),camMat,distCoeffs,imagePoint);
        
        // Publish image point
        aruco_ros::Center msg;
        msg.header.stamp = ros::Time::now();
        char buf[10];
        std::sprintf(buf,"%d",markerID);
        msg.header.frame_id = buf;
        msg.x = imagePoint.at<double>(0);
        msg.y = imagePoint.at<double>(1);
        featurePub.publish(msg);
    }
    
    void joyCB(const sensor_msgs::JoyConstPtr& joyMsg)
    {
        if (joyMsg->buttons[2]) // x - drive in circle
        {
            radius += 0.1*joyMsg->axes[6];
            period -= 10*joyMsg->axes[7];
            targetAngVel << 0, 0, 2*M_PI/period;
            targetLinVel << 2*M_PI*radius/period, 0, 0;
        }
        else if (joyMsg->buttons[1]) // b - reset target
        {
            targetPos << 0, -radius, 0;
            targetOrient.setIdentity();
        }
        else
        {
            Eigen::Vector3d linVel(joyMsg->axes[0], joyMsg->axes[1], 0);
            Eigen::Vector3d angVel(joyMsg->axes[4], joyMsg->axes[3], joyMsg->axes[5]-joyMsg->axes[2]);
            if (joyMsg->buttons[5]) // Right bumper, control camera
            {
                camLinVel = linVel;
                camAngVel = angVel;
            }
            else // control target
            {
                targetLinVel = linVel;
                targetAngVel = angVel;
                camLinVel << 0,0,0;
                camAngVel << 0,0,0;
            }
        }
    }
}; // End Sim class

// Calculate differential matrix for relationship between quaternion derivative and angular velocity.
// qDot = 1/2*B*omega 
// See strapdown inertial book. If quaternion is orientation of frame 
// B w.r.t N in the sense that nP = q*bP*q', omega is ang. vel of frame B w.r.t. N,
// i.e. N_w_B, expressed in the B coordinate system
// q = [w,x,y,z]
Eigen::Matrix<double,4,3> diffMat(const Eigen::Quaterniond q)
{
    Eigen::Matrix<double,4,3> B;
    B << -q.x(), -q.y(), -q.z(), q.w(), -q.z(), q.y(), q.z(), q.w(), -q.x(), -q.y(), q.x(), q.w();
    return B;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sim_node");
    
    Sim obj;
    
    ros::spin();
    return 0;
}
