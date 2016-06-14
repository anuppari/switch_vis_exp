#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/TwistStamped.h>
#include <image_geometry/pinhole_camera_model.h>
#include <nav_msgs/Odometry.h>
#include <aruco_ros/Center.h>
#include <sensor_msgs/Joy.h>
#include <switch_vis_exp/MapVel.h>

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
    ros::Publisher targetOdomPub;
    ros::Publisher camVelPub;
    ros::Publisher featurePub;
    ros::Publisher camInfoPub;
    ros::Subscriber joySub;
    ros::ServiceClient velocityMapClient;
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
    std::string cameraName;
    std::string targetName;
    double startTime;
    
    // States
    Eigen::Vector3d camPos;
    Eigen::Quaterniond camOrient;
    Eigen::Vector3d targetPos;
    Eigen::Quaterniond targetOrient;
    bool useVelocityMap;
    bool driveCircle;
    Eigen::Vector3d camLinVel; // expressed in body coordinate system
    Eigen::Vector3d camAngVel; // expressed in body coordinate system
    Eigen::Vector3d targetLinVel; // expressed in body coordinate system
    Eigen::Vector3d targetAngVel; // expressed in body coordinate system
    
public:
    Sim()
    {
        // Get Parameters
        ros::NodeHandle nhp("~");
        
        nhp.param<std::string>("cameraName",cameraName,"camera");
        nhp.param<std::string>("targetName",targetName,"ugv0");
        
        // Publishers
        camVelPub = nh.advertise<geometry_msgs::TwistStamped>("image/body_vel",10);
        targetVelPub = nh.advertise<geometry_msgs::TwistStamped>(targetName+"/body_vel",10);
        targetOdomPub = nh.advertise<nav_msgs::Odometry>(targetName+"/odom",10);
        featurePub = nh.advertise<aruco_ros::Center>("markerCenters",10);
        camInfoPub = nh.advertise<sensor_msgs::CameraInfo>(cameraName+"/camera_info",10,true); // latched
        joySub = nh.subscribe<sensor_msgs::Joy>("/joy",1,&Sim::joyCB,this);
        
        // velocity map service handle
        velocityMapClient = nh.serviceClient<switch_vis_exp::MapVel>("/get_velocity");
        
        // Initialize states
        useVelocityMap = false;
        driveCircle = false;
        camPos << 0,0,-10; //Eigen::Vector3d::Zero();
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
        startTime = ros::Time::now().toSec();
        
        // Initialize and Publish camera info
        sensor_msgs::CameraInfo camInfoMsg;
        double K[] = {1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0};
        double D[] = {0.0,0.0,0.0,0.0,0.0};
        std::vector<double> Dvec(D,D + sizeof(D)/sizeof(D[0]));
        for (int i = 0; i < 9; i++)
        {
            camInfoMsg.K[i] = K[i];
        }
        camInfoMsg.D = Dvec;
        image_geometry::PinholeCameraModel cam_model;
        cam_model.fromCameraInfo(camInfoMsg);
        camMat = cv::Mat(cam_model.fullIntrinsicMatrix());
        camMat.convertTo(camMat,CV_32FC1);
        cam_model.distortionCoeffs().convertTo(distCoeffs,CV_32FC1);
        camInfoPub.publish(camInfoMsg); // latched
        
        // Integrator
        integrateTimer = nh.createTimer(ros::Duration(intTime),&Sim::integrateCB,this,false);
        
        // Other publishers
        velPubTimer = nh.createTimer(ros::Duration(1.0/30.0),&Sim::velPubCB,this,false);
        imagePubTimer = nh.createTimer(ros::Duration(1.0/30.0),&Sim::imagePubCB,this,false);
    }
    
    void integrateCB(const ros::TimerEvent& event)
    {
        // Integrate camera pose
        camPos += camOrient*camLinVel*intTime; // convert from body to world and integrate
        Eigen::Vector4d camOrientTemp(camOrient.w(),camOrient.x(),camOrient.y(),camOrient.z());
        camOrientTemp += 0.5*diffMat(camOrient)*camAngVel*intTime;
        camOrient = Eigen::Quaterniond(camOrientTemp(0),camOrientTemp(1),camOrientTemp(2),camOrientTemp(3));
        camOrient.normalize();
        
        // Integrate target pose
        
        targetPos += targetOrient*targetLinVel*intTime; // convert from body to world and integrate
        Eigen::Vector4d targetOrientTemp(targetOrient.w(),targetOrient.x(),targetOrient.y(),targetOrient.z());
        targetOrientTemp += 0.5*diffMat(targetOrient)*targetAngVel*intTime;
        targetOrient = Eigen::Quaterniond(targetOrientTemp(0),targetOrientTemp(1),targetOrientTemp(2),targetOrientTemp(3));
        targetOrient.normalize();
        
        // Publish camera tf
        tf::Transform camTransform;
        camTransform.setOrigin(tf::Vector3(camPos(0),camPos(1),camPos(2)));
        camTransform.setRotation(tf::Quaternion(camOrient.x(),camOrient.y(),camOrient.z(),camOrient.w()));
        tfbr.sendTransform(tf::StampedTransform(camTransform,ros::Time::now(),"world","image"));
        
        // Publish target tf
        tf::Transform targetTransform;
        targetTransform.setOrigin(tf::Vector3(targetPos(0),targetPos(1),targetPos(2)));
        targetTransform.setRotation(tf::Quaternion(targetOrient.x(),targetOrient.y(),targetOrient.z(),targetOrient.w()));
        tfbr.sendTransform(tf::StampedTransform(targetTransform,ros::Time::now(),"world",targetName));
        tfbr.sendTransform(tf::StampedTransform(targetTransform,ros::Time::now(),targetName+"/odom",targetName+"/base_footprint"));
        
        // Publish marker tf for dead reckoning
        tf::Transform tfMarker2Cam;
        Eigen::Vector3d tM2C = camOrient.inverse()*(targetPos - camPos);
        Eigen::Quaterniond qM2C = camOrient.inverse()*targetOrient;
        targetTransform.setOrigin(tf::Vector3(tM2C(0),tM2C(1),tM2C(2)));
        targetTransform.setRotation(tf::Quaternion(qM2C.x(),qM2C.y(),qM2C.z(),qM2C.w()));
        tfbr.sendTransform(tf::StampedTransform(targetTransform,ros::Time::now(),"image","marker100"));
    }
    
    void velPubCB(const ros::TimerEvent& event)
    {
        geometry_msgs::TwistStamped twistMsg;
        twistMsg.header.stamp = ros::Time::now();
        
        // Publish camera velocities
        twistMsg.twist.linear.x = camLinVel(0);
        twistMsg.twist.linear.y = camLinVel(1);
        twistMsg.twist.linear.z = camLinVel(2);
        twistMsg.twist.angular.x = camAngVel(0);
        twistMsg.twist.angular.y = camAngVel(0);
        twistMsg.twist.angular.z = camAngVel(0);
        camVelPub.publish(twistMsg);
        
        // Publish target velocities
        if (useVelocityMap) {get_target_velocity_from_map();}
        if (driveCircle)
        {
            double timeNow = twistMsg.header.stamp.toSec();
            targetLinVel << std::sin(3*(timeNow - startTime)), std::cos(3*(timeNow - startTime)), 0;
            targetAngVel << 0,0,0;
        }
        twistMsg.twist.linear.x = targetLinVel(0);
        twistMsg.twist.linear.y = targetLinVel(1);
        twistMsg.twist.linear.z = targetLinVel(2);
        twistMsg.twist.angular.x = targetAngVel(0);
        twistMsg.twist.angular.y = targetAngVel(0);
        twistMsg.twist.angular.z = targetAngVel(0);
        targetVelPub.publish(twistMsg);
        
        // Publish target Odometry
        nav_msgs::Odometry odomMsg;
        odomMsg.header.stamp = ros::Time::now();
        odomMsg.header.frame_id = targetName+"/odom";
        odomMsg.child_frame_id = targetName+"/base_footprint";
        odomMsg.twist.twist = twistMsg.twist;
        targetOdomPub.publish(odomMsg);
        
        /*
        std::cout << "VelPubCB:" << std::endl;
        std::cout << "time: " << twistMsg.header.stamp.toSec() << std::endl;
        std::cout << "targetLinVel: " << targetLinVel.transpose() << std::endl;
        std::cout << "t2cPos: " << (camOrient.inverse()*(targetPos - camPos)).transpose() << std::endl;
        std::cout << "useVelocityMap: " << useVelocityMap << std::endl;
        */
    }
    
    void imagePubCB(const ros::TimerEvent& event)
    {
        // Target w.r.t. camera
        Eigen::Vector3d t2cPos = camOrient.inverse()*(targetPos - camPos);
        
        // Convert to OpenCV
        cv::Mat t2cPosCV;
        cv::eigen2cv((Eigen::MatrixXd) t2cPos.transpose(),t2cPosCV);
        
        // Project points to determine pixel coordinates
        cv::Mat imagePoint;
        cv::projectPoints(t2cPosCV,cv::Mat::zeros(1,3,CV_32F),cv::Mat::zeros(1,3,CV_32F),camMat,distCoeffs,imagePoint);
        
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
    
    void get_target_velocity_from_map()
    {
        // service msg handle
        switch_vis_exp::MapVel srv;
        
        // Construct request
        geometry_msgs::Pose poseMsg;
        poseMsg.position.x = targetPos(0);
        poseMsg.position.y = targetPos(1);
        poseMsg.position.z = targetPos(2);
        srv.request.pose.push_back(poseMsg);
        if (velocityMapClient.call(srv))
        {
            // get velocity
            Eigen::Vector3d des_lin_vel;
            Eigen::Vector3d des_ang_vel;
            des_lin_vel << srv.response.twist.at(0).linear.x, srv.response.twist.at(0).linear.y, srv.response.twist.at(0).linear.z;
            des_ang_vel << srv.response.twist.at(0).angular.x, srv.response.twist.at(0).angular.y, srv.response.twist.at(0).angular.z;
            
            // rotate velocity into target body frame
            targetLinVel = targetOrient.inverse()*des_lin_vel;
            targetAngVel = targetOrient.inverse()*des_ang_vel;
        }
    }
    
    void joyCB(const sensor_msgs::JoyConstPtr& joyMsg)
    {
        if (joyMsg->buttons[2]) // x - drive along velocity map
        {
            useVelocityMap = true;
            driveCircle = false;
            /*
            radius += 0.1*(joyMsg->buttons[12]-joyMsg->buttons[11]);
            period -= 10*(joyMsg->buttons[13]-joyMsg->buttons[14]);
            targetAngVel << 0, 0, 2*M_PI/period;
            targetLinVel << 2*M_PI*radius/period, 0, 0;
            */
        }
        else if (joyMsg->buttons[0]) // a - drive in circle
        {
            driveCircle = true;
            useVelocityMap = false;
        }
        else if (joyMsg->buttons[1]) // b - reset target
        {
            useVelocityMap = false;
            targetPos << 0, -radius, 0;
            targetOrient.setIdentity();
        }
        else
        {
            useVelocityMap = false;
            driveCircle = false;
            Eigen::Vector3d linVel(-1*joyMsg->axes[0], -1*joyMsg->axes[1], 0);
            Eigen::Vector3d angVel(-1*joyMsg->axes[4], joyMsg->axes[3], joyMsg->axes[2]-joyMsg->axes[5]);
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
