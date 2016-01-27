#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <image_geometry/pinhole_camera_model.h>
#include <switch_vis_exp/Output.h>
#include <aruco_ros/Center.h>

#include <opencv2/imgproc.hpp>
#include <Eigen/Dense> // Defines MatrixXd, Matrix3d, Vector3d, Quaterniond
#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace std;
using namespace Eigen;

// need a class in order publish in the callback
class EKF
{
    ros::NodeHandle nh;
    ros::Publisher outputPub;
    ros::Publisher pointPub;
    ros::Subscriber camInfoSub;
    ros::Subscriber targetVelSub;
    ros::Subscriber camVelSub;
    ros::Subscriber featureSub;
    tf::TransformListener tfl;
    ros::Timer watchdogTimer;
    ros::Timer switchingTimer;
    
    // parameters
    cv::Mat camMat;
    cv::Mat distCoeffs;
    bool gotCamParam;
    bool deadReckoning;
    bool artificialSwitching;
    double visibilityTimeout;
    string cameraName;
    string markerID;
    double q;   // EKF process noise scale
    double r;   // EKF measurement noise scale
    double delTon;
    double delToff;
    
    //states
    Vector3d xhat;
    Vector3d xlast;
    double lastImageTime;
    double lastVelTime;
    bool estimatorOn;
    Quaterniond qWorld2Odom; // Rotation for initializing dead reckoning
    Vector3d vTt;   // target linear velocity w.r.t. ground, expressed in target coordinate system
    Vector3d vCc;   // camera linear velocity w.r.t. ground, expressed in camera coordinate system
    Vector3d wGCc;  // camera angular velocity w.r.t. ground, expressed in camera coordinate system
    Matrix3d Q;     // EKF process noise covariance
    Matrix3d R;     // EKF measurement noise covariance
    Matrix<double,2,3> H;   // EKF measurement model
public:
    EKF()
    {
        // Get Parameters
        ros::NodeHandle nhp("~");
        nhp.param<bool>("deadReckoning", deadReckoning, false);
        nhp.param<bool>("artificialSwitching", artificialSwitching, false);
        nhp.param<double>("visibilityTimeout", visibilityTimeout, 0.2);
        nhp.param<string>("cameraName",cameraName,"camera");
        nhp.param<string>("markerID",markerID,"100");
        nhp.param<double>("q",q,4.0); // process noise
        nhp.param<double>("r",r,4.0); // measurement noise
        nhp.param<double>("delTon",delTon,4.0);
        nhp.param<double>("delToff",delToff,1.0);
        
        // Initialize states
        xhat << 0,0,0.1;
        xlast << 0,0,0.1;
        lastImageTime = ros::Time::now().toSec();
        lastVelTime = lastImageTime;
        estimatorOn = true;
        gotCamParam = false;
        
        // Initialize EKF matrices
        Q = q*Matrix3d::Identity();
        R = r*Matrix3d::Identity();
        H << 1,0,0,
             0,1,0;
        
        // Get camera parameters
        cout << cameraName+"/camera_info" << endl;
        camInfoSub = nh.subscribe(cameraName+"/camera_info",1,&EKF::camInfoCB,this);
        ROS_DEBUG("Waiting for camera parameters on topic ...");
        do {
            ros::spinOnce();
            ros::Duration(0.1).sleep();
        } while (!(ros::isShuttingDown()) and !gotCamParam);
        ROS_DEBUG("Got camera parameters");
        
        // Output publishers
        outputPub = nh.advertise<switch_vis_exp::Output>("output",10);
        pointPub = nh.advertise<geometry_msgs::PointStamped>("output_point",10);
        
        // Subscribers for feature and velocity data
        if (deadReckoning)
        {
            targetVelSub = nh.subscribe("ugv0/odom",1,&EKF::targetVelCBdeadReckoning,this);
        }
        else
        {
            targetVelSub = nh.subscribe("ugv0/body_vel",1,&EKF::targetVelCBmocap,this);
        }
        camVelSub = nh.subscribe("image/body_vel",1,&EKF::camVelCB,this);
        featureSub = nh.subscribe("markerCenters",1,&EKF::featureCB,this);
        
        // Switching
        if (artificialSwitching)
        {
            switchingTimer = nh.createTimer(ros::Duration(delTon),&EKF::switchingTimerCB,this,true);
        }
        else
        {
            // Initialize watchdog timer for feature visibility check
            watchdogTimer = nh.createTimer(ros::Duration(visibilityTimeout),&EKF::timeout,this,true);
            watchdogTimer.stop() // Dont start watchdog until feature first visible
        }
    }
    
    // If artificial switching, this method is called at set intervals (according to delTon, delToff) to toggle the estimator
    void switchingTimerCB(const ros::TimerEvent& event)
    {
        if (estimatorOn)
        {
            estimatorOn = false;
            switchingTimer = nh.createTimer(ros::Duration(delToff),&EKF::switchingTimerCB,this,true);
        }
        else
        {
            estimatorOn = true;
            switchingTimer = nh.createTimer(ros::Duration(delTon),&EKF::switchingTimerCB,this,true);
        }
    }
    
    // Watchdog callback. The only way to detect if the target is not visible is if the estimator callback has not been
    // called for a long time. If sufficient time has passed (visibilityTimeout), this is called to notify that target
    // is no longer visible. The estimator method (featureCB) resets the watchdog every time it is called, preventing
    // this method from running if featureCB has been called recently.
    void timeout(const ros::TimerEvent& event)
    {
        estimatorOn = false;
    }
    
    // gets camera intrinsic parameters
    void camInfoCB(const sensor_msgs::CameraInfoConstPtr& camInfoMsg)
    {
        //get camera info
        image_geometry::PinholeCameraModel cam_model;
        cam_model.fromCameraInfo(camInfoMsg);
        camMat = cv::Mat(cam_model.fullIntrinsicMatrix());
        camMat.convertTo(camMat,CV_32FC1);
        cam_model.distortionCoeffs().convertTo(distCoeffs,CV_32FC1);
        
        //unregister subscriber
        ROS_DEBUG("Got camera intrinsic parameters!");
        camInfoSub.shutdown();
        gotCamParam = true;
    }
    
    // Gets target velocities from mocap
    void targetVelCBmocap(const geometry_msgs::TwistStampedConstPtr& twist)
    {
        vTt << twist->twist.linear.x,twist->twist.linear.y,twist->twist.linear.z;
    }
    
    // Gets target velocities from turtlebot odometry
    void targetVelCBdeadReckoning(const nav_msgs::OdometryConstPtr& odom)
    {
        vTt << odom->twist.twist.linear.x,odom->twist.twist.linear.y,odom->twist.twist.linear.z;
    }
    
    // Gets camera velocities. Also, if target not visible (i.e. estimatorOn = false) publishes output (ground truth)
    // and does prediction 
    void camVelCB(const geometry_msgs::TwistStampedConstPtr& twist)
    {
        // Time
        ros::Time timeStamp = twist->header.stamp;
        double timeNow = timeStamp.toSec();
        double delT = timeNow - lastVelTime;
        lastVelTime = timeNow;
        
        // Camera velocities, expressed in camera coordinate system
        vCc << twist->twist.linear.x,twist->twist.linear.y,twist->twist.linear.z;
        wGCc << twist->twist.angular.x,twist->twist.angular.y,twist->twist.angular.z;
        
        if (!estimatorOn)
        {
            // Object trans w.r.t. image frame, for ground truth
            Vector3d trans;
            tf::StampedTransform transform;
            tfl.waitForTransform("image","ugv0",timeStamp,ros::Duration(0.1));
            tfl.lookupTransform("image","ugv0",timeStamp,transform);
            tf::Vector3 temp_trans = transform.getOrigin();
            trans << temp_trans.getX(),temp_trans.getY(),temp_trans.getZ();
            
            // Ground truth
            Vector3d x;
            x << trans.segment<2>(0)/trans(2),1/trans(2);
            xlast << x; // Update for optical flow
            
            // Object rotation w.r.t. image frame, for rotating target velocities into image coordinates
            try
            {
                Quaterniond quat;
                if (deadReckoning)
                {
                    tf::StampedTransform tfImage2World;
                    tf::StampedTransform tfOdom2Marker;
                    tfl.waitForTransform("image","world",timeStamp,ros::Duration(0.1));
                    tfl.lookupTransform("image","world",timeStamp,tfImage2World);
                    tfl.waitForTransform("ugv0/odom","ugv0/base_footprint",timeStamp,ros::Duration(0.1));
                    tfl.lookupTransform("ugv0/odom","ugv0/base_footprint",timeStamp,tfOdom2Marker);
                    
                    tf::Quaternion temp_quat = tfImage2World.getRotation();
                    Quaterniond qIm2W = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
                    temp_quat = tfOdom2Marker.getRotation();
                    Quaterniond qO2M = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
                    quat = qIm2W*qWorld2Odom*qO2M;
                }
                else
                {
                    tf::Quaternion temp_quat = transform.getRotation();
                    quat = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
                }
                
                // Target velocities expressed in camera coordinates
                Vector3d vTc = quat*vTt;
                
                // Update so that delT in featureCB is reasonable after switch
                lastImageTime = timeNow;
                
                // Convert to scalars to match notation in papers
                double vc1 = vCc(0);        double vc2 = vCc(1);        double vc3 = vCc(2);
                double vq1 = vTc(0);        double vq2 = vTc(1);        double vq3 = vTc(2);
                double w1 = wGCc(0);        double w2 = wGCc(1);        double w3 = wGCc(2);
                double x1hat = xhat(0);     double x2hat = xhat(1);     double x3hat = xhat(2);
                
                // Predictor
                double Omega1 = w3*x2hat - w2 - w2*pow(x1hat,2) + w1*x1hat*x2hat;
                double Omega2 = w1 - w3*x1hat - w2*x1hat*x2hat + w1*pow(x2hat,2);
                double xi1 = (vc3*x1hat - vc1)*x3hat;
                double xi2 = (vc3*x2hat - vc2)*x3hat;
                
                double y1hatDot = Omega1 + xi1 + vq1*x3hat - y1hat*vq3*x3hat;
                double y2hatDot = Omega2 + xi2 + vq2*x3hat - y2hat*vq3*x3hat;
                double y3hatDot = vc3*pow(x3hat,2) - (w2*x1hat - w1*x2hat)*x3hat - vq3*pow(x3hat,2);
                
                // Predict Covariance
                Matrix3d F = calculate_F(x,VCc,VTc,wGCc);
                MatrixXd Pdot = F*P+P*F.transpose() + Q;
                
                // Update states and covariance
                Vector3d xhatDot;
                xhatDot << x1hatDot, x2hatDot, x3hatDot;
                xhat += xhatDot*delT;
                P += Pdot*delT;
                
                // Publish output
                publishOutput(x,xhat,trans,timeStamp);
            }
            catch (tf::TransformException e)
            {
            }
        }
    }
    
    // Calculate linearization of dynamics (F) for EKF
    Matrix3d calculate_F(Vector3d x_,Vector3d VCc_,Vector3d VTc_,Vector3d wGCc_)
    {
        Matrix3d F;
        
        return F;
    }
    
    // Callback for estimator
    void featureCB(const aruco_ros::CenterConstPtr& center)
    {
        // Disregard erroneous tag tracks
        if (markerID.compare(center->header.frame_id) != 0)
        {
            return;
        }
        
        // Switching
        if (artificialSwitching)
        {
            if (!estimatorOn)
            {
                return;
            }
        }
        else
        {
            // Feature in FOV
            watchdogTimer.stop();
            estimatorOn = true;
        }
        
        // Time
        ros::Time timeStamp = center->header.stamp;
        double timeNow = timeStamp.toSec();
        double delT = timeNow - lastImageTime;
        lastImageTime = timeNow;
        
        // Object trans w.r.t. image frame, for ground truth
        Vector3d trans;
        tf::StampedTransform transform;
        tfl.waitForTransform("image","ugv0",timeStamp,ros::Duration(0.1));
        tfl.lookupTransform("image","ugv0",timeStamp,transform);
        tf::Vector3 temp_trans = transform.getOrigin();
        trans << temp_trans.getX(),temp_trans.getY(),temp_trans.getZ();
        
        // Object pose w.r.t. image frame
        if (deadReckoning)
        {
            tfl.waitForTransform("image",string("marker")+markerID,timeStamp,ros::Duration(0.1));
            tfl.lookupTransform("image",string("marker")+markerID,timeStamp,transform);
            
            try
            {
                // Additional transforms for predictor
                tf::StampedTransform tfWorld2Marker;
                tf::StampedTransform tfMarker2Odom;
                tfl.waitForTransform("world",string("marker")+markerID,timeStamp,ros::Duration(0.1));
                tfl.lookupTransform("world",string("marker")+markerID,timeStamp,tfWorld2Marker);
                tfl.waitForTransform("ugv0/base_footprint","ugv0/odom",timeStamp,ros::Duration(0.1));
                tfl.lookupTransform("ugv0/base_footprint","ugv0/odom",timeStamp,tfMarker2Odom);
                
                // Save transform
                tf::Quaternion temp_quat = tfWorld2Marker.getRotation();
                Quaterniond qW2M = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
                temp_quat = tfMarker2Odom.getRotation();
                Quaterniond qM2O = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
                qWorld2Odom = qW2M*qM2O;
            }
            catch (tf::TransformException e)
            {
            }
        }
        // else, use quaternion from image to ugv0 transform
        tf::Quaternion temp_quat = transform.getRotation();
        Quaterniond quat = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
        
        // Undistort image coordinates. Returns normalized Euclidean coordinates
        double ptsArray[2] = {center->x,center->y};
        cv::Mat pts(1,1,CV_64FC2,ptsArray);
        cv::Mat undistPts;
        cv::undistortPoints(pts,undistPts,camMat,distCoeffs);
        Vector3d x;
        x << undistPts.at<double>(0,0),undistPts.at<double>(0,1),1/trans(2);
        
        // Target velocities expressed in camera coordinates
        Vector3d vTc = quat*vTt;
        
        // Observer velocities
        Vector3d b = vTc - vCc;
        Vector3d w = -wGCc; //Vector3d::Zero();
        
        // EKF update
        Matrix3d K = P*H.transpose()*(H*P*H.transpose()+R).inverse();
        xhat += K*(x.head<2>()-xhat.head<2>());
        P = (Matrix3d::Identity()-K*H)*P;
        
        // Publish output
        publishOutput(x,xhat,trans,timeStamp);
        
        if (!artificialSwitching)
        {
            // Restart watchdog timer for feature visibility check
            watchdogTimer.start();
        }
    }
    
    // Method for publishing data (estimate, ground truth, etc) and publishing Point message for visualization
    void publishOutput(Vector3d y, Vector3d yhat, Vector3d trans, ros::Time timeStamp)
    {
        // Extra signals
        Vector3d XYZ = trans;
        Vector3d XYZhat;
        XYZhat << yhat(0)/yhat(2),yhat(1)/yhat(2),1/yhat(2);
        Vector3d error = y - yhat;
        Vector3d XYZerror = XYZ - XYZhat;
        
        // Publish output
        switch_vis_exp::Output outMsg = switch_vis_exp::Output();
        outMsg.header.stamp = timeStamp;
        outMsg.y[0] = y(0);                 outMsg.y[1] = y(1);                 outMsg.y[2] = y(2);
        outMsg.yhat[0] = yhat(0);           outMsg.yhat[1] = yhat(1);           outMsg.yhat[2] = yhat(2);
        outMsg.error[0] = error(0);         outMsg.error[1] = error(1);         outMsg.error[2] = error(2);
        outMsg.XYZ[0] = XYZ(0);             outMsg.XYZ[1] = XYZ(1);             outMsg.XYZ[2] = XYZ(2);
        outMsg.XYZhat[0] = XYZhat(0);       outMsg.XYZhat[1] = XYZhat(1);       outMsg.XYZhat[2] = XYZhat(2);
        outMsg.XYZerror[0] = XYZerror(0);   outMsg.XYZerror[1] = XYZerror(1);   outMsg.XYZerror[2] = XYZerror(2);
        outMsg.estimatorOn = estimatorOn;
        outMsg.usePredictor = true;
        outputPub.publish(outMsg);
        
        // Publish point
        geometry_msgs::PointStamped pntMsg = geometry_msgs::PointStamped();
        pntMsg.header.stamp = timeStamp;
        pntMsg.header.frame_id = "image";
        pntMsg.point.x = XYZhat(0);
        pntMsg.point.y = XYZhat(1);
        pntMsg.point.z = XYZhat(2);
        pointPub.publish(pntMsg);
    }


};//End of class EKF


int main(int argc, char** argv)
{
    ros::init(argc, argv, "ekf_node");
    
    EKF obj;
    
    ros::spin();
    return 0;
}
