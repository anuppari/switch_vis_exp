#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <image_geometry/pinhole_camera_model.h>
#include <switch_vis_exp/Output.h>
#include <switch_vis_exp/MapVel.h>
#include <aruco_ros/Center.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace std;
using namespace Eigen;

// need a class in order publish in the callback
class SubscribeAndPublish
{
    ros::NodeHandle nh;
    ros::Publisher outputPub;
    ros::Publisher pointPub;
    ros::Subscriber camInfoSub;
    ros::Subscriber targetVelSub;
    ros::Subscriber camVelSub;
    ros::Subscriber featureSub;
    ros::ServiceClient targetVelClient;
    tf::TransformListener tfl;
    ros::Timer watchdogTimer;
    ros::Timer switchingTimer;
    
    // parameters
    cv::Mat camMat;
    cv::Mat distCoeffs;
    bool gotCamParam;
    bool usePredictor;
    bool deadReckoning;
    bool artificialSwitching;
    bool useVelocityMap;
    double visibilityTimeout;
    string cameraName;
    string targetName;
    string markerID;
    double k1;
    double k2;
    double k3;
    double initTime;
    double delTon;
    double delToff;
    
    //states
    Vector3d yhat;
    Vector2d ylast;
    double lastImageTime;
    double lastVelTime;
    bool estimatorOn;
    Quaterniond qWorld2Odom; // Rotation for initializing dead reckoning
    Vector3d vTt;   // target linear velocity w.r.t. ground, expressed in target coordinate system
    Vector3d vCc;   // camera linear velocity w.r.t. ground, expressed in camera coordinate system
    Vector3d wGCc;  // camera angular velocity w.r.t. ground, expressed in camera coordinate system
public:
    SubscribeAndPublish()
    {
        // Get Parameters
        ros::NodeHandle nhp("~");
        nhp.param<bool>("usePredictor", usePredictor, true);
        nhp.param<bool>("deadReckoning", deadReckoning, false);
        nhp.param<bool>("artificialSwitching", artificialSwitching, false);
        nhp.param<bool>("useVelocityMap", useVelocityMap, true);
        nhp.param<double>("visibilityTimeout", visibilityTimeout, 0.2);
        nhp.param<string>("cameraName",cameraName,"camera");
        nhp.param<string>("targetName",targetName,"ugv0");
        nhp.param<string>("markerID",markerID,"100");
        nhp.param<double>("delTon",delTon,4.0);
        nhp.param<double>("delToff",delToff,1.0);
        nhp.param<double>("k1",k1,4.0);
        nhp.param<double>("k2",k2,4.0);
        nhp.param<double>("k3",k3,4.0);
        
        // Initialize states
        double X0,Y0,Z0;
        nhp.param<double>("X0",X0,0);
        nhp.param<double>("Y0",Y0,0);
        nhp.param<double>("Z0",Z0,10);
        yhat << X0/Z0, Y0/Z0, 1.0/Z0;
        ylast = yhat.head<2>();
        
        lastImageTime = ros::Time::now().toSec();
        lastVelTime = lastImageTime;
        estimatorOn = true;
        gotCamParam = false;
        initTime = lastImageTime;
        
        // Get camera parameters
        cout << cameraName+"/camera_info" << endl;
        camInfoSub = nh.subscribe(cameraName+"/camera_info",1,&SubscribeAndPublish::camInfoCB,this);
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
        if (useVelocityMap)
        {
            targetVelClient = nh.serviceClient<switch_vis_exp::MapVel>("get_velocity");
        }
        if (deadReckoning)
        {
            targetVelSub = nh.subscribe(targetName+"/odom",1,&SubscribeAndPublish::targetVelCBdeadReckoning,this);
        }
        else
        {
            targetVelSub = nh.subscribe(targetName+"/body_vel",1,&SubscribeAndPublish::targetVelCBmocap,this);
        }
        camVelSub = nh.subscribe("image/body_vel",1,&SubscribeAndPublish::camVelCB,this);
        featureSub = nh.subscribe("markerCenters",1,&SubscribeAndPublish::featureCB,this);
        
        // Switching
        if (artificialSwitching)
        {
            switchingTimer = nh.createTimer(ros::Duration(delTon),&SubscribeAndPublish::switchingTimerCB,this,true);
        }
        else
        {
            // Initialize watchdog timer for feature visibility check
            watchdogTimer = nh.createTimer(ros::Duration(visibilityTimeout),&SubscribeAndPublish::timeout,this,true);
            watchdogTimer.stop(); // Dont start watchdog until feature first visible
        }
    }
    
    // If artificial switching, this method is called at set intervals (according to delTon, delToff) to toggle the estimator
    void switchingTimerCB(const ros::TimerEvent& event)
    {
        if (estimatorOn)
        {
            estimatorOn = false;
            switchingTimer = nh.createTimer(ros::Duration(delToff),&SubscribeAndPublish::switchingTimerCB,this,true);
        }
        else
        {
            estimatorOn = true;
            switchingTimer = nh.createTimer(ros::Duration(delTon),&SubscribeAndPublish::switchingTimerCB,this,true);
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
            // Update so that delT in featureCB is reasonable after switch
            lastImageTime = timeNow;
            
            // Object trans w.r.t. image frame, for ground truth
            tf::StampedTransform tfTarget2Im;
            try
            {
                tfl.waitForTransform("image",targetName,timeStamp,ros::Duration(0.1));
                tfl.lookupTransform("image",targetName,timeStamp,tfTarget2Im);
            }
            catch(tf::TransformException ex) { return; }
            Vector3d trans(tfTarget2Im.getOrigin().getX(), tfTarget2Im.getOrigin().getY(), tfTarget2Im.getOrigin().getZ());
            
            // Ground truth
            Vector3d y;
            y << trans.segment<2>(0)/trans(2),1/trans(2);
            ylast << y.head<2>(); // Update for optical flow
            
            // Target velocity in camera frame
            Vector3d vTc;
            
            // Get expected target velocity from velocity map
            if (useVelocityMap)
            {
                // service msg handle
                switch_vis_exp::MapVel srv;
                
                // transform state estimate to global frame
                tf::StampedTransform tfIm2W;
                try
                {
                    tfl.waitForTransform("world","image",timeStamp,ros::Duration(0.1));
                    tfl.lookupTransform("world","image",timeStamp,tfIm2W);
                }
                catch(tf::TransformException ex) { return; }
                Quaterniond qIm2W(tfIm2W.getRotation().getW(),tfIm2W.getRotation().getX(),tfIm2W.getRotation().getY(),tfIm2W.getRotation().getZ());
                Vector3d XYZworld = qIm2W*Vector3d(yhat(0)/yhat(2), yhat(1)/yhat(2), 1/yhat(2)) + Vector3d(tfIm2W.getOrigin().getX(),tfIm2W.getOrigin().getY(),tfIm2W.getOrigin().getZ());
                
                // Construct request
                geometry_msgs::Pose poseMsg;
                poseMsg.position.x = XYZworld(0);
                poseMsg.position.y = XYZworld(1);
                poseMsg.position.z = XYZworld(2);
                srv.request.pose.push_back(poseMsg);
                
                // Call and get response
                if (targetVelClient.call(srv))
                {
                    // get expected target velocity, expressed in world coordinates
                    Vector3d vTw(srv.response.twist.at(0).linear.x, srv.response.twist.at(0).linear.y, srv.response.twist.at(0).linear.z);
                    
                    // rotate velocity into image coordinate frame
                    vTc = qIm2W.inverse()*vTw;
                }
                else { return; }
            }
            else
            { // Get target velocities from communication
                
                // Object rotation w.r.t. image frame, for rotating target velocities into image coordinates
                Quaterniond qTarget2Im;
                if (deadReckoning)
                {
                    tf::StampedTransform tfImage2World;
                    tf::StampedTransform tfOdom2Marker;
                    try
                    {
                        tfl.waitForTransform("image","world",timeStamp,ros::Duration(0.1));
                        tfl.lookupTransform("image","world",timeStamp,tfImage2World);
                        tfl.waitForTransform(targetName+"/odom",targetName+"/base_footprint",timeStamp,ros::Duration(0.1));
                        tfl.lookupTransform(targetName+"/odom",targetName+"/base_footprint",timeStamp,tfOdom2Marker);
                    }
                    catch(tf::TransformException ex) { return; }
                    tf::Quaternion temp_quat = tfImage2World.getRotation();
                    Quaterniond qIm2W = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
                    temp_quat = tfOdom2Marker.getRotation();
                    Quaterniond qO2M = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
                    qTarget2Im = qIm2W*qWorld2Odom*qO2M;
                }
                else
                {
                    tf::Quaternion temp_quat = tfTarget2Im.getRotation();
                    qTarget2Im = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
                }
                
                // Target velocities expressed in camera coordinates
                vTc = qTarget2Im*vTt;
            }
            
            // Predictor/ZOH
            double y1hatDot = 0;
            double y2hatDot = 0;
            double y3hatDot = 0;
            if (usePredictor)
            {
                // Convert to scalars to match notation in papers
                double vc1 = vCc(0);        double vc2 = vCc(1);        double vc3 = vCc(2);
                double vq1 = vTc(0);        double vq2 = vTc(1);        double vq3 = vTc(2);
                double w1 = wGCc(0);        double w2 = wGCc(1);        double w3 = wGCc(2);
                double y1hat = yhat(0);     double y2hat = yhat(1);     double y3hat = yhat(2);
                
                // Predictor
                double Omega1 = w3*y2hat - w2 - w2*pow(y1hat,2) + w1*y1hat*y2hat;
                double Omega2 = w1 - w3*y1hat - w2*y1hat*y2hat + w1*pow(y2hat,2);
                double xi1 = (vc3*y1hat - vc1)*y3hat;
                double xi2 = (vc3*y2hat - vc2)*y3hat;
                
                y1hatDot = Omega1 + xi1 + vq1*y3hat - y1hat*vq3*y3hat;
                y2hatDot = Omega2 + xi2 + vq2*y3hat - y2hat*vq3*y3hat;
                y3hatDot = vc3*pow(y3hat,2) - (w2*y1hat - w1*y2hat)*y3hat - vq3*pow(y3hat,2);
            }
            else
            {
                y1hatDot = 0;
                y2hatDot = 0;
                y3hatDot = 0;
            }
            
            // Update states
            Vector3d yhatDot;
            yhatDot << y1hatDot, y2hatDot, y3hatDot;
            yhat += yhatDot*delT;
            
            // Publish output
            publishOutput(y,yhat,timeStamp);
        }
    }
    
    // Callback for estimator
    void featureCB(const aruco_ros::CenterConstPtr& center)
    {
        // Disregard erroneous tag tracks
        if (markerID.compare(center->header.frame_id) != 0) { return; }
        
        // Switching
        if (artificialSwitching)
        {
            if (!estimatorOn) { return; }
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
        if (delT == 0) { return; } // skip frame if camera driver misfires
        
        // Object trans w.r.t. image frame, for ground truth
        tf::StampedTransform tfTarget2Im;
        try
        {
            tfl.waitForTransform("image",targetName,timeStamp,ros::Duration(0.1));
            tfl.lookupTransform("image",targetName,timeStamp,tfTarget2Im);
        }
        catch(tf::TransformException ex) { return; }
        Vector3d trans(tfTarget2Im.getOrigin().getX(), tfTarget2Im.getOrigin().getY(), tfTarget2Im.getOrigin().getZ());
        
        // Ground truth
        Vector3d y(trans(0)/trans(2), trans(1)/trans(2), 1/trans(2));
        
        // Object pose w.r.t. image frame, for rotating target velocity into camera frame
        Quaterniond qTarget2Im;
        if (deadReckoning)
        {
            try
            {
                tfl.waitForTransform("image",string("marker")+markerID,timeStamp,ros::Duration(0.1));
                tfl.lookupTransform("image",string("marker")+markerID,timeStamp,tfTarget2Im);
            }
            catch(tf::TransformException ex) { return; }
            tf::Quaternion temp_quat = tfTarget2Im.getRotation();
            qTarget2Im = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());

            // Additional transforms for predictor
            tf::StampedTransform tfWorld2Marker;
            tf::StampedTransform tfMarker2Odom;
            try
            {
                tfl.waitForTransform("world",string("marker")+markerID,timeStamp,ros::Duration(0.1));
                tfl.lookupTransform("world",string("marker")+markerID,timeStamp,tfWorld2Marker);
                tfl.waitForTransform(targetName+"/base_footprint",targetName+"/odom",timeStamp,ros::Duration(0.1));
                tfl.lookupTransform(targetName+"/base_footprint",targetName+"/odom",timeStamp,tfMarker2Odom);
            }
            catch(tf::TransformException ex) { return; }
            
            // Save transform
            temp_quat = tfWorld2Marker.getRotation();
            Quaterniond qW2M = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
            temp_quat = tfMarker2Odom.getRotation();
            Quaterniond qM2O = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
            qWorld2Odom = qW2M*qM2O;
        }
        else
        {
            // use quaternion from image to ugv0 transform
            tf::Quaternion temp_quat = tfTarget2Im.getRotation();
            qTarget2Im = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
        }
        
        // Target velocities expressed in camera coordinates
        Vector3d vTc = qTarget2Im*vTt;
        
        // Undistort image coordinates. Returns normalized Euclidean coordinates
        double ptsArray[2] = {center->x,center->y};
        cv::Mat pts(1,1,CV_64FC2,ptsArray);
        cv::Mat undistPts;
        cv::undistortPoints(pts,undistPts,camMat,distCoeffs);
        Vector2d yMeas(undistPts.at<double>(0,0),undistPts.at<double>(0,1));
        
        // Observer velocities
        Vector3d b = vTc - vCc;
        Vector3d w = -wGCc; //Vector3d::Zero(); 
        
        // Convert to scalars to match notation in papers
        double b1 = b(0);           double b2 = b(1);           double b3 = b(2);
        double w1 = w(0);           double w2 = w(1);           double w3 = w(2);
        double y1hat = yhat(0);     double y2hat = yhat(1);     double y3hat = yhat(2);
        double y1 = yMeas(0);       double y2 = yMeas(1);
        double y1last = ylast(0);   double y2last = ylast(1);
        
        // Estimator
        double h1 = b1 - y1*b3;
        double h2 = b2 - y2*b3;
        double p1 = -y1*y2*w1 + (1+pow(y1,2))*w2 - y2*w3;
        double p2 = -(1+pow(y2,2))*w1 + y1*y2*w2 + y1*w3;
        
        double e1 = y1 - y1hat;
        double e2 = y2 - y2hat;
        
        double y1dot = (y1 - y1last)/delT;
        double y2dot = (y2 - y2last)/delT;
        ylast << yMeas;
        
        double y1hatDot = h1*y3hat + p1 + k1*e1;
        double y2hatDot = h2*y3hat + p1 + k2*e2;
        double y3hatDot = -b3*pow(y3hat,2) + (y1*w2 - y2*w1)*y3hat - k3*(pow(h1,2)+pow(h2,2))*y3hat + k3*h1*(y1dot-p1) + k3*h2*(y2dot-p2) + h1*e1 + h2*e2;
        
        Vector3d yhatDot;
        yhatDot << y1hatDot, y2hatDot, y3hatDot;
        yhat += yhatDot*delT;
        
        // Publish output
        publishOutput(y,yhat,timeStamp);
        
        if (!artificialSwitching)
        {
            // Restart watchdog timer for feature visibility check
            watchdogTimer.start();
        }
    }
    
    // Method for publishing data (estimate, ground truth, etc) and publishing Point message for visualization
    void publishOutput(Vector3d y, Vector3d yhat, ros::Time timeStamp)
    {
        // Extra signals
        Vector3d XYZ(y(0)/y(2),y(1)/y(2),1/y(2));
        Vector3d XYZhat(yhat(0)/yhat(2),yhat(1)/yhat(2),1/yhat(2));
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
        outMsg.delTon = delTon;             outMsg.delToff = delToff;
        outMsg.estimatorOn = estimatorOn;
        outMsg.usePredictor = usePredictor;
        outMsg.deadReckoning = deadReckoning;
        outMsg.normalizedKinematics = true;
        outMsg.artificialSwitching = artificialSwitching;
        outMsg.useVelocityMap = useVelocityMap;
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


};//End of class SubscribeAndPublish


int main(int argc, char** argv)
{
    ros::init(argc, argv, "switch_vis_exp_node");
    
    SubscribeAndPublish sap;
    
    ros::spin();
    return 0;
}

