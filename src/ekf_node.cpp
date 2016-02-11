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

#include <Eigen/Dense> // Defines MatrixXd, Matrix3d, Vector3d
#include <Eigen/Geometry> // Defines Quaterniond
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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
    ros::ServiceClient targetVelClient;
    tf::TransformListener tfl;
    ros::Timer watchdogTimer;
    ros::Timer switchingTimer;
    
    // parameters
    cv::Mat camMat;
    cv::Mat distCoeffs;
    bool gotCamParam;
    bool deadReckoning;
    bool useVelocityMap;
    bool artificialSwitching;
    bool normalizedKinematics;
    double visibilityTimeout;
    string cameraName;
    string targetName;
    string markerID;
    double q;   // EKF process noise scale
    double r;   // EKF measurement noise scale
    double delTon;
    double delToff;
    
    //states
    Vector3d xhat;
    Matrix3d P;         // Estimated covariance
    double lastVelTime;
    double lastImageTime;
    bool estimatorOn;
    Quaterniond qWorld2Odom; // Rotation for initializing dead reckoning
    Vector3d vTt;   // target linear velocity w.r.t. ground, expressed in target coordinate system
    Vector3d vCc;   // camera linear velocity w.r.t. ground, expressed in camera coordinate system
    Vector3d wGCc;  // camera angular velocity w.r.t. ground, expressed in camera coordinate system
    Matrix3d Q;     // EKF process noise covariance
    Matrix2d R;     // EKF measurement noise covariance
    Matrix<double,2,3> H;   // EKF measurement model
public:
    EKF()
    {
        // Get Parameters
        ros::NodeHandle nhp("~");
        nhp.param<bool>("normalizedKinematics", normalizedKinematics, true);
        nhp.param<bool>("deadReckoning", deadReckoning, true);
        nhp.param<bool>("artificialSwitching", artificialSwitching, false);
        nhp.param<bool>("useVelocityMap", useVelocityMap, false);
        nhp.param<double>("visibilityTimeout", visibilityTimeout, 0.2);
        nhp.param<string>("cameraName",cameraName,"camera");
        nhp.param<string>("targetName",targetName,"ugv0");
        nhp.param<string>("markerID",markerID,"100");
        nhp.param<double>("delTon",delTon,4.0);
        nhp.param<double>("delToff",delToff,1.0);
        nhp.param<double>("q",q,10); // process noise
        nhp.param<double>("r",r,0.001); // measurement noise
        
        // Initial conditions
        double X0,Y0,Z0;
        nhp.param<double>("X0",X0,0);
        nhp.param<double>("Y0",Y0,0);
        nhp.param<double>("Z0",Z0,10);
        
        lastVelTime = ros::Time::now().toSec();
        lastImageTime = lastVelTime;
        estimatorOn = true;
        gotCamParam = false;
        
        // Initialize EKF matrices
        Q = q*Matrix3d::Identity();
        R = r*Matrix2d::Identity();
        
        // Get camera parameters
        cout << cameraName+"/camera_info" << endl;
        camInfoSub = nh.subscribe(cameraName+"/camera_info",1,&EKF::camInfoCB,this);
        ROS_DEBUG("Waiting for camera parameters on topic ...");
        do {
            ros::spinOnce();
            ros::Duration(0.1).sleep();
        } while (!(ros::isShuttingDown()) and !gotCamParam);
        ROS_DEBUG("Got camera parameters");
        
        // Initialize states
        if (normalizedKinematics)
        {
            xhat << X0/Z0,Y0/Z0,1/Z0;
            Matrix3d eigCamMat;
            cv::cv2eigen(camMat,eigCamMat);
            P << 1/eigCamMat(0,0), 0, 0,
                  0, 1/eigCamMat(1,1), 0,
                  0, 0, 20;
        }
        else
        {
            xhat << X0,Y0,Z0;
            P << 10,0,0,
                 0,10,0,
                 0,0,20;
        }
        
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
            targetVelSub = nh.subscribe(targetName+"/odom",1,&EKF::targetVelCBdeadReckoning,this);
        }
        else
        {
            targetVelSub = nh.subscribe(targetName+"/body_vel",1,&EKF::targetVelCBmocap,this);
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
            watchdogTimer.stop(); // Dont start watchdog until feature first visible
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
        camMat.convertTo(camMat,CV_64FC1);
        cam_model.distortionCoeffs().convertTo(distCoeffs,CV_64FC1);
        
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
            // update lastImageTime for reasonable delT when target is back in view
            lastImageTime = timeNow;
            
            // Object translation w.r.t. image frame, for ground truth
            tf::StampedTransform tfTarget2Im;
            tfl.waitForTransform("image",targetName,timeStamp,ros::Duration(0.1));
            tfl.lookupTransform("image",targetName,timeStamp,tfTarget2Im);
            Vector3d trans(tfTarget2Im.getOrigin().getX(), tfTarget2Im.getOrigin().getY(), tfTarget2Im.getOrigin().getZ());
            
            // Ground truth
            Vector3d x;
            if (normalizedKinematics)
            {
                x << trans.segment<2>(0)/trans(2),1/trans(2);
            }
            else
            {
                x = trans;
            }
            
            // Target velocity in camera frame
            Vector3d vTc;
            
            // Get expected target velocity from velocity map
            if (useVelocityMap)
            {
                // service msg handle
                switch_vis_exp::MapVel srv;
                
                // transform state estimate to global frame
                tf::StampedTransform tfIm2W;
                tfl.waitForTransform("image","world",timeStamp,ros::Duration(0.1));
                tfl.lookupTransform("image","world",timeStamp,tfIm2W);
                Quaterniond qIm2W(tfIm2W.getRotation().getW(),tfIm2W.getRotation().getX(),tfIm2W.getRotation().getY(),tfIm2W.getRotation().getZ());
                Vector3d xhatWorld;
                if (normalizedKinematics)
                {
                    xhatWorld = qIm2W*Vector3d(xhat(0)/xhat(2), xhat(1)/xhat(2), 1/xhat(2));
                }
                else
                {
                    xhatWorld << qIm2W*xhat;
                }
                
                // Construct request
                srv.request.pose.position.x = xhatWorld(0);
                srv.request.pose.position.y = xhatWorld(1);
                srv.request.pose.position.z = xhatWorld(2);
                
                // Call and get response
                if (targetVelClient.call(srv))
                {
                    // get expected target velocity, expressed in world coordinates
                    Vector3d vTw(srv.response.twist.linear.x, srv.response.twist.linear.y, srv.response.twist.linear.z);
                    
                    // rotate velocity into image coordinate frame
                    vTc = qIm2W.inverse()*vTw;
                }
                else
                {
                    return;
                }
            }
            else
            { // Get target velocities from communication
                
                // determine qTarget2Im to rotate target velocities into camera frame
                Quaterniond qTarget2Im;
                if (deadReckoning)
                {
                    tf::StampedTransform tfImage2World;
                    tf::StampedTransform tfOdom2Marker;
                    tfl.waitForTransform("image","world",timeStamp,ros::Duration(0.1));
                    tfl.lookupTransform("image","world",timeStamp,tfImage2World);
                    tfl.waitForTransform(targetName+"/odom",targetName+"/base_footprint",timeStamp,ros::Duration(0.1));
                    tfl.lookupTransform(targetName+"/odom",targetName+"/base_footprint",timeStamp,tfOdom2Marker);
                    
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
            
            // EKF prediction step
            ekf_predictor(vTc, delT);
            
            // Publish output
            publishOutput(x,xhat,timeStamp);
            
        }
    }
    
    // Calculate linearization of dynamics (F) for EKF
    Matrix3d calculate_F(Vector3d xhat_,Vector3d vCc_,Vector3d vTc_,Vector3d wGCc_)
    {
        double vc1 = vCc_(0);        double vc2 = vCc_(1);        double vc3 = vCc_(2);
        double vq1 = vTc_(0);        double vq2 = vTc_(1);        double vq3 = vTc_(2);
        double w1 = wGCc_(0);        double w2 = wGCc_(1);        double w3 = wGCc_(2);
        double x1 = xhat_(0);        double x2 = xhat_(1);        double x3 = xhat_(2);
        
        Matrix3d F;
        if (normalizedKinematics)
        {
            F << -2*w2*x1+w1*x2+(vc3-vq3)*x3,   w3+w1*x1,                       vq1-vc1-(vq3-vc3)*x1,
                 -w3-w2*x2,                     -w2*x1+2*w1*x2+(vc3-vq3)*x3,    vq2-vc2-(vq3-vc3)*x2,
                 -w2*x3,                        w1*x3,                          2*(vc3-vq3)*x3-(w2*x1-w1*x2);
        }
        else
        {
            F << 0, w3, -1*w2,
                -w3, 0, w1,
                w2, -w1, 0;
        }
        return F;
    }
    
    // Calculate linearization of measurements (H) for EKF
    Matrix<double,2,3> calculate_H(Vector3d xhat_)
    {
        Matrix<double,2,3> H;
        if (normalizedKinematics)
        {
            H << 1,0,0,
                 0,1,0;
        }
        else
        {
            double X = xhat_(0);
            double Y = xhat_(1);
            double Z = xhat_(2);
            
            double fx = camMat.at<double>(0,0);
            double fy = camMat.at<double>(1,1);
            double cx = camMat.at<double>(0,2);
            double cy = camMat.at<double>(1,2);
            
            H << fx/Z, 0, -1*fx*X/pow(Z,2),
                 0, fy/Z, -1*fy*Y/pow(Z,2);
        }
        return H;
    }
    
    void ekf_predictor(Vector3d vTc, double delT)
    {
        // Convert to scalars to match notation in papers
        double vc1 = vCc(0);        double vc2 = vCc(1);        double vc3 = vCc(2);
        double vq1 = vTc(0);        double vq2 = vTc(1);        double vq3 = vTc(2);
        double w1 = wGCc(0);        double w2 = wGCc(1);        double w3 = wGCc(2);
        double x1hat = xhat(0);     double x2hat = xhat(1);     double x3hat = xhat(2);
        
        // Predictor
        double x1hatDot, x2hatDot, x3hatDot;
        if (normalizedKinematics)
        {
            double Omega1 = w3*x2hat - w2 - w2*pow(x1hat,2) + w1*x1hat*x2hat;
            double Omega2 = w1 - w3*x1hat - w2*x1hat*x2hat + w1*pow(x2hat,2);
            double xi1 = (vc3*x1hat - vc1)*x3hat;
            double xi2 = (vc3*x2hat - vc2)*x3hat;
            
            x1hatDot = Omega1 + xi1 + vq1*x3hat - x1hat*vq3*x3hat;
            x2hatDot = Omega2 + xi2 + vq2*x3hat - x2hat*vq3*x3hat;
            x3hatDot = vc3*pow(x3hat,2) - (w2*x1hat - w1*x2hat)*x3hat - vq3*pow(x3hat,2);
        }
        else
        {
            x1hatDot = vq1 - vc1 + w3*x2hat - w2*x3hat;
            x2hatDot = vq2 - vc2 + w1*x3hat - w3*x1hat;
            x3hatDot = vq3 - vc3 + w2*x1hat - w1*x2hat;
        }
        
        // Predict Covariance
        Matrix3d F = calculate_F(xhat,vCc,vTc,wGCc);
        MatrixXd Pdot = F*P+P*F.transpose() + Q;
        
        // Update states and covariance
        Vector3d xhatDot;
        xhatDot << x1hatDot, x2hatDot, x3hatDot;
        xhat += xhatDot*delT;
        P += Pdot*delT;
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
        
        // Object translation w.r.t. image frame, for ground truth
        tf::StampedTransform tfTarget2Im;
        tfl.waitForTransform("image",targetName,timeStamp,ros::Duration(0.1));
        tfl.lookupTransform("image",targetName,timeStamp,tfTarget2Im);
        Vector3d trans(tfTarget2Im.getOrigin().getX(), tfTarget2Im.getOrigin().getY(), tfTarget2Im.getOrigin().getZ());
        
        // Ground truth
        Vector3d x;
        if (normalizedKinematics)
        {
            x << trans(0)/trans(2), trans(1)/trans(2), 1/trans(2);
        }
        else
        {
            x = trans;
        }
        
        // Target to camera frame transform, for rotating target velocity into camera frame
        Quaterniond qTarget2Im;
        if (deadReckoning)
        {
            tfl.waitForTransform("image",string("marker")+markerID,timeStamp,ros::Duration(0.1));
            tfl.lookupTransform("image",string("marker")+markerID,timeStamp,tfTarget2Im);
            tf::Quaternion temp_quat = tfTarget2Im.getRotation();
            qTarget2Im = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
            try
            {
                // Additional transforms for dead reckoning when target not visible
                tf::StampedTransform tfWorld2Marker;
                tf::StampedTransform tfMarker2Odom;
                tfl.waitForTransform("world",string("marker")+markerID,timeStamp,ros::Duration(0.1));
                tfl.lookupTransform("world",string("marker")+markerID,timeStamp,tfWorld2Marker);
                tfl.waitForTransform(targetName+"/base_footprint",targetName+"/odom",timeStamp,ros::Duration(0.1));
                tfl.lookupTransform(targetName+"/base_footprint",targetName+"/odom",timeStamp,tfMarker2Odom);
                
                // Save transform
                tf::Quaternion temp_quat = tfWorld2Marker.getRotation();
                Quaterniond qW2M = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
                temp_quat = tfMarker2Odom.getRotation();
                Quaterniond qM2O = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
                qWorld2Odom = qW2M*qM2O;
            }
            catch (tf::TransformException e)
            {
                return;
            }
        }
        else
        {
            // use quaternion from image to ugv0 transform
            tf::Quaternion temp_quat = tfTarget2Im.getRotation();
            qTarget2Im = Quaterniond(temp_quat.getW(),temp_quat.getX(),temp_quat.getY(),temp_quat.getZ());
        }
        
        // Transform target velocity into camera frame
        Vector3d vTc = qTarget2Im*vTt;
        
        // EKF prediction step
        ekf_predictor(vTc, delT);
        
        // Measurements. Undistort image coordinates.
        double ptsArray[2] = {center->x,center->y};
        cv::Mat pts(1,1,CV_64FC2,ptsArray);
        cv::Mat undistPts;
        Vector2d y;
        if (normalizedKinematics)
        {
            cv::undistortPoints(pts,undistPts,camMat,distCoeffs); // Returns normalized Euclidean coordinates
            y << undistPts.at<double>(0,0),undistPts.at<double>(0,1);
        }
        else
        {
            cv::undistortPoints(pts,undistPts,camMat,distCoeffs,cv::noArray(),camMat); // Returns undistorted pixel coordinates
            y << undistPts.at<double>(0,0),undistPts.at<double>(0,1);
        }
        
        // EKF update
        Matrix<double,2,3> H = calculate_H(xhat);
        Matrix<double,3,2> K = P*H.transpose()*(H*P*H.transpose()+R).inverse();
        Vector2d yhat = calculate_yhat(xhat);
        xhat += K*(y-yhat);
        P = (Matrix3d::Identity()-K*H)*P;
        
        // Publish output
        publishOutput(x,xhat,timeStamp);
        
        if (!artificialSwitching)
        {
            // Restart watchdog timer for feature visibility check
            watchdogTimer.start();
        }
    }
    
    // Calculate expected measurements
    Vector2d calculate_yhat(Vector3d xhat)
    {
        Vector2d yhat;
        
        if (normalizedKinematics)
        {
            yhat = xhat.head<2>();
        }
        else
        {
            double fx = camMat.at<double>(0,0);
            double fy = camMat.at<double>(1,1);
            double cx = camMat.at<double>(0,2);
            double cy = camMat.at<double>(1,2);
            
            yhat << fx*xhat(0)/xhat(2) + cx, fy*xhat(1)/xhat(2) + cy;
        }
        
        return yhat;
    }
    
    // Method for publishing data (estimate, ground truth, etc) and publishing Point message for visualization
    void publishOutput(Vector3d y, Vector3d yhat, ros::Time timeStamp)
    {
        // Extra signals
        Vector3d XYZ;
        Vector3d XYZhat;
        Vector3d error;
        Vector3d XYZerror;
        if (normalizedKinematics)
        {
            XYZ << y(0)/y(2), y(1)/y(2), 1/y(2);
            XYZhat << yhat(0)/yhat(2), yhat(1)/yhat(2), 1/yhat(2);
            error = y - yhat;
            XYZerror = XYZ - XYZhat;
        }
        else
        {
            XYZ = y;
            XYZhat = yhat;
            error << y(0)/y(2) - yhat(0)/yhat(2), y(1)/y(2) - yhat(1)/yhat(2), 1/y(2) - 1/yhat(2);
            XYZerror = XYZ - XYZhat;
        }
        
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
        outMsg.usePredictor = true;
        outMsg.deadReckoning = deadReckoning;
        outMsg.normalizedKinematics = normalizedKinematics;
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

};//End of class EKF


int main(int argc, char** argv)
{
    ros::init(argc, argv, "ekf_node");
    
    EKF obj;
    
    ros::spin();
    return 0;
}

