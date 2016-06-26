#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/Empty.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>

class bebop_control
{
    // ROS stuff
    ros::NodeHandle nh;
    ros::Publisher takeoffPub;
    ros::Publisher landPub;
    ros::Publisher resetPub;
    ros::Publisher velCmdPub;
    ros::Subscriber joySub;
    ros::Subscriber poseSub;
    ros::Subscriber bebopVelSub;
    ros::Subscriber targetVelSub;
    tf::TransformListener tfl;
    
    // Parameters
    //tf::Vector3 boundaryOffsetBottomLeft;
    //tf::Vector3 boundaryOffsetTopRight;
    //tf::Matrix3x3 kpLin;
    //tf::Matrix3x3 kdLin;
    //tf::Matrix3x3 kffLin;
    //double kpAng;
    //double kdAng;
    //double kffAng;
    bool lazy;
    std::string target;
    double radius, kp, kw, kpd;
    
    // States
    bool autonomy;
    bool mocapOn;
    bool bebopVelOn;
    bool targetVelOn;
    double lastVelTime;
    tf::Vector3 joyLinVel;
    tf::Vector3 joyAngVel;
    tf::Vector3 bebopLinVel;
    tf::Vector3 bebopAngVel;
    tf::Vector3 targetLinVel;
    tf::Vector3 targetAngVel;
    //tf::Vector3 lastLinError;
    //tf::Vector3 lastAngError;
    
public:
    bebop_control()
    {
        // Parameters
        ros::NodeHandle nhp("~");
        nhp.param<bool>("lazy", lazy, false);
        nhp.param<std::string>("target", target, "ugv0");
        nhp.param<double>("radius", radius, 1.0);
        nhp.param<double>("kp", kp, 0.5);
        nhp.param<double>("kw", kw, 1.0);
        nhp.param<double>("kpd", kpd, 0.5);
        
        // Publishers
        velCmdPub = nh.advertise<geometry_msgs::Twist>("/bebop/cmd_vel",1);
        takeoffPub = nh.advertise<std_msgs::Empty>("bebop/takeoff",1);
        landPub = nh.advertise<std_msgs::Empty>("bebop/land",1);
        resetPub = nh.advertise<std_msgs::Empty>("bebop/reset",1);
        
        // Initialize states
        autonomy = false;
        mocapOn = false;
        bebopVelOn = false;
        targetVelOn = false;
            
        // Subscribers
        joySub = nh.subscribe("joy",1,&bebop_control::joyCB,this);
        poseSub = nh.subscribe("bebop/pose",1,&bebop_control::poseCB,this);
        bebopVelSub = nh.subscribe("bebop/vel",1,&bebop_control::bebopVelCB,this);
        targetVelSub = nh.subscribe("ugv0/vel",1,&bebop_control::targetVelCB,this);
        
        // Warning message
        while (!(ros::isShuttingDown()) and (!mocapOn or !bebopVelOn or !targetVelOn))
        {
            if (!mocapOn) { std::cout << "BEBOP POSE NOT PUBLISHED! THE WALL IS DOWN!\nIs the mocap on?" << std::endl; }
            if (!bebopVelOn) { std::cout << "BEBOP VELOCITY NOT PUBLISHED! => Bad tracking performance\nIs velocity filter running?" << std::endl; }
            if (!targetVelOn) { std::cout << "TARGET VELOCITY NOT PUBLISHED! => Bad tracking performance\nIs velocity filter running?" << std::endl; }
            ros::spinOnce();
            ros::Duration(0.1).sleep();
        }
    }
    
    void poseCB(const geometry_msgs::PoseStamped& poseMsg)
    {
        if (!mocapOn)
        {
            std::cout << "Bebop pose is publishing. THE WALL IS UP!" << std::endl;
            mocapOn = true;
        }
        
        tf::Vector3 desBodyLinVel, desBodyAngVel;
        if (autonomy)
        {
            // Bebop pose
            tf::Transform bebopPoseTF(tf::Quaternion(poseMsg.pose.orientation.x,poseMsg.pose.orientation.y,poseMsg.pose.orientation.z,poseMsg.pose.orientation.w),
                                        tf::Vector3(poseMsg.pose.position.x,poseMsg.pose.position.y,poseMsg.pose.position.z));
            
            // Target pose
            tf::StampedTransform targetPoseTF;
            try { tfl.lookupTransform("world",target,ros::Time(0),targetPoseTF); }
            catch(tf::TransformException ex) {}
            
            // Desired pose
            tf::Vector3 desPos, desForward;
            if (lazy)
            {
                tf::Vector3 unitRelPos = (bebopPoseTF.getOrigin() - targetPoseTF.getOrigin());
                unitRelPos.setZ(0);
                unitRelPos.normalize();
                desPos = targetPoseTF.getOrigin() + radius*unitRelPos;
                desPos.setZ(desPos.getZ() + 1);
                desForward = -unitRelPos;
            }
            else
            {
                desPos = targetPoseTF.getOrigin();
                desPos.setZ(desPos.getZ() + 1);
                desForward = targetPoseTF.getBasis()*tf::Vector3(1,0,0);
            }
            
            // Desired velocity
            tf::Vector3 desLinVel = kp*(desPos - bebopPoseTF.getOrigin()) + kpd*(targetLinVel - bebopLinVel);
            tf::Vector3 desAngVel = kw*((bebopPoseTF.getBasis()*tf::Vector3(1,0,0)).cross(desForward));
            desBodyLinVel = bebopPoseTF.getBasis().inverse()*desLinVel + tf::Vector3(0,0.4*joyAngVel.getZ(),0);
            desBodyAngVel = bebopPoseTF.getBasis().inverse()*desAngVel + tf::Vector3(0,0,-0.5*joyAngVel.getZ());
        }
        else
        {
            desBodyLinVel = joyLinVel;
            desBodyAngVel = joyAngVel;
        }
        
        // Publish
        geometry_msgs::Twist twistMsg;
        twistMsg.linear.x = desBodyLinVel.getX();
        twistMsg.linear.y = desBodyLinVel.getY();
        twistMsg.linear.z = desBodyLinVel.getZ();
        twistMsg.angular.x = desBodyAngVel.getX();
        twistMsg.angular.y = desBodyAngVel.getY();
        twistMsg.angular.z = -1*desBodyAngVel.getZ();
        velCmdPub.publish(twistMsg);
    }
    
    void bebopVelCB(const geometry_msgs::TwistStampedConstPtr& twist)
    {
        if (!bebopVelOn)
        {
            std::cout << "Bebop velocity is publishing." << std::endl;
            bebopVelOn = true;
        }
        
        // Measurements
        bebopLinVel =  tf::Vector3(twist->twist.linear.x,twist->twist.linear.y,twist->twist.linear.z);
        bebopAngVel =  tf::Vector3(twist->twist.angular.x,twist->twist.angular.y,twist->twist.angular.z);
    }
    
    void targetVelCB(const geometry_msgs::TwistStampedConstPtr& twist)
    {
        if (!targetVelOn)
        {
            std::cout << "Target velocity is publishing." << std::endl;
            targetVelOn = true;
        }
        
        // Measurements
        targetLinVel =  tf::Vector3(twist->twist.linear.x,twist->twist.linear.y,twist->twist.linear.z);
        targetAngVel =  tf::Vector3(twist->twist.angular.x,twist->twist.angular.y,twist->twist.angular.z);
    }
    
    void joyCB(const sensor_msgs::JoyConstPtr& joyMsg)
    {
        autonomy = false;
        if (joyMsg->buttons[1]) // b - reset
        {
            resetPub.publish(std_msgs::Empty());
        }
        else if (joyMsg->buttons[0]) // a - land
        {
            landPub.publish(std_msgs::Empty());
        }
        else if (joyMsg->buttons[3]) // y - takeoff
        {
            takeoffPub.publish(std_msgs::Empty());
        }
        else
        {
            joyLinVel = tf::Vector3(joy_deadband(joyMsg->axes[4]), joy_deadband(joyMsg->axes[3]), joy_deadband(joyMsg->axes[1]));
            joyAngVel = tf::Vector3(0,0, joy_deadband(joyMsg->axes[0]));
            
            if (joyMsg->buttons[4]) { autonomy = true; } // LB - autonomy
        }
    }
    
    double joy_deadband(double input_value)
    {
        double filtered_value = 0;
        if (std::abs(input_value) > 0.15)
        {
            filtered_value = input_value;
        }
        return filtered_value;
    }
    
}; // end bebop_control

int main(int argc, char** argv)
{
    ros::init(argc, argv, "bebop_control");
    
    bebop_control obj;
    
    ros::spin();
    return 0;
}
