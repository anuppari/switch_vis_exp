#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_listener.h>
#include <switch_vis_exp/MapVel.h>

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class turtlebot_control
{
    ros::NodeHandle nh;
    ros::ServiceClient client;
    ros::Timer controlLoop;
    ros::Publisher velPub;
    ros::Publisher desVelPub;
    tf::TransformListener tfl;
    
    // parameters
    double kw;
    double kminus;
    double loopRate;
    std::string turtlebotName;
    
public:
    turtlebot_control()
    {
        // parameters
        ros::NodeHandle nhp("~");
        nhp.param<double>("kw",kw,1);
        nhp.param<double>("kminus",kminus,0.2);
        nhp.param<double>("loopRate",loopRate,10); //Hz
        turtlebotName = ros::names::clean(ros::this_node::getNamespace()).substr(1,std::string::npos);
        
        // publishers
        velPub = nh.advertise<geometry_msgs::Twist>("cmd_vel_mux/input/navi",1);
        desVelPub = nh.advertise<geometry_msgs::Twist>("cmd_vel_des",1);
        
        // get service handle
        client = nh.serviceClient<switch_vis_exp::MapVel>("/get_velocity");
        
        // subscribers
        controlLoop = nh.createTimer(ros::Duration(1.0/loopRate),&turtlebot_control::controlCB,this);
    }
    
    void controlCB(const ros::TimerEvent&)
    {
        // Get turtlebot pose
        tf::StampedTransform transform;
        try
        {
            ros::Time timeStamp = ros::Time::now();
            tfl.waitForTransform("world",turtlebotName,timeStamp,ros::Duration(0.1));
            tfl.lookupTransform("world",turtlebotName,timeStamp,transform);
        }
        catch(tf::TransformException ex)
        {
            return;
        }
        
        // Construct service call
        switch_vis_exp::MapVel srv;
        tf::Vector3 position = transform.getOrigin();
        tf::Quaternion orientation = transform.getRotation();
        srv.request.pose.position.x = position.getX();
        srv.request.pose.position.y = position.getY();
        srv.request.pose.position.z = position.getZ();
        srv.request.pose.orientation.x = orientation.getX();
        srv.request.pose.orientation.y = orientation.getY();
        srv.request.pose.orientation.z = orientation.getZ();
        srv.request.pose.orientation.w = orientation.getW();
        
        // Call service
        if (client.call(srv))
        {
            // get velocity
            Eigen::Vector3d des_vel;
            des_vel << srv.response.twist.linear.x, srv.response.twist.linear.y, srv.response.twist.linear.z;
            
            // get orientation of turtlebot
            Eigen::Quaterniond quat(orientation.getW(), orientation.getX(), orientation.getY(), orientation.getZ());
            
            // rotate velocity into turtlebot body frame
            Eigen::Vector3d des_body_vel = quat.inverse()*des_vel;
            
            // non-holonomic controller
            Eigen::Vector3d xDir(1,0,0);
            Eigen::Vector3d w = kw*xDir.cross(des_body_vel/des_body_vel.norm()); // angular velocity command
            
            // publish desired
            desVelPub.publish(srv.response.twist);
            
            // publish to turtlebot
            geometry_msgs::Twist twistMsg;
            twistMsg.linear.x = des_body_vel[0] - kminus*sgn(des_body_vel[0])*std::abs(w[2]);
            twistMsg.angular.z = w[2];
            velPub.publish(twistMsg);
        }
    }
    
    int sgn(double val)
    {
        return ((val > 0) - (val < 0));
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "turtlebot_control_node");
    
    turtlebot_control tc;
    
    ros::spin();
    return 0;
}