#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <switch_vis_exp/MapVel.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

class turtlebot_control
{
    ros::NodeHandle nh;
    ros::ServiceClient client;
    ros::Subscriber poseSub;
    ros::Publisher velPub;
    
    // parameters
    double kw;
    
public:
    turtlebot_control()
    {
        // parameters
        nh.param<double>("kw",kw,1);
        
        // publishers
        velPub = nh.advertise<geometry_msgs::Twist>("cmd_vel_mux/input/navi",1);
        
        // get service handle
        client = nh.serviceClient<switch_vis_exp::MapVel>("get_velocity");
        
        // subscribers
        poseSub = nh.subscribe("pose",10,&turtlebot_control::poseCB,this);
    }
    
    void poseCB(const geometry_msgs::PoseStampedConstPtr& pose)
    {
        switch_vis_exp::MapVel srv;
        srv.request.pose = pose->pose;
        if (client.call(srv))
        {
            // get velocity
            Eigen::Vector3d vel;
            vel << srv.response.twist.linear.x, srv.response.twist.linear.y, srv.response.twist.linear.z;
            
            // get orientation of turtlebot
            Eigen::Quaterniond quat(pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z, pose->pose.orientation.w);
            
            // rotate velocity into turtlebot body frame
            Eigen::Vector3d body_vel = quat.inverse()*vel;
            
            // non-holonomic controller
            Eigen::Vector3d xDir(1,0,0);
            Eigen::Vector3d w = kw*xDir.cross(body_vel); // angular velocity command
            
            // publish to turtlebot
            geometry_msgs::Twist twistMsg;
            twistMsg.linear.x = body_vel[0];
            twistMsg.angular.z = w[2];
            velPub.publish(twistMsg);
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "turtlebot_control_node");
    
    turtlebot_control tc;
    
    ros::spin();
    return 0;
}
