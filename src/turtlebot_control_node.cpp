#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <tf/transform_listener.h>
#include <switch_vis_exp/MapVel.h>

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class turtlebot_control
{
    ros::NodeHandle nh;
    ros::ServiceServer service;
    ros::ServiceClient client;
    ros::ServiceClient otherBotPlan;
    ros::Timer controlLoop;
    ros::Publisher velPub;
    ros::Publisher desVelPub;
    ros::Publisher desBodyVelPub;
    tf::TransformListener tfl;
    
    // parameters
    double kw1;
    double kw2;
    double loopRate;
    std::string turtlebotName;
    std::string otherBotName;
    
    int fromNode;
    int toNode;
    bool streetMultiBot;
    
public:
    turtlebot_control()
    {
        // parameters
        ros::NodeHandle nhp("~");
        nhp.param<double>("kw1",kw1,1);
        nhp.param<double>("kw2",kw2,1);
        nhp.param<double>("loopRate",loopRate,10); //Hz
        nhp.param<bool>("streetMultiBot",streetMultiBot,false); //Hz
        turtlebotName = ros::names::clean(ros::this_node::getNamespace()).substr(1,std::string::npos);
        otherBotName = turtlebotName == "ugv0" ? "ugv1" : "ugv0";
        fromNode = -1;
        toNode = -1;
        
        // publishers
        velPub = nh.advertise<geometry_msgs::Twist>("cmd_vel_mux/input/navi",1);
        desVelPub = nh.advertise<geometry_msgs::Twist>("cmd_vel_des",1);
        desBodyVelPub = nh.advertise<geometry_msgs::Twist>("cmd_body_vel_des",1);
        
        // Advertise service for other bots to get path plan to avoid collision
        service = nh.advertiseService("plan", &turtlebot_control::plan,this);
        
        // get service handle
        client = nh.serviceClient<switch_vis_exp::MapVel>("/get_velocity");
        otherBotPlan = nh.serviceClient<switch_vis_exp::MapVel>("/" + otherBotName + "/plan");
        
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
        geometry_msgs::Pose poseMsg;
        poseMsg.position.x = position.getX();
        poseMsg.position.y = position.getY();
        poseMsg.position.z = position.getZ();
        poseMsg.orientation.x = orientation.getX();
        poseMsg.orientation.y = orientation.getY();
        poseMsg.orientation.z = orientation.getZ();
        poseMsg.orientation.w = orientation.getW();
        srv.request.pose.push_back(poseMsg);
        srv.request.fromNode.push_back(fromNode);
        srv.request.toNode.push_back(toNode);
        
        if (streetMultiBot)
        {
            bool goodNode = false;
            do
            {
                while(!otherBotPlan.call(srv)) { ros::Duration(0.001).sleep(); ros::spinOnce(); };
                int otherFromNode = srv.response.fromNode.at(0);
                int otherToNode = srv.response.toNode.at(0);
                while(!client.call(srv)) {};
                if ((srv.response.toNode.at(0) == otherToNode and srv.response.fromNode.at(0) != otherFromNode) or (srv.response.toNode.at(0) == otherFromNode and srv.response.fromNode.at(0) == otherToNode))
                {
                    goodNode = false;
                }
                else { goodNode = true; }
            } while(!goodNode);
        }
        else
        {
            while(!client.call(srv)) {};
        }
        
        // Get nodes
        fromNode = srv.response.fromNode.at(0);
        toNode = srv.response.toNode.at(0);
        
        // get velocity
        Eigen::Vector3d des_vel;
        des_vel << srv.response.twist.at(0).linear.x, srv.response.twist.at(0).linear.y, srv.response.twist.at(0).linear.z;
        
        // get orientation of turtlebot
        Eigen::Quaterniond quat(orientation.getW(), orientation.getX(), orientation.getY(), orientation.getZ());
        double heading = 2*std::asin(orientation.getZ());
        
        // rotate velocity into turtlebot body frame
        Eigen::Vector3d des_body_vel = quat.inverse()*des_vel;
        
        // non-holonomic controller
        Eigen::Vector3d v;
        v[0] = des_body_vel[0]; // - kminus*sgn(des_body_vel[0])*std::abs(w[2]);
        Eigen::Vector3d xDot = quat*v;
        srv.request.pose.at(0).position.x += xDot[0];
        srv.request.pose.at(0).position.y += xDot[1];
        srv.request.pose.at(0).position.z += xDot[2];
        srv.request.fromNode.at(0) = fromNode;
        srv.request.toNode.at(0) = toNode;
        
        if (streetMultiBot)
        {
            bool goodNode = false;
            do
            {
                while(!otherBotPlan.call(srv)) { ros::Duration(0.001).sleep(); ros::spinOnce(); };
                int otherFromNode = srv.response.fromNode.at(0);
                int otherToNode = srv.response.toNode.at(0);
                while(!client.call(srv)) {};
                if ((srv.response.toNode.at(0) == otherToNode and srv.response.fromNode.at(0) != otherFromNode) or (srv.response.toNode.at(0) == otherFromNode and srv.response.fromNode.at(0) == otherToNode))
                {
                    goodNode = false;
                }
                else { goodNode = true; }
            } while(!goodNode);
        }
        else
        {
            while(!client.call(srv)) {};
        }
        
        Eigen::Vector3d vNext;
        vNext << srv.response.twist.at(0).linear.x, srv.response.twist.at(0).linear.y, srv.response.twist.at(0).linear.z;
        fromNode = srv.response.fromNode.at(0); toNode = srv.response.toNode.at(0);
        Eigen::Vector3d xDir(1,0,0);
        //Eigen::Vector3d w = kw*xDir.cross(des_body_vel/des_body_vel.norm()); // angular velocity command
        Eigen::Vector3d crossTerm = xDir.cross(des_body_vel/des_body_vel.norm());
        if (xDir.dot(des_body_vel/des_body_vel.norm()) < 0) { crossTerm = Eigen::Vector3d(0,0,5*sgn(crossTerm(2))); }; 
        Eigen::Vector3d w = kw1*crossTerm + kw2*(((vNext/vNext.norm() - des_vel/des_vel.norm())*loopRate).norm())*((des_vel.cross(vNext)).normalized());
        
        // publish desired
        desVelPub.publish(srv.response.twist.at(0));
        geometry_msgs::Twist bodyTwist;
        bodyTwist.linear.x = des_body_vel[0];
        bodyTwist.linear.y = des_body_vel[1];
        bodyTwist.linear.z = des_body_vel[2];
        desBodyVelPub.publish(bodyTwist);
        
        // publish to turtlebot
        geometry_msgs::Twist twistMsg;
        twistMsg.linear.x = std::max(v[0],0.0);
        twistMsg.angular.z = w[2];
        velPub.publish(twistMsg);
    }
    
    bool plan(switch_vis_exp::MapVel::Request &req,switch_vis_exp::MapVel::Response &resp)
    {
        resp.fromNode.push_back(fromNode);
        resp.toNode.push_back(toNode);
        return true;
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
    
    ros::AsyncSpinner spinner(4); // Use 4 threads
    spinner.start();
    ros::waitForShutdown();
    return 0;
}
