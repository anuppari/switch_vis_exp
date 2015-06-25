#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Time.h>
#include <nav_msgs/Odometry.h>
#include <switch_vis_exp/Velocity.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>

double vel[3];
double acc[3];
double w[3];
bool velReady = false;
bool accReady = false;
double dataTime[2];
double last_vel[3];

// need a class in order publish in the callback
class SubscribeAndPublish
{
    ros::NodeHandle n;
    ros::Subscriber sub;
    ros::Publisher filterPub;
    switch_vis_exp::Velocity vel_msg;
    double RCvel;
    double RCacc;
public:
    SubscribeAndPublish()
    {
        // Filter input and output
        filterPub = n.advertise<switch_vis_exp::Velocity>("filter",100);
        sub = n.subscribe("odom", 100, &SubscribeAndPublish::filterCb,this);
        
        // Filter parameters
        n.param<double>(ros::this_node::getName()+"/vel_filter_time_const", RCvel, 0.05);
        n.param<double>(ros::this_node::getName()+"/acc_filter_time_const", RCacc, 0.1);
    }

    void filterCb(const nav_msgs::OdometryConstPtr& msg)
    {
        if(!velReady) { //dont publish on first call
            dataTime[0] = double(msg->header.stamp.sec) + double(msg->header.stamp.nsec)/double(1e9);
            vel[0] = msg->twist.twist.linear.x;
            vel[1] = msg->twist.twist.linear.y;
            vel[2] = msg->twist.twist.linear.z;
            last_vel[0] = msg->twist.twist.linear.x;
            last_vel[1] = msg->twist.twist.linear.y;
            last_vel[2] = msg->twist.twist.linear.z;
            w[0] = msg->twist.twist.angular.x;
            w[1] = msg->twist.twist.angular.y;
            w[2] = msg->twist.twist.angular.z;
            velReady = true;
        }
        else {
            dataTime[1] = double(msg->header.stamp.sec) + double(msg->header.stamp.nsec)/double(1e9);
            double dt = dataTime[1] - dataTime[0];
            double alphaVel = dt/(RCvel+dt);
            vel[0] = alphaVel*msg->twist.twist.linear.x+(1-alphaVel)*vel[0];
            vel[1] = alphaVel*msg->twist.twist.linear.y+(1-alphaVel)*vel[1];
            vel[2] = alphaVel*msg->twist.twist.linear.z+(1-alphaVel)*vel[2];
            w[0] = alphaVel*msg->twist.twist.angular.x+(1-alphaVel)*w[0];
            w[1] = alphaVel*msg->twist.twist.angular.y+(1-alphaVel)*w[1];
            w[2] = alphaVel*msg->twist.twist.angular.z+(1-alphaVel)*w[2];
            
            double raw_acc[3];
            raw_acc[0] = (vel[0]-last_vel[0])/dt;
            raw_acc[1] = (vel[1]-last_vel[1])/dt;
            raw_acc[2] = (vel[2]-last_vel[2])/dt;
            
            if(!accReady) { // dont publish until enough data is recieved
                acc[0] = raw_acc[0];
                acc[1] = raw_acc[1];
                acc[2] = raw_acc[2];
                accReady = true;
            }
            else {
                double alphaAcc = dt/(RCacc+dt);
                acc[0] = alphaAcc*raw_acc[0]+(1-alphaAcc)*acc[0];
                acc[1] = alphaAcc*raw_acc[1]+(1-alphaAcc)*acc[1];
                acc[2] = alphaAcc*raw_acc[2]+(1-alphaAcc)*acc[2];
                
                // Publish filtered data
                vel_msg.header.stamp = msg->header.stamp;
                vel_msg.vel[0] = vel[0];
                vel_msg.vel[1] = vel[1];
                vel_msg.vel[2] = vel[2];
                vel_msg.acc[0] = acc[0];
                vel_msg.acc[1] = acc[1];
                vel_msg.acc[2] = acc[2];
                vel_msg.w[0] = w[0];
                vel_msg.w[1] = w[1];
                vel_msg.w[2] = w[2];

                filterPub.publish(vel_msg);
            }
            last_vel[0] = vel[0];
            last_vel[1] = vel[1];
            last_vel[2] = vel[2];
            dataTime[0] = dataTime[1];
        }
    }
};//End of class SubscribeAndPublish


int main(int argc, char** argv)
{
    ros::init(argc, argv, "filter");
    
    SubscribeAndPublish sap;
    
    ros::spin();
    return 0;
}
