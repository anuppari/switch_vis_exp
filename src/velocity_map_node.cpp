#include <ros/ros.h>
#include <switch_vis_exp/MapVel.h>
#include <geometry_msgs/Twist.h>

class velocity_map
{
    ros::NodeHandle nh;
    ros::ServiceServer service;
    
    // Parameters
    double a;       // length of track
    double b;       // width of track
    double x0;      // track center
    double y0;      // track center
    double k1;      // overall velocity gain
    double k2;      // weighting between getting back to track and moving in direction of track
    int n;          // sharpness and squareness of track
    
public:
    velocity_map()
    {
        // Get parameters
        ros::NodeHandle nhp("~");
        nhp.param<double>("a",a,2);
        nhp.param<double>("b",b,2);
        nhp.param<double>("x0",x0,0);
        nhp.param<double>("y0",y0,0);
        nhp.param<double>("k1",k1,0.5);
        nhp.param<double>("k2",k2,0.1);
        nhp.param<int>("n",n,4);
        
        // Start service
        service = nh.advertiseService("get_velocity", &velocity_map::get_velocity,this);
    }

    bool get_velocity(switch_vis_exp::MapVel::Request &req,switch_vis_exp::MapVel::Response &resp)
    {
        for (int i = 0; i < req.pose.size(); i++)
        {
            // point on map
            double x = req.pose.at(i).position.x;
            double y = req.pose.at(i).position.y;
            
            // velocity component toward track
            double factor = (std::pow((x-x0)/a,n) + std::pow((y-y0)/b,n) - 1)/(std::pow((x-x0),n) + std::pow((y-y0),n));
            double up = -factor*(x-x0);
            double vp = -factor*(y-y0);
            
            // velocity component moving along track
            double m = (y-y0)/(x-x0);
            double xt = sgn(x-x0)*std::pow((1/(1/std::pow(a,n)+std::pow(m/b,n))),1.0/n) + x0; // intersection of track and line from origin to point, i.e. ~closest point on track
            double yt = m*(xt-x0) + y0;
            double ut = sgn(yt-y0);                        // tangent slope at point
            double vt = -1*sgn(yt-y0)*std::pow(b/a,n)*std::pow(1/m,n-1);
            double norm = std::pow(std::pow(ut,2)+std::pow(vt,2),0.5);
            
            // total velocity
            double u = k1*(ut/norm + k2*up);
            double v = k1*(vt/norm + k2*vp);
            
            // response
            geometry_msgs::Twist twistMsg;
            twistMsg.linear.x = u;
            twistMsg.linear.y = v;
            twistMsg.linear.z = 0;
            twistMsg.angular.x = 0;
            twistMsg.angular.y = 0;
            twistMsg.angular.z = 0;
            resp.twist.push_back(twistMsg);
        }
        
        return true;
    }

    int sgn(double val)
    {
        return ((val > 0) - (val < 0));
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "velocity_map_node");
    
    velocity_map vm;
    
    ros::spin();
    return 0;
}
