#include <ros/ros.h>
#include <switch_vis_exp/MapVel.h>

class velocity_map
{
    ros::NodeHandle nh;
    ros::ServiceServer service;
    
    // Parameters
    double a;       // length of track
    double b;       // width of track
    double k1;      // overall velocity gain
    double k2;      // weighting between getting back to track and moving in direction of track
    int n;          // sharpness and squareness of track
    
public:
    velocity_map()
    {
        // Get parameters
        nh.param<double>("a",a,1);
        nh.param<double>("b",b,1);
        nh.param<double>("k1",k1,0.5);
        nh.param<double>("k2",k2,0.5);
        nh.param<int>("n",n,4);
        
        // Start service
        service = nh.advertiseService("get_velocity", &velocity_map::get_velocity,this);
    }

    bool get_velocity(switch_vis_exp::MapVel::Request &req,switch_vis_exp::MapVel::Response &resp)
    {
        // point on map
        double x = req.pose.position.x;
        double y = req.pose.position.y;
        
        // velocity component toward track
        double factor = (std::pow(x/a,n) + std::pow(y/b,n) - 1)/(std::pow(x,n) + std::pow(y,n));
        double up = -factor*x;
        double vp = -factor*y;
        
        // velocity component moving along track
        double xt = sgn(x)*std::pow((1/(1/std::pow(a,n)+std::pow(y/(x*b),n))),1.0/n); // intersection of track and line from origin to point, i.e. ~closest point on track
        double yt = (y/x)*xt;
        double ut = sgn(yt);                        // tangent slope at point
        double vt = -1*sgn(yt)*std::pow(b/a,n)*std::pow(xt/yt,n-1);
        double norm = std::pow(std::pow(ut,2)+std::pow(vt,2),0.5);
        
        // total velocity
        double u = k1*(ut/norm + k2*up);
        double v = k1*(vt/norm + k2*vp);
        
        // response
        resp.twist.linear.x = u;
        resp.twist.linear.y = v;
        resp.twist.linear.z = 0;
        resp.twist.angular.x = 0;
        resp.twist.angular.y = 0;
        resp.twist.angular.z = 0;
        
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
