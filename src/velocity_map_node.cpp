#include <ros/ros.h>
#include <switch_vis_exp/MapVel.h>
#include <switch_vis_exp/RoadMap.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Point.h>
#include <tf/transform_datatypes.h>

#include <Eigen/Dense>
#include <random>

struct Road
{
    int node1;
    int node2;
    Eigen::Vector3d pt1;
    Eigen::Vector3d pt2;
};

class velocity_map
{
    ros::NodeHandle nh;
    ros::ServiceServer service;
    ros::ServiceServer mapService;
    
    // Parameters
    double a;       // length of track
    double b;       // width of track
    double x0;      // track center
    double y0;      // track center
    double k1;      // overall velocity gain
    double k2;      // weighting between getting back to track and moving in direction of track
    double nodeDistThresh;
    int n;          // sharpness and squareness of track
    bool doRotation;
    
    // graph parameters
    bool streets;
    int numNodes;
    Eigen::MatrixXd nodeLocations;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> adjMat;
    Eigen::MatrixXi node2road;
    std::vector<Road> roads;
    
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
        nhp.param<double>("nodeDistThresh",nodeDistThresh,0.2);
        nhp.param<int>("n",n,4);
        nhp.param<bool>("doRotation",doRotation,false);
        nhp.param<bool>("streets",streets,false);
        
        // graph
        std::srand(ros::Time::now().nsec);
        numNodes = 7;
        nodeLocations.resize(numNodes,3);
        nodeLocations << -1.0, 1.0, 0.0,
                         0.0, 1.0, 0.0,
                         1.0, 1.0, 0.0,
                         0.0, 0.0, 0.0,
                         1.0, 0.0, 0.0,
                         -1.0, -1.0, 0.0,
                         1.0, -1.0, 0.0;
        nodeLocations *= 2;
        node2road = -1*Eigen::MatrixXi::Ones(numNodes,numNodes);
        adjMat.resize(numNodes,numNodes);
        adjMat << 1, 1, 0, 1, 0, 0, 0,
                  1, 1, 1, 1, 0, 0, 0,
                  0, 1, 1, 0, 1, 0, 0,
                  1, 1, 0, 1, 1, 1, 0,
                  0, 0, 1, 1, 1, 0, 1,
                  0, 0, 0, 1, 0, 1, 1,
                  0, 0, 0, 0, 1, 1, 1;
        
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = i+1; j < numNodes; j++)
            {
                if (adjMat(i,j))
                {
                    node2road(i,j) = roads.size();
                    node2road(j,i) = roads.size();
                    
                    Road oneRoad;
                    oneRoad.node1 = i;
                    oneRoad.node2 = j;
                    oneRoad.pt1 = nodeLocations.row(i).transpose();
                    oneRoad.pt2 = nodeLocations.row(j).transpose();
                    roads.push_back(oneRoad);
                }
            }
        }
        
        // Start service
        mapService = nh.advertiseService("get_map", &velocity_map::get_map,this);
        service = nh.advertiseService("get_velocity", &velocity_map::get_velocity,this);
    }
    
    bool get_map(switch_vis_exp::RoadMap::Request &req,switch_vis_exp::RoadMap::Response &resp)
    {
        for (int i = 0; i < roads.size(); i++)
        {
            geometry_msgs::Point pointMsg1;
            pointMsg1.x = roads.at(i).pt1(0);
            pointMsg1.y = roads.at(i).pt1(1);
            pointMsg1.z = roads.at(i).pt1(2);
            
            geometry_msgs::Point pointMsg2;
            pointMsg2.x = roads.at(i).pt2(0);
            pointMsg2.y = roads.at(i).pt2(1);
            pointMsg2.z = roads.at(i).pt2(2);
            
            resp.pt1.push_back(pointMsg1);
            resp.pt2.push_back(pointMsg2);
        }
        
        for (int i = 0; i < numNodes; i++)
        {
            geometry_msgs::Point nodeMsg;
            nodeMsg.x = nodeLocations(i,0);
            nodeMsg.y = nodeLocations(i,1);
            nodeMsg.z = nodeLocations(i,2);
            
            resp.nodes.push_back(nodeMsg);
        }
        
        return true;
    }

    bool get_velocity(switch_vis_exp::MapVel::Request &req,switch_vis_exp::MapVel::Response &resp)
    {
        if (streets)
        {
            double speed = 0.5;
            
            for (int i = 0; i < req.pose.size(); i++)
            {
                Eigen::Vector3d pos(req.pose.at(i).position.x,req.pose.at(i).position.y,req.pose.at(i).position.z);
                Eigen::Quaterniond q(req.pose.at(i).orientation.w,req.pose.at(i).orientation.x,req.pose.at(i).orientation.y,req.pose.at(i).orientation.z);
                q.normalize();
                int fromNode;
                int toNode;
                
                // initialize if needed
                if (req.fromNode.at(i) < 0)
                {
                    // Closest road
                    double minDist = 200;
                    int minInd;
                    Eigen::Vector3d minLine;
                    for (int j = 0; j < roads.size(); j++)
                    {
                        Eigen::Vector3d line = roads.at(j).pt2 - roads.at(j).pt1;
                        Eigen::Vector3d v = pos - roads.at(j).pt1;
                        double scale = line.normalized().dot(v);
                        double dist = (v - scale*line.normalized()).norm();
                        
                        if ((dist < minDist) && (0.0 <= scale) && (scale <= 1.0))
                        {
                            minDist = dist;
                            minInd = j;
                            minLine = line;
                        }
                    }
                    
                    Eigen::Vector3d forward = q*Eigen::Vector3d(1,0,0);
                    if (minLine.normalized().dot(forward) > (-1*minLine).normalized().dot(forward))
                    {
                        fromNode = roads.at(minInd).node1;
                        toNode = roads.at(minInd).node2;
                    }
                    else
                    {
                        fromNode = roads.at(minInd).node2;
                        toNode = roads.at(minInd).node1;
                    }
                }
                else
                {
                    fromNode = req.fromNode.at(i);
                    toNode = req.toNode.at(i);
                }
                
                // Update path if needed
                int roadInd = node2road(fromNode,toNode);
                Eigen::Vector3d line = roads.at(roadInd).pt2 - roads.at(roadInd).pt1;
                Eigen::Vector3d vec = pos - roads.at(roadInd).pt1;
                double scale = line.normalized().dot(vec);
                if (((nodeLocations.row(toNode) - pos.transpose()).norm() < nodeDistThresh))
                {
                    // Determine possible next nodes
                    std::vector<int> nextNodes;
                    for (int j = 0; j < numNodes; j++)
                    {
                        if (adjMat(toNode,j) && (j != toNode) && (j != fromNode))
                        {
                            nextNodes.push_back(j);
                        }
                    }
                    
                    // Pick one at random
                    std::default_random_engine generator(std::rand());
                    std::uniform_int_distribution<int> distribution(0,nextNodes.size()-1);
                    fromNode = toNode;
                    toNode = nextNodes.at(distribution(generator));
                }
                
                // Along path
                roadInd = node2road(fromNode,toNode);
                if (roads.at(roadInd).node1 == fromNode)
                {
                    line = roads.at(roadInd).pt2 - roads.at(roadInd).pt1;
                    vec = pos - roads.at(roadInd).pt1;
                }
                else
                {
                    line = roads.at(roadInd).pt1 - roads.at(roadInd).pt2;
                    vec = pos - roads.at(roadInd).pt2;
                }
                Eigen::Vector3d vecp = vec - line.normalized().dot(vec)*line.normalized();
                double dist = vecp.norm();
                Eigen::Vector3d vt = speed*line.normalized();
                Eigen::Vector3d vp = -1*10*dist*vecp;
                Eigen::Vector3d velocity = vt+vp;
                velocity.normalize();
                velocity *= speed;
                
                geometry_msgs::Twist twistMsg;
                if (doRotation)
                {
                    Eigen::Vector3d xAxis = q*Eigen::Vector3d(1,0,0);
                    Eigen::Vector3d outLinVel = std::abs(xAxis.dot(velocity.normalized()))*velocity;
                    Eigen::Vector3d outAngVel = 2*xAxis.cross(velocity.normalized());
                    
                    //std::cout << "velocity: " << velocity.transpose() << std::endl;
                    //std::cout << "outLinVel: " << outLinVel.transpose() << std::endl;
                    //std::cout << "outAngVel: " << outAngVel.transpose() << std::endl;
                    
                    twistMsg.linear.x = outLinVel(0);
                    twistMsg.linear.y = outLinVel(1);
                    twistMsg.linear.z = outLinVel(2);
                    twistMsg.angular.x = outAngVel(0);
                    twistMsg.angular.y = outAngVel(1);
                    twistMsg.angular.z = outAngVel(2);
                }
                else
                {
                    twistMsg.linear.x = velocity(0);
                    twistMsg.linear.y = velocity(1);
                    twistMsg.linear.z = 0;
                    twistMsg.angular.x = 0;
                    twistMsg.angular.y = 0;
                    twistMsg.angular.z = 0;
                }
                resp.fromNode.push_back(fromNode);
                resp.toNode.push_back(toNode);
                resp.twist.push_back(twistMsg);
            }
        }
        else
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
                double u = k1*(ut/norm + k2*std::tanh(up));
                double v = k1*(vt/norm + k2*std::tanh(vp));
                
                // response
                geometry_msgs::Twist twistMsg;
                if (doRotation)
                {
                    tf::Vector3 vel(u,v,0);
                    tf::Quaternion quat(req.pose.at(i).orientation.x,req.pose.at(i).orientation.y,req.pose.at(i).orientation.z,req.pose.at(i).orientation.w);
                    tf::Vector3 xAxis = tf::Transform(quat,tf::Vector3(0,0,0))*tf::Vector3(1,0,0);
                    tf::Vector3 outLinVel = std::abs(xAxis.dot(vel.normalized()))*vel;
                    tf::Vector3 outAngVel = xAxis.cross(vel.normalized());
                    
                    twistMsg.linear.x = outLinVel.getX();
                    twistMsg.linear.y = outLinVel.getY();
                    twistMsg.linear.z = outLinVel.getZ();
                    twistMsg.angular.x = outAngVel.getX();
                    twistMsg.angular.y = outAngVel.getY();
                    twistMsg.angular.z = outAngVel.getZ();
                }
                else
                {
                    twistMsg.linear.x = u;
                    twistMsg.linear.y = v;
                    twistMsg.linear.z = 0;
                    twistMsg.angular.x = 0;
                    twistMsg.angular.y = 0;
                    twistMsg.angular.z = 0;
                }
                resp.fromNode.push_back(-1);
                resp.toNode.push_back(-1);
                resp.twist.push_back(twistMsg);
            }
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
