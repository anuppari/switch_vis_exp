#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/CameraInfo.h>
#include <switch_vis_exp/MapVel.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <Eigen/Dense>

template <typename T>
T trapz(std::deque<double>, std::deque<T>);
Eigen::MatrixXd sigmaGen(const Eigen::Matrix<double, 7, Eigen::Dynamic>&, const Eigen::Vector3d&, const Eigen::Quaterniond&, const Eigen::Matrix<double, Eigen::Dynamic, 2>&, const Eigen::Matrix2d&);
Eigen::Matrix<double, 7, 1> fFunc(const Eigen::Matrix<double, 7, 1>&, const Eigen::Vector3d&, const Eigen::Vector3d&);
Eigen::Matrix<double,4,3> diffMat(const Eigen::Quaterniond);

class DataHandler
{
    ros::NodeHandle nh;
    tf::TransformListener tfl;
    std::deque<double> tBuff;
    std::deque< Eigen::Matrix<double, 7, 1> > etaBuff;
    std::deque< Eigen::Matrix<double, 7, 1> > fBuff;
    std::deque<Eigen::MatrixXd> sigmaBuff;
    Eigen::MatrixXd mu;
    Eigen::Matrix2d cov;
    ros::Timer watchdogTimer;
    double intWindow;
    
public:
    Eigen::Vector3d vCc;
    Eigen::Vector3d wGCc;
    Eigen::Vector3d vTt;
    Eigen::Vector3d wGTt;
    
    Eigen::Vector3d xCam;
    Eigen::Quaterniond qCam;
    
    bool estimatorOn;
    
    Eigen::Matrix<double, 7, 1> eta;
    Eigen::Matrix<double, 7, 1> scriptEta;
    Eigen::Matrix<double, 7, 1> scriptF;
    Eigen::MatrixXd scriptY;
    
    DataHandler(ros::NodeHandle& nhIn, double visibilityTimeout, double intWindowIn, const Eigen::MatrixXd& muIn, const Eigen::Matrix2d& covIn)
    {
        nh = nhIn;
        
        vCc << 0,0,0;
        wGCc << 0,0,0;
        vTt << 0,0,0;
        wGTt << 0,0,0;
        
        xCam << 0,0,0;
        qCam = Eigen::Quaterniond(1,0,0,0);
        
        mu = muIn;
        cov = covIn;
        
        intWindow = intWindowIn;
        estimatorOn = false;
        
        // Initialize watchdog timer for feature visibility check
        watchdogTimer = nh.createTimer(ros::Duration(visibilityTimeout),&DataHandler::timeout,this,true);
        watchdogTimer.stop(); // Dont start watchdog until feature first visible
    }
    
    void timeout(const ros::TimerEvent& event)
    {
        estimatorOn = false;
    }
    
    void targetVelCB(const geometry_msgs::TwistStampedConstPtr& twist)
    {
        vTt << twist->twist.linear.x,twist->twist.linear.y,twist->twist.linear.z;
        wGTt << twist->twist.angular.x,twist->twist.angular.y,twist->twist.angular.z;
    }
    
    void camVelCB(const geometry_msgs::TwistStampedConstPtr& twist)
    {
        vCc << twist->twist.linear.x,twist->twist.linear.y,twist->twist.linear.z;
        wGCc << twist->twist.angular.x,twist->twist.angular.y,twist->twist.angular.z;
    }
    
    void camPoseCB(const geometry_msgs::PoseStampedConstPtr& pose)
    {
        if (!estimatorOn)
        {
            xCam << pose->pose.position.x, pose->pose.position.y, pose->pose.position.z;
            qCam = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z);
        }
    }
    
    void targetPoseCB(const geometry_msgs::PoseStampedConstPtr& pose)
    {
        // stop timer
        watchdogTimer.stop();
        
        // get pose data
        Eigen::Vector3d x(pose->pose.position.x,pose->pose.position.y,pose->pose.position.z);
        Eigen::Quaterniond q(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z);
        eta << x, q;
        
        // get camera pose
        tf::StampedTransform tfCamPose;
        try
        {
            tfl.waitForTransform("world","image",pose->header.stamp,ros::Duration(0.01));
            tfl.lookupTransform("world","image",pose->header.stamp,tfCamPose);
        }
        catch(tf::TransformException ex) { return; }
        xCam << tfCamPose.getOrigin().getX(), tfCamPose.getOrigin().getY(), tfCamPose.getOrigin().getZ();
        qCam = Eigen::Quaterniond(tfCamPose.getRotation().getW(), tfCamPose.getRotation().getX(), tfCamPose.getRotation().getY(), tfCamPose.getRotation().getZ());
        
        // update integration buffers
        tBuff.push_back(pose->header.stamp.toSec());
        etaBuff.push_back(eta);
        fBuff.push_back(fFunc(eta,vCc,wGCc));
        sigmaBuff.push_back(sigmaGen(eta,xCam,qCam,mu,cov));
        while ((tBuff.back() - tBuff.front()) > intWindow)
        {
            tBuff.pop_front();
            etaBuff.pop_front();
            fBuff.pop_front();
            sigmaBuff.pop_front();
        }
        
        // Integrate
        scriptEta = trapz(tBuff,etaBuff);
        scriptF = trapz(tBuff,fBuff);
        scriptY = trapz(tBuff,sigmaBuff);
        
        // restart timer
        watchdogTimer.start();
        estimatorOn = true;
    }
    
};

void camInfoCB(const sensor_msgs::CameraInfoConstPtr& camInfoMsg) { }

template <typename T>
T trapz(std::deque<double> tBuff, std::deque<T> dataBuff)
{
    T out(dataBuff.at(0).rows(),dataBuff.at(0).cols());
    for (int i = 1; i < tBuff.size(); i++)
    {
        out += 0.5*(tBuff.at(i)-tBuff.at(i-1))*(dataBuff.at(i) + dataBuff.at(i-1));
    }
    
    return out;
}

//Eigen::VectorXd linspace(double start, double finish, int num)
//{
    //double factor = (finish-start)/((double) (num-1));
    //Eigen::Matrix<double, num, 1> out;
    //for (int i = 0; i < num; i++)
    //{
        //out(i) = i*factor;
    //}
    //out(0) = start;
    //out(num-1) = finish;
    //return out;
//}

void meshgrid(const Eigen::VectorXd& xVec, const Eigen::VectorXd& yVec, Eigen::MatrixXd& xMat, Eigen::MatrixXd& yMat)
{
    xMat.resize(xVec.rows(),yVec.rows());
    yMat.resize(xVec.rows(),yVec.rows());
    for (int i = 0; i < xVec.rows(); i++)
    {
        for (int j = 0; j < yVec.rows(); j++)
        {
            xMat(i,j) = xVec(i);
            yMat(i,j) = yVec(j);
        }
    }
}

Eigen::MatrixXd sigmaGen(const Eigen::Matrix<double, 7, Eigen::Dynamic>& eta, const Eigen::Vector3d& xCam, const Eigen::Quaterniond& qCam, const Eigen::Matrix<double, Eigen::Dynamic, 2>& mu, const Eigen::Matrix2d& cov)
{
    return Eigen::MatrixXd();
}

Eigen::Matrix<double, 7, 1> fFunc(const Eigen::Matrix<double, 7, 1>& eta, const Eigen::Vector3d& vCc, const Eigen::Vector3d& wGCc)
{
    Eigen::Vector3d x = eta.head<3>();
    Eigen::Quaterniond q(eta(3),eta(4),eta(5),eta(6));
    
    Eigen::Vector3d f1 = vCc + wGCc.cross(x);
    Eigen::Vector4d f2 = 0.5*diffMat(q)*(q.inverse()*wGCc);
    
    Eigen::Matrix<double, 7, 1> out;
    out << f1, f2;
    return out;
}

// Calculate differential matrix for relationship between quaternion derivative and angular velocity.
// qDot = 1/2*B*omega 
// See strapdown inertial book. If quaternion is orientation of frame 
// B w.r.t N in the sense that nP = q*bP*q', omega is ang. vel of frame B w.r.t. N,
// i.e. N_w_B, expressed in the B coordinate system
// q = [w,x,y,z]
Eigen::Matrix<double,4,3> diffMat(const Eigen::Quaterniond q)
{
    Eigen::Matrix<double,4,3> B;
    B << -q.x(), -q.y(), -q.z(), q.w(), -q.z(), q.y(), q.z(), q.w(), -q.x(), -q.y(), q.x(), q.w();
    return B;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "model_learning_node");
    ros::NodeHandle nh;
    tf::TransformBroadcaster tfbr;
    
    // Node Parameters
    ros::NodeHandle nhp("~");
    double k1 = 3;
    double k2 = 0.1;
    double kCL = 1;
    double intWindow = 1;
    int CLstackSize = 200;
    int stackFill = 0;
    double visibilityTimeout = 0.2;
    
    // Initialize Neural Network
    double a, b, x0, y0, mapWidth, mapHeight;
    int numKernalWidth, numKernalHeight;
    nhp.param<double>("a", a, 1.0);
    nhp.param<double>("b", b, 1.0);
    nhp.param<double>("x0", x0, 1.0);
    nhp.param<double>("y0", y0, 1.0);
    nhp.param<int>("numKernalWidth", numKernalWidth, 11);
    nhp.param<int>("numKernalHeight", numKernalHeight, 11);
    mapWidth = 2*a;
    mapHeight = 2*b;
    int numKernal = numKernalWidth*numKernalHeight; // N
    Eigen::Matrix2d cov = 0.3*Eigen::Matrix2d::Identity();
    //Eigen::VectorXd muXvec = linspace(x0-mapWidth,x0+mapWidth,numKernalWidth);
    //Eigen::VectorXd muYvec = linspace(y0-mapHeight,y0+mapHeight,numKernalHeight);
    Eigen::VectorXd muXvec = Eigen::VectorXd::LinSpaced(numKernalWidth,x0-mapWidth,x0+mapWidth);
    Eigen::VectorXd muYvec = Eigen::VectorXd::LinSpaced(numKernalHeight,y0-mapHeight,y0+mapHeight);
    Eigen::MatrixXd muXmat, muYmat;
    meshgrid(muXvec,muYvec,muXmat,muYmat);
    muXvec = Eigen::Map<Eigen::VectorXd>(muXmat.data(),muXmat.cols()*muXmat.rows()); // mat to vec
    muYvec = Eigen::Map<Eigen::VectorXd>(muYmat.data(),muYmat.cols()*muYmat.rows());
    Eigen::Matrix<double,2,Eigen::Dynamic> mu(2,muXvec.rows());
    mu << muXvec, muYvec;
    mu.transposeInPlace(); // Nx2
    //Eigen::MatrixXd (*sigma)(const Eigen::Matrix<double,7,Eigen::Dynamic>&, const Eigen::Vector3d&, const Eigen::Quaterniond&);
    //sigma = [](const Eigen::Matrix<double,7,Eigen::Dynamic>& eta, const Eigen::Vector3d& xCam, const Eigen::Quaterniond& qCam) {return sigmaGen(eta,xCam,qCam,
    
    // Subscribers
    DataHandler callbacks(nh, visibilityTimeout, intWindow, mu, cov);
    ros::Subscriber camVelSub = nh.subscribe("image/body_vel",1,&DataHandler::camVelCB,&callbacks);
    ros::Subscriber targetVelSub = nh.subscribe("ugv0/body_vel",1,&DataHandler::targetVelCB,&callbacks);
    ros::Subscriber targetPoseSub = nh.subscribe("relPose",1,&DataHandler::targetPoseCB,&callbacks);
    ros::Subscriber camPoseSub = nh.subscribe("image/body_vel",1,&DataHandler::camPoseCB,&callbacks);
    
    // DEBUG / SIM
    ros::Subscriber camInfoSub = nh.subscribe("camera/camera_info",1,camInfoCB);
    ros::Duration(1.0).sleep();
    camInfoSub.shutdown();
    
    // Generate pre-seed data
    ros::ServiceClient client = nh.serviceClient<switch_vis_exp::MapVel>("/get_velocity");
    Eigen::Vector3d xCam(0,0,0);
    Eigen::Quaterniond qCam(1,0,0,0);
    int numPts = 2000; // M
    Eigen::Matrix<double, 7, Eigen::Dynamic> eta(7,numPts);
    eta << Eigen::Matrix<double, 2, Eigen::Dynamic>::Random(2,numPts), Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(1,numPts), Eigen::Matrix<double, 4, Eigen::Dynamic>::Zero(4,numPts);
    Eigen::MatrixXd Y = sigmaGen(eta, xCam, qCam, mu, cov);
    
    
    
    ros::spin();
    return 0;
}
