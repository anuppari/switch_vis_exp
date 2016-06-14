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
template <typename Derived>
Eigen::MatrixXd signum(const Eigen::MatrixBase<Derived>&);

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
        eta << x, q.vec(), q.w();
        
        // get camera pose
        tf::StampedTransform tfCamPose;
        try
        {
            tfl.waitForTransform("world","image",pose->header.stamp,ros::Duration(0.01));
            tfl.lookupTransform("world","image",pose->header.stamp,tfCamPose);
        }
        catch(tf::TransformException ex)
        {
            std::cout << ex.what() << std::endl; 
            return;
        }
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
    T out = T::Zero(dataBuff.at(0).rows(),dataBuff.at(0).cols());
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

//void meshgrid(const Eigen::VectorXd& xVec, const Eigen::VectorXd& yVec, Eigen::MatrixXd& xMat, Eigen::MatrixXd& yMat)
//{
    //xMat.resize(xVec.rows(),yVec.rows());
    //yMat.resize(xVec.rows(),yVec.rows());
    //for (int i = 0; i < xVec.rows(); i++)
    //{
        //for (int j = 0; j < yVec.rows(); j++)
        //{
            //xMat(i,j) = xVec(i);
            //yMat(i,j) = yVec(j);
        //}
    //}
//}

Eigen::MatrixXd sigmaGen(const Eigen::Matrix<double, 7, Eigen::Dynamic>& eta, const Eigen::Vector3d& xCam, const Eigen::Quaterniond& qCam, const Eigen::Matrix<double, Eigen::Dynamic, 2>& mu, const Eigen::Matrix2d& cov)
{
    // eta: 7xM
    // mu: Nxdim
    
    int dim = 2;
    int M = eta.cols();
    int N = mu.rows();
    
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(7,6);
    Eigen::Matrix3d Q = qCam.inverse().toRotationMatrix();
    T.block<3,3>(0,0) = Q;
    Eigen::MatrixXd Y(7*M,6*N);
    std::cout << "sigma" << std::endl;
    for (int i = 0; i < M; i++)
    {
        // inputs
        Eigen::Vector3d x = eta.block<3,1>(0,i);
        Eigen::Quaterniond q(eta(6,i),eta(3,i),eta(4,i),eta(5,i));
        Eigen::Vector3d xg = qCam*x + xCam;
        Eigen::Quaterniond qg = qCam*q;
        Eigen::Vector2d pts(x(0),x(1));
        
        // construct sigma
        Eigen::MatrixXd dev = mu.transpose().colwise() - pts;
        Eigen::MatrixXd sigma = (-0.5*(dev.array()*(cov*dev).array()).colwise().sum()).exp();
        Eigen::MatrixXd sigmaMat = Eigen::MatrixXd::Zero(6,6*N);
        for (int j = 0; j < 6; j++) { sigmaMat.block(j,j*N,1,N) = sigma/sigma.sum(); } // kron(I(6),sigma)
        
        std::cout << sigma/sigma.sum() << std::endl;
        
        // construct T
        Eigen::Matrix<double, 4, 3> Bq = diffMat(q);
        T.block<4,3>(3,3) = Bq;
        
        Y.middleRows(7*i,7) = T*sigmaMat;
    }
    
    return Y;
}

Eigen::Matrix<double, 7, 1> fFunc(const Eigen::Matrix<double, 7, 1>& eta, const Eigen::Vector3d& vCc, const Eigen::Vector3d& wGCc)
{
    Eigen::Vector3d x = eta.head<3>();
    Eigen::Quaterniond q(eta(6),eta(3),eta(4),eta(5));
    
    Eigen::Vector3d f1 = vCc + wGCc.cross(x);
    Eigen::Vector4d f2 = 0.5*diffMat(q)*(q.inverse()*wGCc);
    
    Eigen::Matrix<double, 7, 1> out;
    out << f1, f2;
    return -1*out;
}

// Calculate differential matrix for relationship between quaternion derivative and angular velocity.
// qDot = 1/2*B*omega 
// See strapdown inertial book. If quaternion is orientation of frame 
// B w.r.t N in the sense that nP = q*bP*q', omega is ang. vel of frame B w.r.t. N,
// i.e. N_w_B, expressed in the B coordinate system
// q = [x,y,z,w]
Eigen::Matrix<double,4,3> diffMat(const Eigen::Quaterniond q)
{
    Eigen::Matrix<double,4,3> B;
    B << q.w(), -q.z(), q.y(), q.z(), q.w(), -q.x(), -q.y(), q.x(), q.w(), -q.x(), -q.y(), -q.z();
    return B;
}

template <typename Derived>
Eigen::MatrixXd signum(const Eigen::MatrixBase<Derived>& inMat)
{
    //Eigen::ArrayXXd zero = Eigen::ArrayXXd::Zero(inMat.rows(),inMat.cols);
    return ((inMat.array() > 0).template cast<double>() - (inMat.array() < 0).template cast<double>()).matrix();
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
    nhp.param<int>("numKernalWidth", numKernalWidth, 4);
    nhp.param<int>("numKernalHeight", numKernalHeight, 4);
    mapWidth = 2*a;
    mapHeight = 2*b;
    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "x0: " << x0 << std::endl;
    std::cout << "y0: " << y0 << std::endl;
    std::cout << "mapWidth: " << mapWidth << std::endl;
    std::cout << "mapHeight: " << mapHeight << std::endl;
    std::cout << "numKernalWidth: " << numKernalWidth << std::endl;
    std::cout << "numKernalHeight: " << numKernalHeight << std::endl;
    int numKernal = numKernalWidth*numKernalHeight; // N
    Eigen::Matrix2d cov = (0.3*Eigen::Matrix2d::Identity()).inverse();
    //Eigen::VectorXd muXvec = linspace(x0-mapWidth,x0+mapWidth,numKernalWidth);
    //Eigen::VectorXd muYvec = linspace(y0-mapHeight,y0+mapHeight,numKernalHeight);
    Eigen::VectorXd muXvec = Eigen::VectorXd::LinSpaced(numKernalWidth,x0-mapWidth,x0+mapWidth);
    Eigen::VectorXd muYvec = Eigen::VectorXd::LinSpaced(numKernalHeight,y0-mapHeight,y0+mapHeight);
    //Eigen::MatrixXd muXmat, muYmat;
    //meshgrid(muXvec,muYvec,muXmat,muYmat);
    Eigen::MatrixXd muXmat = muXvec.transpose().replicate(muYvec.rows(),1);
    Eigen::MatrixXd muYmat = muYvec.replicate(1,muXvec.rows());
    muXvec = Eigen::Map<Eigen::VectorXd>(muXmat.data(),muXmat.cols()*muXmat.rows()); // mat to vec
    muYvec = Eigen::Map<Eigen::VectorXd>(muYmat.data(),muYmat.cols()*muYmat.rows());
    Eigen::MatrixXd mu(2,muXvec.rows());
    mu << muXvec.transpose(), muYvec.transpose();
    mu.transposeInPlace(); // Nx2
    std::cout << "mu: " << mu.rows() << " " << mu.cols() << std::endl << mu << std::endl;
    //Eigen::MatrixXd (*sigma)(const Eigen::Matrix<double,7,Eigen::Dynamic>&, const Eigen::Vector3d&, const Eigen::Quaterniond&);
    //sigma = [](const Eigen::Matrix<double,7,Eigen::Dynamic>& eta, const Eigen::Vector3d& xCam, const Eigen::Quaterniond& qCam) {return sigmaGen(eta,xCam,qCam,
    
    // Initialize integral concurrent learning history stacks
    std::vector< Eigen::Matrix<double,7,1> > etaStack(CLstackSize,Eigen::Matrix<double,7,1>::Zero());
    std::vector< Eigen::Matrix<double,7,1> > scriptFstack(CLstackSize,Eigen::Matrix<double,7,1>::Zero());
    std::vector< Eigen::Matrix<double,7,Eigen::Dynamic> > scriptYstack(CLstackSize,Eigen::Matrix<double,7,Eigen::Dynamic>::Zero(7,6*numKernal));
    Eigen::MatrixXd Gamma = Eigen::MatrixXd::Identity(6*numKernal,6*numKernal);
    
    //// Subscribers
    DataHandler callbacks(nh, visibilityTimeout, intWindow, mu, cov);
    ros::Subscriber camVelSub = nh.subscribe("image/body_vel",1,&DataHandler::camVelCB,&callbacks);
    ros::Subscriber targetVelSub = nh.subscribe("ugv0/body_vel",1,&DataHandler::targetVelCB,&callbacks);
    ros::Subscriber targetPoseSub = nh.subscribe("relPose",1,&DataHandler::targetPoseCB,&callbacks);
    ros::Subscriber camPoseSub = nh.subscribe("image/pose",1,&DataHandler::camPoseCB,&callbacks);
    
    // DEBUG / SIM
    ros::Subscriber camInfoSub = nh.subscribe("camera/camera_info",1,camInfoCB);
    ros::Duration(1.0).sleep();
    camInfoSub.shutdown();
    
    // Generate pre-seed data
    ros::ServiceClient client = nh.serviceClient<switch_vis_exp::MapVel>("/get_velocity");
    Eigen::Vector3d xCam(0,0,0);
    Eigen::Quaterniond qCam(1,0,0,0);
    int numPts = 16; // M
    Eigen::Matrix<double, 7, Eigen::Dynamic> eta(7,numPts); // 7xM
    ROS_INFO("here1");
    //eta << (mapWidth*Eigen::VectorXd::Random(numPts).array()-x0).transpose(),
           //(mapHeight*Eigen::VectorXd::Random(numPts).array()-y0).transpose(),
           //Eigen::Matrix<double, 4, Eigen::Dynamic>::Zero(4,numPts),
           //Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(1,numPts);
    
    
    Eigen::VectorXd etaXvec = Eigen::VectorXd::LinSpaced(4,x0-mapWidth,x0+mapWidth);
    Eigen::VectorXd etaYvec = Eigen::VectorXd::LinSpaced(4,y0-mapHeight,y0+mapHeight);
    //Eigen::MatrixXd muXmat, muYmat;
    //meshgrid(muXvec,muYvec,muXmat,muYmat);
    Eigen::MatrixXd etaXmat = etaXvec.transpose().replicate(etaYvec.rows(),1);
    Eigen::MatrixXd etaYmat = etaYvec.replicate(1,etaXvec.rows());
    etaXvec = Eigen::Map<Eigen::VectorXd>(etaXmat.data(),etaXmat.cols()*etaXmat.rows()); // mat to vec
    etaYvec = Eigen::Map<Eigen::VectorXd>(etaYmat.data(),etaYmat.cols()*etaYmat.rows());
    eta << etaXvec.transpose(), etaYvec.transpose(), Eigen::Matrix<double, 4, Eigen::Dynamic>::Zero(4,numPts), Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(1,numPts);
    
    
    std::cout << "eta:" << std::endl << eta.transpose() << std::endl;
    ROS_INFO("here2");
    Eigen::MatrixXd Y = sigmaGen(eta, xCam, qCam, mu, cov);
    ROS_INFO("here2b");
    switch_vis_exp::MapVel srv;
    for (int i = 0; i < eta.cols(); i++)
    {
        geometry_msgs::Pose poseMsg;
        poseMsg.position.x = eta(0,i);
        poseMsg.position.y = eta(1,i);
        poseMsg.position.z = eta(2,i);
        poseMsg.orientation.x = eta(3,i);
        poseMsg.orientation.y = eta(4,i);
        poseMsg.orientation.z = eta(5,i);
        poseMsg.orientation.w = eta(6,i);
        srv.request.pose.push_back(poseMsg);
    }
    ROS_INFO("here3");
    while (!client.call(srv)) {}
    Eigen::MatrixXd bMat(eta.cols(),7);
    Eigen::Matrix3d Q = qCam.inverse().toRotationMatrix();
    ROS_INFO("here4");
    for (int i = 0; i < eta.cols(); i++)
    {
        Eigen::Quaterniond q(eta(6,i),eta(3,i),eta(4,i),eta(5,i));
        Eigen::Quaterniond qTG = qCam*q;
        Eigen::Vector3d vTg(srv.response.twist.at(i).linear.x,srv.response.twist.at(i).linear.y,srv.response.twist.at(i).linear.z);
        Eigen::Vector3d wTGt = qTG.inverse()*Eigen::Vector3d(srv.response.twist.at(i).angular.x,srv.response.twist.at(i).angular.y,srv.response.twist.at(i).angular.z);
        bMat.block<1,7>(i,0) << (Q*vTg).transpose(), (0.5*diffMat(q)*wTGt).transpose();
    }
    ROS_INFO("here5");
    //std::cout << "bMat:" << std::endl << bMat << std::endl;
    bMat.transposeInPlace();
    Eigen::VectorXd bVec(Eigen::Map<Eigen::VectorXd>(bMat.data(),bMat.rows()*bMat.cols()));
    ROS_INFO("here6");
    //std::cout << "bVec:" << bVec.rows() << " " << bVec.cols() << std::endl << bVec << std::endl;
    std::cout << "Y:" << std::endl;
    for (int j = 0; j < numPts; j++)
    {
        std::cout << j << std::endl;
        std::cout << Y.middleRows(j*7,7).transpose() << std::endl;
    }
    //Eigen::VectorXd thetaIdeal = Y.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bVec);
    Eigen::VectorXd thetaIdeal = Y.colPivHouseholderQr().solve(bVec);
    ROS_INFO("here7");
    std::cout << "thetaIdeal:" << std::endl << thetaIdeal << std::endl;
    
    // Wait for initial data
    while(!callbacks.estimatorOn) {
        ros::spinOnce();
        ros::Duration(0.2).sleep();
    }
    
    ROS_INFO("here8");
    
    return 0;
    
    // Main loop
    double lastTime = ros::Time::now().toSec();
    Eigen::Matrix<double,7,1> etaHat = Eigen::Matrix<double,7,1>::Zero();
    Eigen::Matrix<double,Eigen::Dynamic,1> thetaHat = Eigen::Matrix<double,Eigen::Dynamic,1>::Zero(6*numKernal);
    ros::Rate r(300);
    while (ros::ok())
    {
        // Time
        double timeNow = ros::Time::now().toSec();
        double delT = timeNow - lastTime;
        lastTime = timeNow;
        
        // Get latest data
        ros::spinOnce();
        Eigen::Matrix<double,7,1> eta = callbacks.eta;
        Eigen::Vector3d x = eta.head<3>();
        Eigen::Quaterniond q(eta(6),eta(3),eta(4),eta(5));
        Eigen::Matrix<double,7,1> etaTilde = eta - etaHat;
        Eigen::Vector3d vCc = callbacks.vCc;
        Eigen::Vector3d wGCc = callbacks.wGCc;
        Eigen::Vector3d xCam = callbacks.xCam;
        Eigen::Quaterniond qCam = callbacks.qCam;
        Eigen::Matrix<double,7,1> scriptEta = callbacks.scriptEta;
        Eigen::Matrix<double,7,1> scriptF = callbacks.scriptF;
        Eigen::MatrixXd scriptY = callbacks.scriptY;
        Eigen::Vector3d vTt = callbacks.vTt; //DEBUG
        Eigen::Vector3d wGTt = callbacks.wGTt; //DEBUG
        Eigen::Matrix<double,7,1> phi; //DEBUG
        phi << q*vTt, 0.5*diffMat(q)*wGTt; //DEBUG
        
        // Estimation
        Eigen::Matrix<double,7,1> etaHatDot;
        Eigen::Matrix<double,Eigen::Dynamic,1> thetaHatDot;
        if (callbacks.estimatorOn) // estimator
        {
            etaHatDot = sigmaGen(eta,xCam,qCam,mu,cov)*thetaIdeal + fFunc(eta,vCc,wGCc) + k1*etaTilde + k2*signum(etaTilde);
            //etaHatDot = phi + fFunc(eta,vCc,wGCc) + k1*etaTilde + k2*signum(etaTilde);
        }
        else // predictor
        {
            etaHatDot = sigmaGen(etaHat,xCam,qCam,mu,cov)*thetaIdeal + fFunc(etaHat,vCc,wGCc);
            //etaHatDot = phi + fFunc(etaHat,vCc,wGCc);
        }
        
        etaHat += etaHatDot*delT;
        
        // Publish
        tf::Transform tfT2C(tf::Quaternion(etaHat(3),etaHat(4),etaHat(5),etaHat(6)),tf::Vector3(etaHat(0),etaHat(1),etaHat(2)));
        tfbr.sendTransform(tf::StampedTransform(tfT2C,ros::Time::now(),"image","ugv0estimate"));
        
        r.sleep();
    }
    
    return 0;
}
























