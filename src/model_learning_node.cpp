#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Joy.h>
#include <switch_vis_exp/Output.h>
#include <switch_vis_exp/MapVel.h>
#include <switch_vis_exp/RoadMap.h>
#include <switch_vis_exp/CLdata.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <thread>
#include <mutex>
#include <atomic>
#include <random>

#define PI 3.14159265358979323846264

// ETA: [x,y,z,qx,qy,qz,qw]


// Calculate differential matrix for relationship between quaternion derivative and angular velocity.
// qDot = 1/2*B*omega 
// See strapdown inertial book. If quaternion is orientation of frame 
// B w.r.t N in the sense that nP = q*bP*q', omega is ang. vel of frame B w.r.t. N,
// i.e. N_w_B, expressed in the B coordinate system
// q = [x,y,z,w]
Eigen::Matrix<double,4,3> diffMat(const Eigen::Quaterniond& q)
{
    Eigen::Matrix<double,4,3> B;
    B << q.w(), -q.z(), q.y(), q.z(), q.w(), -q.x(), -q.y(), q.x(), q.w(), -q.x(), -q.y(), -q.z();
    return B;
}

template <typename T> int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

template <typename Derived>
Eigen::MatrixXd signum(const Eigen::MatrixBase<Derived>& inMat)
{
    //Eigen::ArrayXXd zero = Eigen::ArrayXXd::Zero(inMat.rows(),inMat.cols);
    return ((inMat.array() > 0).template cast<double>() - (inMat.array() < 0).template cast<double>()).matrix();
}

// CHANGE THIS TEMPLATE TO MATCH SIGNUM FUNCTION BELOW?
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

Eigen::Quaterniond continuousQuat(Eigen::Quaterniond qOld, Eigen::Quaterniond qNew)
{
    qNew.normalize();
    return qOld.dot(qNew) > qOld.coeffs().dot(-1*qNew.coeffs()) ? qNew : Eigen::Quaterniond(-1*qNew.coeffs());
}

Eigen::MatrixXd sigmaGen(bool streets, const Eigen::Matrix<double, 7, Eigen::Dynamic>& eta, const Eigen::Vector3d& xCam, const Eigen::Quaterniond& qCam, const Eigen::MatrixXd& mu, const Eigen::MatrixXd& cov)
{
    // eta: 7xM
    // mu: Nxdim
    // Y: 7*Mx6*N
    
    int dim = streets ? 3 : 2;
    int M = eta.cols();
    int N = mu.rows();
    
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(7,6);
    Eigen::Matrix3d Q = qCam.inverse().toRotationMatrix();
    T.block<3,3>(0,0) = Q;
    Eigen::MatrixXd Y(7*M,6*N);
    for (int i = 0; i < M; i++)
    {
        // inputs
        Eigen::Vector3d x = eta.block<3,1>(0,i);
        Eigen::Quaterniond q(eta(6,i),eta(3,i),eta(4,i),eta(5,i));
        Eigen::Vector3d xg = qCam*x + xCam;
        Eigen::Quaterniond qg = qCam*q;
        qg.normalize();
        Eigen::VectorXd pts;
        if (streets)
        {
            Eigen::AngleAxisd axisAng(qg);
            double yaw = std::fmod(sgn(axisAng.axis().dot(Eigen::Vector3d::UnitZ()))*axisAng.angle()+2*PI,2*PI); // [0,2*pi]
            pts = Eigen::Vector3d(xg(0),xg(1),yaw);
        }
        else
        {
            pts = Eigen::Vector2d(xg(0),xg(1));
        }
        
        // construct sigma
        Eigen::MatrixXd dev;
        if (streets)
        {
            Eigen::MatrixXd temp1 = mu.middleCols(0,2).transpose().colwise() - pts.head<2>();
            Eigen::ArrayXd diff1 = (mu.col(2).array() - pts(2)).abs();
            Eigen::MatrixXd temp2 = diff1.min(2*PI-diff1);
            dev.resize(3,N);
            dev << temp1, temp2.transpose();
        
            Eigen::MatrixXd test = Eigen::MatrixXd::Zero(N,4);
            test.col(0) = Eigen::VectorXd::LinSpaced(mu.rows(),0,mu.rows()-1);
            test.middleCols(1,3) = dev.transpose();
            //test.middleCols(4,2) = temp1.transpose();
            //test.col(6) = temp2;
            //std::cout << "dev \t\t\t temp1 \t\t temp2" << std::endl;
            //std::cout << test << std::endl;
        }
        else
        {
            dev = mu.transpose().colwise() - pts;
        }
        Eigen::MatrixXd sigma = (-0.5*(dev.array()*(cov*dev).array()).colwise().sum()).exp();
        
        int sigmaMaxInd1, sigmaMaxInd2;
        sigma.maxCoeff(&sigmaMaxInd1,&sigmaMaxInd2);
        //std::cout << "muMax: " << mu.block<1,2>(sigmaMaxInd2,0) << " " << mu(sigmaMaxInd2,2)*180/PI << std::endl;
        //std::cout << "sigmaMaxInd2: " << sigmaMaxInd2 << std::endl;
        int devMinInd;
        dev.colwise().norm().minCoeff(&devMinInd);
        //std::cout << "dev min: " << devMinInd << std::endl;
        
        
        Eigen::MatrixXd sigmaMat = Eigen::MatrixXd::Zero(6,6*N);
        for (int j = 0; j < 6; j++) { sigmaMat.block(j,j*N,1,N) = sigma; } // kron(I(6),sigma)  NORMALIZATION: /sigma.sum()
        
        // construct T
        Eigen::Matrix<double, 4, 3> Bq = diffMat(q);
        T.block<4,3>(3,3) = 0.5*Bq;
        
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
    //std::cout << "f: " << out.transpose() << std::endl;
    return -1*out;
}


class DataHandler
{
    ros::NodeHandle nh;
    tf::TransformListener& tfl;
    
    // Integration buffers
    std::deque<double> tBuff;
    std::deque< Eigen::Matrix<double, 7, 1> > etaBuff;
    std::deque< Eigen::Matrix<double, 7, 1> > fBuff;
    std::deque<Eigen::MatrixXd> sigmaBuff;
    double intWindow;
    
    // Regressor parameters
    Eigen::MatrixXd mu;
    Eigen::MatrixXd cov;
    bool streets;
    
    // Visibility
    ros::Timer watchdogTimer;
    ros::Timer switchingTimer;
    std::string markerID;
    std::string targetName;
    std::string imageName;
    bool artificialSwitching;
    bool joySwitching;
    std::vector<double> delTon;
    std::vector<double> delToff;
    int joySwitchButton;
    int dataTransferButton;
    
    // data transfer
    ros::ServiceServer service;
    const Eigen::VectorXd *etaStack;
    const Eigen::VectorXd *scriptFstack;
    const Eigen::MatrixXd *scriptYstack;
    
public:
    Eigen::Vector3d vCc;
    Eigen::Vector3d wGCc;
    Eigen::Vector3d vTt;
    Eigen::Vector3d wGTt;
    
    Eigen::Vector3d xCam;
    Eigen::Quaterniond qCam;
    
    bool estimatorOn;
    bool gotData;
    bool transferData;
    
    Eigen::Matrix<double, 7, 1> eta;
    Eigen::Matrix<double, 7, 1> scriptEta;
    Eigen::Matrix<double, 7, 1> scriptF;
    Eigen::MatrixXd scriptY;
    
    DataHandler(tf::TransformListener& tflIn, double visibilityTimeout, double intWindowIn, const Eigen::MatrixXd& muIn, const Eigen::MatrixXd& covIn, const Eigen::VectorXd& etaStackIn, const Eigen::VectorXd& scriptFstackIn, const Eigen::MatrixXd& scriptYstackIn) : tfl(tflIn)
    {
        ros::NodeHandle nhp("~");
        nhp.param<std::string>("markerID",markerID,"100");
        nhp.param<std::string>("targetName", targetName, "ugv0");
        nhp.param<std::string>("imageName", imageName, "image");
        nhp.param<bool>("artificialSwitching", artificialSwitching, false);
        nhp.param<bool>("joySwitching", joySwitching, false);
        nhp.param<bool>("streets", streets, false);
        std::vector<double> delTonDefault; delTonDefault.push_back(15.0); delTonDefault.push_back(30.0);
        std::vector<double> delToffDefault; delToffDefault.push_back(10.0); delToffDefault.push_back(20.0);
        nhp.param< std::vector<double> >("delTon", delTon, delTonDefault);
        nhp.param< std::vector<double> >("delToff", delToff, delToffDefault);
        service = nh.advertiseService(targetName+"/get_cl_data", &DataHandler::transfer_data,this);
        
        if (targetName == "ugv0")
        {
            joySwitchButton = 5;
            dataTransferButton = 11;
        }
        else
        {
            joySwitchButton = 4;
            dataTransferButton = 12;
        }
        
        std::cout << "delTon: " << delTon.at(0) << " " << delTon.at(1) << std::endl;
        std::cout << "delToff: " << delToff.at(0) << " " << delToff.at(1) << std::endl;
        
        intWindow = intWindowIn;
        mu = muIn;
        cov = covIn;
        gotData = false;
        etaStack = &etaStackIn;
        scriptFstack = &scriptFstackIn;
        scriptYstack = &scriptYstackIn;
        
        // Initialize
        vCc << 0,0,0;
        wGCc << 0,0,0;
        vTt << 0,0,0;
        wGTt << 0,0,0;
        
        xCam << 0,0,0;
        qCam = Eigen::Quaterniond(1,0,0,0);
        
        eta << Eigen::Matrix<double,6,1>::Zero(), 1;
        scriptEta << Eigen::Matrix<double,7,1>::Zero();
        scriptF << Eigen::Matrix<double,7,1>::Zero();
        scriptY = Eigen::MatrixXd::Zero(7,6*mu.rows());
        
        // Switching
        if (artificialSwitching)
        {
            estimatorOn = true;
            switchingTimer = nh.createTimer(ros::Duration(15.0),&DataHandler::switchingTimerCB,this,true);
        }
        else if (joySwitching)
        {
            estimatorOn = false;
        }
        else
        {
            // Initialize watchdog timer for feature visibility check
            watchdogTimer = nh.createTimer(ros::Duration(visibilityTimeout),&DataHandler::timeout,this,true);
            watchdogTimer.stop(); // Dont start watchdog until feature first visible
            estimatorOn = false;
        }
    }
    
    void timeout(const ros::TimerEvent& event)
    {
        std::cout << targetName+" watchdog: predictor" << std::endl;
        estimatorOn = false;
    }
    
    // If artificial switching, this method is called at set intervals (according to delTon, delToff) to toggle the estimator
    void switchingTimerCB(const ros::TimerEvent& event)
    {
        std::default_random_engine generator(std::rand());
        if (estimatorOn)
        {
            std::cout << "artificial switching: predictor" << std::endl;
            std::uniform_real_distribution<double> distribution(delToff.at(0),delToff.at(1));
            estimatorOn = false;
            switchingTimer = nh.createTimer(ros::Duration(distribution(generator)),&DataHandler::switchingTimerCB,this,true);
        }
        else
        {
            std::cout << "artificial switching: estimator" << std::endl;
                
            // Flush integration buffers
            tBuff.clear();
            etaBuff.clear();
            fBuff.clear();
            sigmaBuff.clear();
            
            std::uniform_real_distribution<double> distribution(delTon.at(0),delTon.at(1));
            estimatorOn = true;
            switchingTimer = nh.createTimer(ros::Duration(distribution(generator)),&DataHandler::switchingTimerCB,this,true);
        }
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
            qCam = continuousQuat(qCam,Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z));
            //qCam.normalize();
        }
    }
    
    void joyCB(const sensor_msgs::JoyConstPtr& joyMsg)
    {
        if (joySwitching) { estimatorOn = (bool) joyMsg->buttons[joySwitchButton]; } // RB
        transferData = (bool) joyMsg->buttons[dataTransferButton];
    }
    
    bool transfer_data(switch_vis_exp::CLdata::Request &req,switch_vis_exp::CLdata::Response &resp)
    {
        resp.etaStack = std::vector<double>(etaStack->data(), etaStack->data() + etaStack->size());
        resp.scriptFstack = std::vector<double>(scriptFstack->data(), scriptFstack->data() + scriptFstack->size());
        resp.scriptYstack = std::vector<double>(scriptYstack->data(), scriptYstack->data() + scriptYstack->size());
        
        //resp.etaStack.resize(etaStack->size());
        //Eigen::VectorXd::Map(&resp.etaStack[0],etaStack->size()) = *etaStack;
        
        //resp.scriptFstack.resize(scriptFstack->size());
        //Eigen::VectorXd::Map(&resp.scriptFstack[0],scriptFstack->size()) = *scriptFstack;
        
        //resp.etaStack.resize(scriptYstack->size());
        //Eigen::MatrixXd::Map(&resp.scriptYstack[0],scriptYstack->size()) = *scriptYstack;
        
        return true;
    }
    
    void targetPoseCB(const geometry_msgs::PoseStampedConstPtr& pose)
    {
        // Disregard erroneous tag tracks
        if (markerID.compare(pose->header.frame_id) != 0) { return; }
        
        // Switching
        if (artificialSwitching or joySwitching)
        {
            if (!estimatorOn) { return; }
        }
        else
        {
            // Feature in FOV, pause watchdog timer
            watchdogTimer.stop();
            if (!estimatorOn)
            {
                std::cout << targetName+" watchdog: estimator" << std::endl;
                
                // Flush integration buffers
                tBuff.clear();
                etaBuff.clear();
                fBuff.clear();
                sigmaBuff.clear();
            }
        }
        
        // get pose data
        Eigen::Vector3d x(pose->pose.position.x,pose->pose.position.y,pose->pose.position.z);
        Eigen::Quaterniond q = continuousQuat(Eigen::Quaterniond(eta.tail<4>()),Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z));
        //q.normalize();
        //eta << x, q.vec(), q.w();
        eta << x, q.coeffs();
        
        // get camera pose
        tf::StampedTransform tfCamPose;
        try
        {
            tfl.waitForTransform("world",imageName,pose->header.stamp,ros::Duration(0.01));
            tfl.lookupTransform("world",imageName,pose->header.stamp,tfCamPose);
        }
        catch(tf::TransformException ex) { std::cout << ex.what() << std::endl; return; }
        xCam << tfCamPose.getOrigin().getX(), tfCamPose.getOrigin().getY(), tfCamPose.getOrigin().getZ();
        qCam = continuousQuat(qCam,Eigen::Quaterniond(tfCamPose.getRotation().getW(), tfCamPose.getRotation().getX(), tfCamPose.getRotation().getY(), tfCamPose.getRotation().getZ()));
        //qCam.normalize();
        
        // update integration buffers
        tBuff.push_back(pose->header.stamp.toSec());
        etaBuff.push_back(eta);
        fBuff.push_back(fFunc(eta,vCc,wGCc));
        sigmaBuff.push_back(sigmaGen(streets,eta,xCam,qCam,mu,cov));
        while ((tBuff.back() - tBuff.front()) > intWindow)
        {
            tBuff.pop_front();
            etaBuff.pop_front();
            fBuff.pop_front();
            sigmaBuff.pop_front();
        }
        
        // Integrate
        scriptEta = etaBuff.back() - etaBuff.front();
        scriptF = trapz(tBuff,fBuff);
        scriptY = trapz(tBuff,sigmaBuff);
        
        if (!(artificialSwitching or joySwitching))
        {
            // Restart watchdog timer for feature visibility check
            watchdogTimer.start();
            estimatorOn = true;
        }
        gotData = true;
    }
};

//void updateStack(std::atomic<bool>& stackUpdateDone, std::mutex& m, const Eigen::Matrix<double,7,1>& scriptEta, const Eigen::Matrix<double,7,1>& scriptF, const Eigen::MatrixXd& scriptY, std::vector< Eigen::Matrix<double,7,1> >& etaStack, std::vector< Eigen::Matrix<double,7,1> >& scriptFstack, std::vector< Eigen::Matrix<double,7,Eigen::Dynamic> >& scriptYstack)
void updateStack(std::atomic<bool>& stackUpdateDone, std::mutex& m, const Eigen::Matrix<double,7,1>& scriptEta, const Eigen::Matrix<double,7,1>& scriptF, const Eigen::MatrixXd& scriptY, Eigen::VectorXd& etaStack, Eigen::VectorXd& scriptFstack, Eigen::MatrixXd& scriptYstack, Eigen::MatrixXd& term1, Eigen::MatrixXd& term2)
{
    // Copy data
    m.lock();
    Eigen::Matrix<double,7,1> scriptEtaHere = scriptEta;
    Eigen::Matrix<double,7,1> scriptFHere = scriptF;
    Eigen::MatrixXd scriptYHere = scriptY;
    m.unlock();
    
    int numData = scriptYstack.rows()/7;
    
    ros::Duration(0.2).sleep();
    
    //Eigen::MatrixXd sumYtY = scriptYstack.transpose()*scriptYstack;
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigSolver1(sumYtY,Eigen::EigenvaluesOnly);
    //double currEig = eigSolver1.eigenvalues().minCoeff();
    //std::cout << "currEig: " << currEig << std::endl;
    
    if (true) //(currEig < 1e-10)
    {
        std::default_random_engine generator(std::rand());
        std::uniform_int_distribution<int> distribution(0,numData-1);
        int maxInd = distribution(generator);
        
        //std::cout << "New data at random index " << maxInd << std::endl;
        
        Eigen::MatrixXd tempTerm1 = term1;
        Eigen::MatrixXd tempTerm2 = term2;
        
        // subtract old
        tempTerm1 -= scriptYstack.middleRows(7*maxInd,7).transpose()*(etaStack.middleRows(7*maxInd,7) - scriptFstack.middleRows(7*maxInd,7));
        tempTerm2 -= -1*scriptYstack.middleRows(7*maxInd,7).transpose()*scriptYstack.middleRows(7*maxInd,7);
        
        // add new new
        tempTerm1 += scriptYHere.transpose()*(scriptEtaHere - scriptFHere);
        tempTerm2 += -1*scriptYHere.transpose()*scriptYHere;
        
        etaStack.middleRows(7*maxInd,7) = scriptEtaHere;
        scriptFstack.middleRows(7*maxInd,7) = scriptFHere;
        scriptYstack.middleRows(7*maxInd,7) = scriptYHere;
        
        //Eigen::MatrixXd tempTerm1 = scriptYstack.transpose()*(etaStack - scriptFstack);
        //Eigen::MatrixXd tempTerm2 = -1*scriptYstack.transpose()*scriptYstack;
        
        m.lock();
        term1 = tempTerm1;
        term2 = tempTerm2;
        m.unlock();
    }
    //else
    //{
        //// New stack
        //sumYtY += scriptYHere.transpose()*scriptYHere;
        //Eigen::VectorXd newEigVals = Eigen::VectorXd::Zero(numData);
        //for (int i = 0; i < numData; i++)
        //{
            //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigSolver2(sumYtY-scriptYstack.middleRows(7*i,7).transpose()*scriptYstack.middleRows(7*i,7),Eigen::EigenvaluesOnly);
            //newEigVals(i) = eigSolver2.eigenvalues().minCoeff();
        //}
        
        //// Replace with new data
        //int maxInd;
        //if (newEigVals.maxCoeff(&maxInd) > currEig)
        //{
            //std::cout << "New data at index " << maxInd << std::endl;
            
            //etaStack.middleRows(7*maxInd,7) = scriptEtaHere;
            //scriptFstack.middleRows(7*maxInd,7) = scriptFHere;
            //scriptYstack.middleRows(7*maxInd,7) = scriptYHere;
            
            //Eigen::MatrixXd tempTerm1 = scriptYstack.transpose()*(etaStack - scriptFstack);
            //Eigen::MatrixXd tempTerm2 = -1*scriptYstack.transpose()*scriptYstack;
            
            //m.lock();
            //term1 = tempTerm1;
            //term2 = tempTerm2;
            //m.unlock();
        //}
    //}
    stackUpdateDone = true;
}

int initNN(bool streets, Eigen::MatrixXd& cov, Eigen::MatrixXd& mu)
{
    ros::NodeHandle nhp("~");
    int numKernal;
    if (streets)
    {
        double kernalSeparation;
        nhp.param<double>("kernalSeparation", kernalSeparation, 0.3);
        
        ros::NodeHandle nh;
        ros::ServiceClient client = nh.serviceClient<switch_vis_exp::RoadMap>("/get_map");
        switch_vis_exp::RoadMap srv;
        while (!client.call(srv)) {}
        std::vector<Eigen::Vector3d> MUs;
        std::cout << "numRoads: " << srv.response.pt1.size() << std::endl;
        for (int i = 0; i < srv.response.pt1.size(); i++)
        {
            Eigen::Vector3d pt1(srv.response.pt1.at(i).x,srv.response.pt1.at(i).y,srv.response.pt1.at(i).z);
            Eigen::Vector3d pt2(srv.response.pt2.at(i).x,srv.response.pt2.at(i).y,srv.response.pt2.at(i).z);
            Eigen::Vector3d line = pt2 - pt1;
            
            int numNewKernals = (int) std::round(line.norm()/kernalSeparation);
            double angle = std::fmod(std::atan2(line(1),line(0)) + 2*PI,2*PI); // [0,2*pi]
            for (int j = 1; j < numNewKernals; j++)
            {
                double actualKernalSeparation = line.norm()/numNewKernals;
                Eigen::Vector3d newPt = pt1 + j*actualKernalSeparation*line.normalized();
                MUs.push_back(Eigen::Vector3d(newPt(0),newPt(1),angle));
                MUs.push_back(Eigen::Vector3d(newPt(0),newPt(1),std::fmod(angle+PI,2*PI)));
            }
        }
        mu.resize(MUs.size(),3);// Nx3
        numKernal = mu.rows();
        for (int i = 0; i < MUs.size(); i++)
        {
            mu.row(i) = MUs.at(i).transpose();
        }
        
        
        Eigen::MatrixXd muTemp = Eigen::MatrixXd::Zero(mu.rows(),4);
        muTemp.col(0) = Eigen::VectorXd::LinSpaced(mu.rows(),0,mu.rows()-1);
        muTemp.middleCols(1,3) = mu;
        muTemp.col(3) *= 180/PI;
        
        Eigen::MatrixXd tempCov = 0.1*Eigen::Matrix3d::Identity();
        tempCov(2,2) = 0.1;
        std::cout << "tempCov: " << std::endl << tempCov << std::endl;
        cov = tempCov.inverse();
    }
    else
    {
        double a, b, x0, y0, mapWidth, mapHeight;
        int numKernalWidth, numKernalHeight;
        nhp.param<double>("a", a, 1.0);
        nhp.param<double>("b", b, 1.0);
        nhp.param<double>("x0", x0, 1.0);
        nhp.param<double>("y0", y0, 1.0);
        nhp.param<int>("numKernalWidth", numKernalWidth, 9);
        nhp.param<int>("numKernalHeight", numKernalHeight, 9);
        mapWidth = 2*a;
        mapHeight = 2*b;
        numKernal = numKernalWidth*numKernalHeight; // N
        cov = (0.3*Eigen::Matrix2d::Identity()).inverse();
        Eigen::VectorXd muXvec = Eigen::VectorXd::LinSpaced(numKernalWidth,x0-mapWidth,x0+mapWidth);
        Eigen::VectorXd muYvec = Eigen::VectorXd::LinSpaced(numKernalHeight,y0-mapHeight,y0+mapHeight);
        Eigen::MatrixXd muXmat = muXvec.transpose().replicate(muYvec.rows(),1);
        Eigen::MatrixXd muYmat = muYvec.replicate(1,muXvec.rows());
        muXvec = Eigen::Map<Eigen::VectorXd>(muXmat.data(),muXmat.cols()*muXmat.rows()); // mat to vec
        muYvec = Eigen::Map<Eigen::VectorXd>(muYmat.data(),muYmat.cols()*muYmat.rows());
        mu.resize(2,muXvec.rows());
        mu << muXvec.transpose(), muYvec.transpose();
        mu.transposeInPlace(); // Nx2
        
    }
    std::cout << "mu: " << mu.rows() << std::endl << mu << std::endl;
    
    return numKernal;
}

void genData(double x0, double y0, double mapWidth, double mapHeight, const Eigen::MatrixXd& mu, const Eigen::MatrixXd& cov, int& fillAmount, Eigen::VectorXd& etaStack, Eigen::VectorXd& scriptFstack, Eigen::MatrixXd& scriptYstack)
{
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    bool streets;
    nhp.param<bool>("streets", streets, false);
    ros::ServiceClient client = nh.serviceClient<switch_vis_exp::MapVel>("/get_velocity");
    Eigen::Vector3d xCam(0,0,0);
    Eigen::Quaterniond qCam(1,0,0,0);
    int numPts = 1600; // M
    Eigen::Matrix<double, 7, Eigen::Dynamic> eta(7,numPts); // 7xM
    eta << (mapWidth*Eigen::VectorXd::Random(numPts).array()+x0).transpose(),
           (mapHeight*Eigen::VectorXd::Random(numPts).array()+y0).transpose(),
           Eigen::Matrix<double, 4, Eigen::Dynamic>::Zero(4,numPts),
           Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(1,numPts);
    Eigen::MatrixXd Y = sigmaGen(streets,eta, xCam, qCam, mu, cov);
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
    while (!client.call(srv)) {}
    Eigen::MatrixXd bMat(eta.cols(),7);
    Eigen::Matrix3d Q = qCam.inverse().toRotationMatrix();
    for (int i = 0; i < eta.cols(); i++)
    {
        Eigen::Quaterniond q(eta(6,i),eta(3,i),eta(4,i),eta(5,i));
        Eigen::Quaterniond qTG = qCam*q;
        Eigen::Vector3d vTg(srv.response.twist.at(i).linear.x,srv.response.twist.at(i).linear.y,srv.response.twist.at(i).linear.z);
        Eigen::Vector3d wTGt = qTG.inverse()*Eigen::Vector3d(srv.response.twist.at(i).angular.x,srv.response.twist.at(i).angular.y,srv.response.twist.at(i).angular.z);
        bMat.block<1,7>(i,0) << (Q*vTg).transpose(), (0.5*diffMat(q)*wTGt).transpose();
    }
    bMat.transposeInPlace();
    Eigen::VectorXd bVec(Eigen::Map<Eigen::VectorXd>(bMat.data(),bMat.rows()*bMat.cols()));
    //Eigen::VectorXd thetaIdeal = Y.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bVec);
    Eigen::VectorXd thetaIdeal = Y.colPivHouseholderQr().solve(bVec);
    
    //Prefill stack
    int prefillNum = 1500;
    fillAmount = prefillNum;
    etaStack.head(prefillNum) = bVec.head(prefillNum);
    scriptFstack.head(prefillNum) = Eigen::VectorXd::Zero(prefillNum);
    scriptYstack.topRows(prefillNum) = Y.topRows(prefillNum);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "model_learning_node");
    ros::NodeHandle nh;
    tf::TransformListener tfl;
    tf::TransformBroadcaster tfbr;
    
    // Node Parameters
    ros::NodeHandle nhp("~");
    double k1 = 3;
    double k2 = 0.1;
    double kCL = 1;
    double intWindow = 0.1;
    int CLstackSize = 1500;
    int stackFill = 0;
    double visibilityTimeout = 0.06;
    bool artificialSwitching, streetMultiBot;
    nhp.param<bool>("artificialSwitching", artificialSwitching, false);
    nhp.param<bool>("streetMultiBot",streetMultiBot,false); //Hz
    std::string imageName, targetName;
    nhp.param<std::string>("imageName", imageName, "image");
    nhp.param<std::string>("targetName", targetName, "ugv0");
    ros::ServiceClient dataTransferServiceClient;
    dataTransferServiceClient = targetName == "ugv0" ? nh.serviceClient<switch_vis_exp::CLdata>("ugv1/get_cl_data") : nh.serviceClient<switch_vis_exp::CLdata>("ugv0/get_cl_data");
    
    // Initialize Neural Network
    bool streets;
    nhp.param<bool>("streets", streets, false);
    Eigen::MatrixXd cov, mu;
    int numKernal = initNN(streets,cov,mu);
    
    // Initialize integral concurrent learning history stacks
    Eigen::VectorXd etaStack = Eigen::VectorXd::Zero(7*CLstackSize);
    Eigen::VectorXd scriptFstack = Eigen::VectorXd::Zero(7*CLstackSize);
    Eigen::MatrixXd scriptYstack = Eigen::MatrixXd::Zero(7*CLstackSize,6*numKernal);
    int fillAmount = 0;
    Eigen::Matrix<double,7,1> scriptEta = Eigen::Matrix<double,7,1>::Zero();
    Eigen::Matrix<double,7,1> scriptF = Eigen::Matrix<double,7,1>::Zero();
    Eigen::MatrixXd scriptY = Eigen::MatrixXd::Zero(7,6*numKernal);
    Eigen::MatrixXd Gamma = Eigen::MatrixXd::Identity(6*numKernal,6*numKernal);
    std::srand(ros::Time::now().nsec);
    
    // Publisher
    ros::Publisher outputPub = nh.advertise<switch_vis_exp::Output>("output",10);
    
    // Subscribers
    DataHandler callbacks(tfl, visibilityTimeout, intWindow, mu, cov, etaStack, scriptFstack, scriptYstack);
    ros::Subscriber camVelSub = nh.subscribe(imageName+"/body_vel",1,&DataHandler::camVelCB,&callbacks);
    ros::Subscriber targetVelSub = nh.subscribe(targetName+"/body_vel",1,&DataHandler::targetVelCB,&callbacks);
    ros::Subscriber targetPoseSub = nh.subscribe("markers",1,&DataHandler::targetPoseCB,&callbacks);
    ros::Subscriber camPoseSub = nh.subscribe(imageName+"/pose",1,&DataHandler::camPoseCB,&callbacks);
    ros::Subscriber joySub = nh.subscribe("joy",1,&DataHandler::joyCB,&callbacks);
    
    // Generate pre-seed data
    double a, b, x0, y0, mapWidth, mapHeight;
    nhp.param<double>("a", a, 1.0);
    nhp.param<double>("b", b, 1.0);
    nhp.param<double>("x0", x0, 1.0);
    nhp.param<double>("y0", y0, 1.0);
    mapWidth = 2*a;
    mapHeight = 2*b;
    //genData(x0, y0, mapWidth, mapHeight, mu, cov, fillAmount, etaStack, scriptFstack, scriptYstack);
    
    // Wait for initial data
    while(!callbacks.gotData) {
        ros::spinOnce();
        ros::Duration(0.2).sleep();
    }
    
    ros::AsyncSpinner spinner(4); // Use 4 threads
    spinner.start();
    
    ROS_INFO("here Main");
    
    // Main loop
    std::mutex m;
    std::atomic<bool> stackUpdateDone(true);
    std::thread stackThread;
    double lastTime = ros::Time::now().toSec();
    Eigen::Matrix<double,7,1> etaHat = Eigen::Matrix<double,7,1>::Zero();
    Eigen::Matrix<double,Eigen::Dynamic,1> thetaHat = Eigen::Matrix<double,Eigen::Dynamic,1>::Zero(6*numKernal);
    Eigen::MatrixXd term1 = scriptYstack.transpose()*(etaStack - scriptFstack);
    Eigen::MatrixXd term2 = -1*scriptYstack.transpose()*scriptYstack;
    ros::Rate r(300);
    while (ros::ok())
    {
        // Time
        ros::Time timeNowStamp = ros::Time::now();
        double timeNow = timeNowStamp.toSec();
        double delT = timeNow - lastTime;
        lastTime = timeNow;
        
        // Get latest data
        //ros::spinOnce();
        Eigen::Matrix<double,7,1> eta = callbacks.eta;
        Eigen::Vector3d x = eta.head<3>();
        Eigen::Quaterniond q(eta(6),eta(3),eta(4),eta(5));
        Eigen::Matrix<double,7,1> etaTilde = eta - etaHat;
        Eigen::Vector3d vCc = callbacks.vCc;
        Eigen::Vector3d wGCc = callbacks.wGCc;
        Eigen::Vector3d xCam = callbacks.xCam;
        Eigen::Quaterniond qCam = callbacks.qCam;
        m.lock();
        scriptEta = callbacks.scriptEta;
        scriptF = callbacks.scriptF;
        scriptY = callbacks.scriptY;
        m.unlock();
        Eigen::Vector3d vTt = callbacks.vTt;
        Eigen::Vector3d wGTt = callbacks.wGTt;
        Eigen::Matrix<double,7,1> phi;
        Eigen::Matrix<double,7,1> phiHat;
        
        // Sync data with other estimator (for multiBot estimation)
        if (streetMultiBot and callbacks.transferData)
        {
            switch_vis_exp::CLdata srv;
            if (dataTransferServiceClient.call(srv))
            {
                m.lock();
                etaStack = Eigen::Map<Eigen::VectorXd>(&srv.response.etaStack[0],etaStack.size());
                scriptFstack = Eigen::Map<Eigen::VectorXd>(&srv.response.scriptFstack[0],scriptFstack.size());
                scriptYstack = Eigen::Map<Eigen::MatrixXd>(&srv.response.scriptYstack[0],scriptYstack.rows(),scriptYstack.cols());
                m.unlock();
            }
        }
        
        // Ground truth
        tf::StampedTransform tfRelPose;
        try
        {
            tfl.waitForTransform(imageName,targetName,ros::Time(0),ros::Duration(0.01));
            tfl.lookupTransform(imageName,targetName,ros::Time(0),tfRelPose);
        }
        catch(tf::TransformException ex) { continue; }
        Eigen::Vector3d pos(tfRelPose.getOrigin().getX(),tfRelPose.getOrigin().getY(),tfRelPose.getOrigin().getZ());
        Eigen::Quaterniond quat = continuousQuat(Eigen::Quaterniond(etaHat.tail<4>()),Eigen::Quaterniond(tfRelPose.getRotation().getW(),tfRelPose.getRotation().getX(),tfRelPose.getRotation().getY(),tfRelPose.getRotation().getZ()));
        //quat.normalize();
        phi << quat*vTt, 0.5*diffMat(quat)*wGTt;
        
        // Sum history stack
        m.lock();
        Eigen::MatrixXd sum = term1 + term2*thetaHat;
        m.unlock();
        
        // Estimation
        Eigen::Matrix<double,7,1> etaHatDot;
        Eigen::Matrix<double,Eigen::Dynamic,1> thetaHatDot;
        if (callbacks.estimatorOn) // estimator
        {
            Eigen::MatrixXd Y = sigmaGen(streets,eta,xCam,qCam,mu,cov);
            phiHat = Y*thetaHat;
            //etaHatDot = phi + fFunc(eta,vCc,wGCc) + k1*etaTilde + k2*signum(etaTilde);
            etaHatDot = phiHat + fFunc(eta,vCc,wGCc) + k1*etaTilde + k2*signum(etaTilde);
            thetaHatDot = Gamma*Y.transpose()*etaTilde + kCL*Gamma*sum;
            
            if (false) //(fillAmount < CLstackSize) // fill stack until full
            {
                etaStack.middleRows(7*fillAmount,7) = scriptEta;
                scriptFstack.middleRows(7*fillAmount,7) = scriptF;
                scriptYstack.middleRows(7*fillAmount,7) = scriptY;
                fillAmount++;
            }
            else // replace data if it increases minimum eigenvalue
            {
                if (stackUpdateDone)
                {
                    if (stackThread.joinable())
                    {
                        stackThread.join();
                    }
                    stackUpdateDone = false;
                    stackThread = std::thread(updateStack,std::ref(stackUpdateDone),std::ref(m),std::ref(scriptEta),std::ref(scriptF),std::ref(scriptY),std::ref(etaStack),std::ref(scriptFstack),std::ref(scriptYstack),std::ref(term1),std::ref(term2));
                }
                
                //updateStack(stackUpdateDone, m, scriptEta, scriptF, scriptY, etaStack, scriptFstack, scriptYstack, term1, term2);
            }
        }
        else // predictor
        {
            phiHat = sigmaGen(streets,etaHat,xCam,qCam,mu,cov)*thetaHat;
            //etaHatDot = phi + fFunc(etaHat,vCc,wGCc);
            etaHatDot = phiHat + fFunc(etaHat,vCc,wGCc);
            thetaHatDot = kCL*Gamma*sum;
        }
        
        etaHat += etaHatDot*delT;
        thetaHat += thetaHatDot*delT;
        
        // Publish tf
        tf::Transform tfT2C(tf::Quaternion(etaHat(3),etaHat(4),etaHat(5),etaHat(6)),tf::Vector3(etaHat(0),etaHat(1),etaHat(2)));
        tfbr.sendTransform(tf::StampedTransform(tfT2C,ros::Time::now(),imageName,targetName+"estimate"));
        
        // Publish
        switch_vis_exp::Output outMsg;
        outMsg.header.stamp = timeNowStamp;
        outMsg.XYZ[0] = pos(0);                  outMsg.XYZ[1] = pos(1);                  outMsg.XYZ[2] = pos(2);
        outMsg.XYZhat[0] = etaHat(0);            outMsg.XYZhat[1] = etaHat(1);            outMsg.XYZhat[2] = etaHat(2);
        outMsg.XYZerror[0] = pos(0)-etaHat(0);   outMsg.XYZerror[1] = pos(1)-etaHat(1);   outMsg.XYZerror[2] = pos(2)-etaHat(2);
        outMsg.q[0] = quat.x();                  outMsg.q[1] = quat.y();                  outMsg.q[2] = quat.z();                  outMsg.q[3] = quat.w();
        outMsg.qhat[0] = etaHat(3);              outMsg.qhat[1] = etaHat(4);              outMsg.qhat[2] = etaHat(5);              outMsg.qhat[3] = etaHat(6);
        outMsg.qError[0] = quat.x()-etaHat(3);   outMsg.qError[1] = quat.y()-etaHat(4);   outMsg.qError[2] = quat.z()-etaHat(5);   outMsg.qError[3] = quat.w()-etaHat(6);
        outMsg.vc[0] = vCc(0);                   outMsg.vc[1] = vCc(1);                   outMsg.vc[2] = vCc(2);
        outMsg.wc[0] = wGCc(0);                  outMsg.wc[1] = wGCc(1);                  outMsg.wc[2] = wGCc(2);
        outMsg.vt[0] = vTt(0);                   outMsg.vt[1] = vTt(1);                   outMsg.vt[2] = vTt(2);
        outMsg.wt[0] = wGTt(0);                  outMsg.wt[1] = wGTt(1);                  outMsg.wt[2] = wGTt(2);
        outMsg.phi[0] = phi(0);                  outMsg.phi[1] = phi(1);                  outMsg.phi[2] = phi(2);                  outMsg.phi[3] = phi(3);
        outMsg.phi[4] = phi(4);                  outMsg.phi[5] = phi(5);                  outMsg.phi[6] = phi(6);
        outMsg.phiHat[0] = phiHat(0);            outMsg.phiHat[1] = phiHat(1);            outMsg.phiHat[2] = phiHat(2);            outMsg.phiHat[3] = phiHat(3);
        outMsg.phiHat[4] = phiHat(4);            outMsg.phiHat[5] = phiHat(5);            outMsg.phiHat[6] = phiHat(6);
        outMsg.thetaHat = std::vector<double>(thetaHat.data(), thetaHat.data() + thetaHat.rows()*thetaHat.cols());
        
        outMsg.estimatorOn = callbacks.estimatorOn;
        outMsg.usePredictor = true;
        outMsg.deadReckoning = false;
        outMsg.normalizedKinematics = false;
        outMsg.artificialSwitching = artificialSwitching;
        outMsg.useVelocityMap = true;
        outputPub.publish(outMsg);
        
        r.sleep();
    }
    
    return 0;
}

