#include <ros/ros.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

// need a class in order publish in the callback
class SubscribeAndPublish
{
    ros::NodeHandle nh;
    ros::Publisher velPub;
    ros::Publisher bodyVelPub;
    ros::Subscriber mocapSub;
    
    int window_size;            // Number of data points to store
    bool buffer_full;           // Estimation will start after buffer is full for first time
    VectorXd t_buff;            // ring buffer for time data
    VectorXd pos_buff;          // ring buffer for position data
    VectorXd quat_buff;         // ring buffer for orientation data
    int tInd;                   // index of oldest time data. Data at this index gets replaced with new data
    int pInd;                   // index of oldest position data Data at this index gets replaced with new data
    int qInd;                   // index of oldest orientation data Data at this index gets replaced with new data
public:
    SubscribeAndPublish()
    {
	// Parameters
	nh.param<int>("window_size",window_size,20);

	// Initialize buffers
	tInd = 0;
	pInd = 0;
	qInd = 0;
	t_buff.resize(window_size);
	pos_buff.resize(3*window_size);
	quat_buff.resize(4*window_size);
	buffer_full = false;
	
	// Velocity publishers
	velPub = nh.advertise<geometry_msgs::TwistStamped>("vel",1);
	bodyVelPub = nh.advertise<geometry_msgs::TwistStamped>("body_vel",1);

	//Mocap subscriber
	mocapSub = nh.subscribe("pose",10,&SubscribeAndPublish::poseCB,this);
    }
    
    void poseCB(const geometry_msgs::PoseStampedConstPtr& pose)
    {
	// Setting up least squares problem A*theta = P. theta is made up of the coefficients for the best fit line,
	// i.e., X = Mx*T + Bx, Y = My*t + By, Z = Mz*t + Bz. Velocity is estimated as the slope of the best fit line, i.e., Vx = Mx, Vy = My, Vz = Mz. 
	// Each block of data is arranged like this:
	// [Xi]     [1,Ti,0,0,0,0] * [Bx]
	// [Yi]  =  [0,0,1,Ti,0,0]   [Mx]
	// [Zi]     [0,0,0,0,1,Ti]   [By]
	//  \/      \_____  _____/   [My]
	//  Pi            \/         [Bz]
	//                Ai         [Mz]
	//                            \/
	//                           theta
	//
	// and then data is all stacked like this, where n is the windows_size:
	// [P1]     [A1] * [Bx]
	// [P2]  =  [A2]   [Mx]
	//  :        :     [By]
	// [Pn]     [An]   [My]
	//                 [Bz]
	//                 [Mz]

	// Fill buffers
	t_buff(tInd) = pose->header.stamp.toSec();
	pos_buff.segment<3>(pInd) << pose->pose.position.x, pose->pose.position.y, pose->pose.position.z;
	Vector4d q;
	q << pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z, pose->pose.orientation.w;
	int lastQind = (qInd + (4*window_size-4))%(4*window_size); // decrement with rollover. Can't do (qInd - 4)%(4*window_size), it results in negative number
	Vector4d lastQ = quat_buff.segment<4>(lastQind);
	if ((lastQ-(-1*q)).norm() < (lastQ-q).norm()) // deal with equivalent quaternions
	{
	    q *= -1;
	}
	quat_buff.segment<4>(qInd) << q;

	// Increment index, roll back over
	tInd = (tInd+1)%window_size;
	pInd = (pInd + 3)%(3*window_size);
	qInd = (qInd + 4)%(4*window_size);

	// If the index has rolled over once, the buffer is full
	if (tInd == 0)
	{
	    buffer_full = true;
	}

	if (buffer_full)
	{
	    // normalize time for numerical stability/accuracy of subsequent matrix inversion
	    double delT = t_buff.maxCoeff() - t_buff.minCoeff();
	    VectorXd t_norm = (t_buff.array() - t_buff.minCoeff())/delT;

	    // Solve LLS for best fit line parameters
	    MatrixXd Apos(3*window_size,6);
	    MatrixXd Aquat(4*window_size,8);
	    for (int ii = 0; ii < window_size; ii++)
	    {
		Apos.block<3,6>(ii*3,0) << 1,t_norm(ii),0,0,0,0, 0,0,1,t_norm(ii),0,0, 0,0,0,0,1,t_norm(ii);
		Aquat.block<4,8>(ii*4,0) << 1,t_norm(ii),0,0,0,0,0,0, 0,0,1,t_norm(ii),0,0,0,0, 0,0,0,0,1,t_norm(ii),0,0, 0,0,0,0,0,0,1,t_norm(ii);
	    }
	    Matrix<double,6,1> theta_pos = Apos.jacobiSvd(ComputeThinU | ComputeThinV).solve(pos_buff);
	    Matrix<double,8,1> theta_quat = Aquat.jacobiSvd(ComputeThinU | ComputeThinV).solve(quat_buff);

	    // Get velocities (linear in world coordinates, angular in body coordinates)
	    Vector3d v;
	    Vector4d qDot;
	    Vector3d wbody;
	    Matrix<double,4,3> B;
	    v << theta_pos(1)/delT, theta_pos(3)/delT, theta_pos(5)/delT;                           // rescaled to account for time normalization
	    qDot << theta_quat(1)/delT, theta_quat(3)/delT, theta_quat(5)/delT, theta_quat(7)/delT; // rescaled to account for time normalization
	    diffMat(q,B);
	    wbody = 2*(B.transpose())*qDot;

	    // Transform velocities (linear in body coordinates, angular in world coordinates)
	    Quaterniond quat(q(3),q(0),q(1),q(2));
	    Vector3d w = quat*wbody;
	    Vector3d vbody = quat.conjugate()*v;

	    // Publish
	    geometry_msgs::TwistStamped msg = geometry_msgs::TwistStamped();
	    msg.header.stamp = pose->header.stamp;
	    msg.header.frame_id = pose->header.frame_id;
	    msg.twist.linear.x = v(0);
	    msg.twist.linear.y = v(1);
	    msg.twist.linear.z = v(2);
	    msg.twist.angular.x = w(0);
	    msg.twist.angular.y = w(1);
	    msg.twist.angular.z = w(2);
	    velPub.publish(msg);

	    geometry_msgs::TwistStamped msgbody = geometry_msgs::TwistStamped();
	    msgbody.header.stamp = pose->header.stamp;
	    msgbody.header.frame_id = pose->header.frame_id+"_body";
	    msgbody.twist.linear.x = vbody(0);
	    msgbody.twist.linear.y = vbody(1);
	    msgbody.twist.linear.z = vbody(2);
	    msgbody.twist.angular.x = wbody(0);
	    msgbody.twist.angular.y = wbody(1);
	    msgbody.twist.angular.z = wbody(2);
	    bodyVelPub.publish(msgbody);
	}
    }
	
    // Calculate differential matrix for relationship between quaternion derivative and angular velocity.
    // qDot = 1/2*B*omega => omega = 2*B^T*qDot 
    // See strapdown inertial book. If quaternion is orientation of frame 
    // B w.r.t N in the sense that nP = q*bP*q', omega is ang. vel of frame B w.r.t. N,
    // i.e. N_w_B, expressed in the B coordinate system
    void diffMat(const Vector4d &q, Matrix<double,4,3> &B)
    {
	B << q(3), -q(2), q(1), q(2), q(3), -q(0), -q(1), q(0), q(3), -q(0), -q(1), -q(2);
    }


};//End of class SubscribeAndPublish


int main(int argc, char** argv)
{
    ros::init(argc, argv, "filter_node");
    
    SubscribeAndPublish sap;
    
    ros::spin();
    return 0;
}

