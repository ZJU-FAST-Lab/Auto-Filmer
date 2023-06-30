#include <iostream>
#include <math.h>
#include <ros/ros.h>
#include <eigen3/Eigen/Eigen>
#include "nav_msgs/Odometry.h"
#include <quadrotor_msgs/GimbalCmd.h>
#include <quadrotor_msgs/GimbalState.h>
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

ros::Subscriber _cmd_sub;
ros::Publisher  _state_pub, _fov_pub_;

bool rcv_cmd = false;
quadrotor_msgs::GimbalState state;
quadrotor_msgs::GimbalCmd _cmd;
double _init_theta;

// fov visualize
double max_dis_ = 4.0;
double x_max_dis_gain_ = 0.64;
double y_max_dis_gain_ = 0.82;
visualization_msgs::Marker markerNode_fov;
visualization_msgs::Marker markerEdge_fov;
std::vector<Eigen::Vector3d> fov_node;
Eigen::Vector3d  tbc(0.0, 0.0, 0.0);

void rcvCmdCallBack(const quadrotor_msgs::GimbalCmd cmd)
{	
	rcv_cmd = true;
	_cmd    = cmd;
}

void pubOdom()
{	
    state.header.stamp    = ros::Time::now();
	state.header.frame_id = "world";

	if(rcv_cmd)
	{
        state.angle = _cmd.angle;
        state.rate = _cmd.rate;        
	}

    _state_pub.publish(state);
}

void pub_fov_visual(Eigen::Vector3d& p, Eigen::Quaterniond& q) {
  visualization_msgs::Marker clear_previous_msg;
  clear_previous_msg.action = visualization_msgs::Marker::DELETEALL;

  visualization_msgs::MarkerArray markerArray_fov;
  markerNode_fov.points.clear();
  markerEdge_fov.points.clear();

  std::vector<geometry_msgs::Point> fov_node_marker;
  for (int i = 0; i < (int)fov_node.size(); i++) {
    Eigen::Vector3d vector_temp;
    Eigen::Matrix3d Rbc = Eigen::AngleAxisd(state.angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    vector_temp = q * (Rbc *fov_node[i] + tbc)+ p;

    geometry_msgs::Point point_temp;
    point_temp.x = vector_temp[0];
    point_temp.y = vector_temp[1];
    point_temp.z = vector_temp[2];
    fov_node_marker.push_back(point_temp);
  }

  markerNode_fov.points.push_back(fov_node_marker[0]);
  markerNode_fov.points.push_back(fov_node_marker[1]);
  markerNode_fov.points.push_back(fov_node_marker[2]);
  markerNode_fov.points.push_back(fov_node_marker[3]);
  markerNode_fov.points.push_back(fov_node_marker[4]);

  markerEdge_fov.points.push_back(fov_node_marker[0]);
  markerEdge_fov.points.push_back(fov_node_marker[1]);

  markerEdge_fov.points.push_back(fov_node_marker[0]);
  markerEdge_fov.points.push_back(fov_node_marker[2]);

  markerEdge_fov.points.push_back(fov_node_marker[0]);
  markerEdge_fov.points.push_back(fov_node_marker[3]);

  markerEdge_fov.points.push_back(fov_node_marker[0]);
  markerEdge_fov.points.push_back(fov_node_marker[4]);

  markerEdge_fov.points.push_back(fov_node_marker[0]);
  markerEdge_fov.points.push_back(fov_node_marker[1]);

  markerEdge_fov.points.push_back(fov_node_marker[1]);
  markerEdge_fov.points.push_back(fov_node_marker[2]);

  markerEdge_fov.points.push_back(fov_node_marker[2]);
  markerEdge_fov.points.push_back(fov_node_marker[3]);

  markerEdge_fov.points.push_back(fov_node_marker[3]);
  markerEdge_fov.points.push_back(fov_node_marker[4]);

  markerEdge_fov.points.push_back(fov_node_marker[4]);
  markerEdge_fov.points.push_back(fov_node_marker[1]);

  markerArray_fov.markers.push_back(clear_previous_msg);
  markerArray_fov.markers.push_back(markerNode_fov);
  markerArray_fov.markers.push_back(markerEdge_fov);
  _fov_pub_.publish(markerArray_fov);
}

void odom_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    if (msg->header.frame_id == std::string("null"))
        return;
    Eigen::Vector3d fov_p(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    Eigen::Quaterniond fov_q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    pub_fov_visual(fov_p, fov_q);
}

void fov_visual_init(std::string msg_frame_id) {
  double x_max_dis = max_dis_ * x_max_dis_gain_;
  double y_max_dis = max_dis_ * y_max_dis_gain_;

  fov_node.resize(5);
  fov_node[0][0] = 0;
  fov_node[0][1] = 0;
  fov_node[0][2] = 0;

  fov_node[1][2] = x_max_dis;
  fov_node[1][1] = y_max_dis;
  fov_node[1][0] = max_dis_;

  fov_node[2][2] = x_max_dis;
  fov_node[2][1] = -y_max_dis;
  fov_node[2][0] = max_dis_;

  fov_node[3][2] = -x_max_dis;
  fov_node[3][1] = -y_max_dis;
  fov_node[3][0] = max_dis_;

  fov_node[4][2] = -x_max_dis;
  fov_node[4][1] = y_max_dis;
  fov_node[4][0] = max_dis_;

  markerNode_fov.header.frame_id = msg_frame_id;
  // markerNode_fov.header.stamp = msg_time;
  markerNode_fov.action = visualization_msgs::Marker::ADD;
  markerNode_fov.type = visualization_msgs::Marker::SPHERE_LIST;
  markerNode_fov.ns = "fov_nodes";
  // markerNode_fov.id = 0;
  markerNode_fov.pose.orientation.w = 1;
  markerNode_fov.scale.x = 0.05;
  markerNode_fov.scale.y = 0.05;
  markerNode_fov.scale.z = 0.05;
  markerNode_fov.color.r = 0;
  markerNode_fov.color.g = 0.8;
  markerNode_fov.color.b = 1;
  markerNode_fov.color.a = 1;

  markerEdge_fov.header.frame_id = msg_frame_id;
  // markerEdge_fov.header.stamp = msg_time;
  markerEdge_fov.action = visualization_msgs::Marker::ADD;
  markerEdge_fov.type = visualization_msgs::Marker::LINE_LIST;
  markerEdge_fov.ns = "fov_edges";
  // markerEdge_fov.id = 0;
  markerEdge_fov.pose.orientation.w = 1;
  markerEdge_fov.scale.x = 0.05;
  markerEdge_fov.color.r = 0.5f;
  markerEdge_fov.color.g = 0.0;
  markerEdge_fov.color.b = 0.0;
  markerEdge_fov.color.a = 1;
}

int main (int argc, char** argv) 
{        
    ros::init (argc, argv, "gimbal_simulator");
    ros::NodeHandle nh( "~" );

    nh.param("init_theta", _init_theta,  0.0);

    _cmd_sub  = nh.subscribe( "gimbal_cmd", 1, rcvCmdCallBack );
    _state_pub = nh.advertise<quadrotor_msgs::GimbalState>("gimbal_state", 1);        
    _fov_pub_ = nh.advertise<visualization_msgs::MarkerArray>("fov_visual", 5);
    ros::Subscriber sub_odom = nh.subscribe("odom", 100, odom_callback);

    state.angle = _init_theta;
    state.rate = 0.0;

    fov_visual_init("world");
    ros::Rate rate(100);
    bool status = ros::ok();
    while(status) 
    {
		pubOdom();                   
        ros::spinOnce();
        status = ros::ok();
        rate.sleep();
    }

    return 0;
}