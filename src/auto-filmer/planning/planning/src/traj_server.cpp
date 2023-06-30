#include "traj_opt/trajectory.hpp"
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PolyTraj.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <quadrotor_msgs/GimbalCmd.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Float32MultiArray.h>

#include <traj_opt/trajectory.hpp>

ros::Publisher pos_cmd_pub_, gimbal_cmd_pub_;
ros::Time heartbeat_time_;
bool receive_traj_ = false;
quadrotor_msgs::PolyTraj trajMsg_, trajMsg_last_;
Eigen::Vector3d last_p_;
double last_yaw_ = 0;

void publish_cmd(int traj_id,
                 const Eigen::Vector3d &p,
                 const Eigen::Vector3d &v,
                 const Eigen::Vector3d &a,
                 const Eigen::Vector3d &j,
                 double y, double yd) {
  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id = traj_id;

  cmd.position.x = p(0);
  cmd.position.y = p(1);
  cmd.position.z = p(2);
  cmd.velocity.x = v(0);
  cmd.velocity.y = v(1);
  cmd.velocity.z = v(2);
  cmd.acceleration.x = a(0);
  cmd.acceleration.y = a(1);
  cmd.acceleration.z = a(2);
  cmd.jerk.x = j(0);
  cmd.jerk.y = j(1);
  cmd.jerk.z = j(2);
  cmd.yaw = y;
  cmd.yaw_dot = yd;
  pos_cmd_pub_.publish(cmd);
  last_p_ = p;
  last_yaw_ = y;
}

void publish_gimbal_cmd(const double &theta,
                    const double &theta_rate)
{
    quadrotor_msgs::GimbalCmd cmd;
    cmd.header.stamp = ros::Time::now();
    cmd.header.frame_id = "world";
    
    cmd.angle = theta;
    cmd.rate = theta_rate;
    cmd.mode = 2;
    gimbal_cmd_pub_.publish(cmd);

}

bool exe_traj(const quadrotor_msgs::PolyTraj &trajMsg) {
  double t = (ros::Time::now() - trajMsg.start_time).toSec();
  if (t > 0) {
    if (trajMsg.hover) {
      if (trajMsg.hover_p.size() != 3) {
        ROS_ERROR("[traj_server] hover_p is not 3d!");
      }
      Eigen::Vector3d p, v0;
      p.x() = trajMsg.hover_p[0];
      p.y() = trajMsg.hover_p[1];
      p.z() = trajMsg.hover_p[2];
      v0.setZero();
      publish_cmd(trajMsg.traj_id, p, v0, v0, v0, last_yaw_, 0); 
      return true;
    }
    if (trajMsg.order != 7) {
      ROS_ERROR("[traj_server] Only support trajectory order equals 5 now!");
      return false;
    }
    if (trajMsg.duration.size() * (trajMsg.order + 1) != trajMsg.coef_x.size()) {
      ROS_ERROR("[traj_server] WRONG trajectory parameters!");
      return false;
    }
    int piece_nums = trajMsg.duration.size();
    std::vector<double> dura(piece_nums);
    std::vector<CoefficientMat> cMats(piece_nums);
    std::vector<AngleCoefficientMat> cMats_angle(piece_nums);
    for (int i = 0; i < piece_nums; ++i) {
      int i8 = i * 8;
      cMats[i].row(0) << trajMsg.coef_x[i8 + 0], trajMsg.coef_x[i8 + 1], trajMsg.coef_x[i8 + 2],
          trajMsg.coef_x[i8 + 3], trajMsg.coef_x[i8 + 4], trajMsg.coef_x[i8 + 5],
          trajMsg.coef_x[i8 + 6], trajMsg.coef_x[i8 + 7];
      cMats[i].row(1) << trajMsg.coef_y[i8 + 0], trajMsg.coef_y[i8 + 1], trajMsg.coef_y[i8 + 2],
          trajMsg.coef_y[i8 + 3], trajMsg.coef_y[i8 + 4], trajMsg.coef_y[i8 + 5],
          trajMsg.coef_y[i8 + 6], trajMsg.coef_y[i8 + 7];
      cMats[i].row(2) << trajMsg.coef_z[i8 + 0], trajMsg.coef_z[i8 + 1], trajMsg.coef_z[i8 + 2],
          trajMsg.coef_z[i8 + 3], trajMsg.coef_z[i8 + 4], trajMsg.coef_z[i8 + 5],
          trajMsg.coef_z[i8 + 6], trajMsg.coef_z[i8 + 7];
      cMats_angle[i].row(0) << trajMsg.coef_psi[4 * i + 0], trajMsg.coef_psi[4 * i + 1], 
          trajMsg.coef_psi[4 * i + 2], trajMsg.coef_psi[4 * i + 3]; 
      cMats_angle[i].row(1) << trajMsg.coef_theta[4 * i + 0], trajMsg.coef_theta[4 * i + 1], 
          trajMsg.coef_theta[4 * i + 2], trajMsg.coef_theta[4 * i + 3]; 
      dura[i] = trajMsg.duration[i];
    }
    Trajectory traj(dura, cMats, cMats_angle);
    if (t > traj.getTotalDuration()) {
      ROS_ERROR("[traj_server] trajectory too short left!");
      return false;
    }
    Eigen::Vector3d j, p, v, a;
    j = traj.getJer(t);
    p = traj.getPos(t);
    v = traj.getVel(t);
    a = traj.getAcc(t);
    double yaw = traj.getAngle(t)(0);
    double yaw_dot = traj.getAngleRate(t)(0);
    publish_cmd(trajMsg.traj_id, p, v, a, j, yaw, yaw_dot);
    double theta =  traj.getAngle(t)(1);
    double theta_dot = traj.getAngleRate(t)(1);
    publish_gimbal_cmd(theta, theta_dot);
    return true;
  }
  return false;
}

void heartbeatCallback(const std_msgs::EmptyConstPtr &msg) {
  heartbeat_time_ = ros::Time::now();
}

void polyTrajCallback(const quadrotor_msgs::PolyTrajConstPtr &msgPtr) {
  trajMsg_ = *msgPtr;
  if (!receive_traj_) {
    trajMsg_last_ = trajMsg_;
    receive_traj_ = true;
  }
}

void cmdCallback(const ros::TimerEvent &e) {
  if (!receive_traj_) {
    return;
  }
  ros::Time time_now = ros::Time::now();
  if ((time_now - heartbeat_time_).toSec() > 0.5) {
    ROS_ERROR_ONCE("[traj_server] Lost heartbeat from the planner, is he dead?");
    publish_cmd(trajMsg_.traj_id, last_p_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), last_yaw_, 0); 
    return;
  }
  if (exe_traj(trajMsg_)) {
    trajMsg_last_ = trajMsg_;
    return;
  } else if (exe_traj(trajMsg_last_)) {
    return;
  }
//   ROS_ERROR_ONCE("[traj server] traj received invalid!");
  // publish_cmd(trajMsg_.traj_id, last_p_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0, 0);  // TODO yaw
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "traj_server");
  ros::NodeHandle nh("~");

  ros::Subscriber poly_traj_sub = nh.subscribe("trajectory", 10, polyTrajCallback);
  ros::Subscriber heartbeat_sub = nh.subscribe("heartbeat", 10, heartbeatCallback);

  pos_cmd_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("position_cmd", 50);
  gimbal_cmd_pub_= nh.advertise<quadrotor_msgs::GimbalCmd>("gimbal_cmd", 50);


  ros::Timer cmd_timer = nh.createTimer(ros::Duration(0.01), cmdCallback);

  ros::Duration(1.0).sleep();

  ROS_WARN("[Traj server]: ready.");

  ros::spin();

  return 0;
}