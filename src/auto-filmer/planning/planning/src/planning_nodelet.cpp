#include "ros/duration.h"
#include "ros/time.h"
#include "traj_opt/trajectory.hpp"
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <mapping/mapping.h>
#include <nav_msgs/Odometry.h>
#include <nodelet/nodelet.h>
#include <quadrotor_msgs/OccMap3d.h>
#include <quadrotor_msgs/PolyTraj.h>
#include <quadrotor_msgs/ReplanState.h>
#include <quadrotor_msgs/ShotParams.h>
#include <quadrotor_msgs/GimbalState.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <sys/types.h>
#include <traj_opt/traj_opt.h>

#include <Eigen/Core>
#include <atomic>
#include <env/env.hpp>
#include <prediction/prediction.hpp>
#include <thread>
#include <visualization/visualization.hpp>
#include <wr_msg/wr_msg.hpp>
#include <shot/shot.hpp>
#include <path/path.hpp>

namespace planning {

Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

class Nodelet : public nodelet::Nodelet {
 private:
  std::thread initThread_;
  ros::Subscriber gridmap_sub_, odom_sub_, target_sub_, triger_sub_, shot_sub, gimbal_sub, debug_sub_;
  ros::Timer plan_timer_, benchmark_timer_;

  ros::Publisher traj_pub_, heartbeat_pub_, replanState_pub_, map_pub_;

  std::shared_ptr<mapping::OccGridMap> gridmapPtr_;
  std::shared_ptr<env::Env> envPtr_;
  std::shared_ptr<path::PathSearch> pathPtr_;
  std::shared_ptr<visualization::Visualization> visPtr_;
  std::shared_ptr<traj_opt::TrajOpt> trajOptPtr_;
  std::shared_ptr<prediction::Predict> prePtr_;
  std::shared_ptr<shot::ShotGenerator> shotPtr_;

  // NOTE planning or fake target
  bool fake_ = false;
  Eigen::Vector3d goal_;

  // NOTE just for debug
  bool debug_ = false;
  quadrotor_msgs::ReplanState replanStateMsg_;
  ros::Publisher gridmap_pub_, inflate_gridmap_pub_;
  quadrotor_msgs::OccMap3d occmap_msg_;

  double tracking_dur_, tracking_dist_, tolerance_d_;

  Trajectory traj_poly_;
  ros::Time replan_stamp_;
  int traj_id_ = 0;
  bool wait_hover_ = true;
  bool force_hover_ = true;

  nav_msgs::Odometry odom_msg_, target_msg_;
  quadrotor_msgs::OccMap3d map_msg_;
  quadrotor_msgs::ShotParams shot_msg_;
  quadrotor_msgs::GimbalState gimbal_msg_;
  std::atomic_flag odom_lock_ = ATOMIC_FLAG_INIT;
  std::atomic_flag target_lock_ = ATOMIC_FLAG_INIT;
  std::atomic_flag gridmap_lock_ = ATOMIC_FLAG_INIT;
  std::atomic_flag shot_lock_ = ATOMIC_FLAG_INIT;
  std::atomic_flag gimbal_lock_ = ATOMIC_FLAG_INIT;
  std::atomic_bool odom_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool map_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool triger_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool target_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool shot_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool gimbal_received_ = ATOMIC_VAR_INIT(false);
  std::atomic_bool replan_received_ = ATOMIC_VAR_INIT(false);
  
  void pub_hover_p(const Eigen::Vector3d& hover_p, const ros::Time& stamp) {
    quadrotor_msgs::PolyTraj traj_msg;
    traj_msg.hover = true;
    traj_msg.hover_p.resize(3);
    for (int i = 0; i < 3; ++i) {
      traj_msg.hover_p[i] = hover_p[i];
    }
    traj_msg.start_time = stamp;
    traj_msg.traj_id = traj_id_++;
    traj_pub_.publish(traj_msg);
  }
  void pub_traj(const Trajectory& traj, const ros::Time& stamp) {
    quadrotor_msgs::PolyTraj traj_msg;
    traj_msg.hover = false;
    traj_msg.order = 7;
    Eigen::VectorXd durs = traj.getDurations();
    int piece_num = traj.getPieceNum();
    traj_msg.duration.resize(piece_num);
    traj_msg.coef_x.resize(8 * piece_num);
    traj_msg.coef_y.resize(8 * piece_num);
    traj_msg.coef_z.resize(8 * piece_num);
    traj_msg.coef_psi.resize(4 * piece_num);
    traj_msg.coef_theta.resize(4 * piece_num);
    for (int i = 0; i < piece_num; ++i) {
      traj_msg.duration[i] = durs(i);
      CoefficientMat cMat = traj[i].getCoeffMat();
      AngleCoefficientMat cMat_angle = traj[i].getAngleCoeffMat();
      int i8 = i * 8;
      for (int j = 0; j < 8; j++) {
        traj_msg.coef_x[i8 + j] = cMat(0, j);
        traj_msg.coef_y[i8 + j] = cMat(1, j);
        traj_msg.coef_z[i8 + j] = cMat(2, j);
      }
      for(int j = 0; j < 4; j++) 
      {
        traj_msg.coef_psi[4 * i + j] = cMat_angle(0, j);
        traj_msg.coef_theta[4 * i + j] = cMat_angle(1, j);
      }
    }
    traj_msg.start_time = stamp;
    traj_msg.traj_id = traj_id_++;
    traj_pub_.publish(traj_msg);
  }

  // debug
  void replanMsg_callback(const quadrotor_msgs::ReplanStateConstPtr& msgPtr)
  {
    replanStateMsg_ = *msgPtr;
    replan_received_ = true;
  }

  void triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    goal_ << msgPtr->pose.position.x, msgPtr->pose.position.y, 1.5;
    // ros::Duration(3.0).sleep();
    triger_received_ = true;
  }

  void shot_callback(const quadrotor_msgs::ShotParams::ConstPtr& msgPtr)
  {
      while(shot_lock_.test_and_set())
        ;
      shot_msg_ = *msgPtr;
      shot_received_ = true;
      shot_lock_.clear();
  }

  void gimbal_state_callback(const quadrotor_msgs::GimbalState::ConstPtr& msgPtr)
  {
      while(gimbal_lock_.test_and_set())
        ;
      gimbal_msg_ = *msgPtr;
      gimbal_received_ = true;
      gimbal_lock_.clear();
  }

  void odom_callback(const nav_msgs::Odometry::ConstPtr& msgPtr) {
    while (odom_lock_.test_and_set())
      ;
    odom_msg_ = *msgPtr;
    odom_received_ = true;
    odom_lock_.clear();
  }

//   void gimbal_odom_callback(const std_msgs::Float32MultiArray::ConstPtr& msg)
//   {

//   }

  void target_callback(const nav_msgs::Odometry::ConstPtr& msgPtr) {
    while (target_lock_.test_and_set())
      ;
    target_msg_ = *msgPtr;
    target_received_ = true;
    target_lock_.clear();
  }

  void gridmap_callback(const quadrotor_msgs::OccMap3dConstPtr& msgPtr) {
    while (gridmap_lock_.test_and_set())
      ;
    map_msg_ = *msgPtr;
    map_received_ = true;
    gridmap_lock_.clear();
  }

  // NOTE main callback
  void plan_timer_callback(const ros::TimerEvent& event) {
    heartbeat_pub_.publish(std_msgs::Empty());
    static bool plan_once_flag = true;
    // if(!plan_once_flag)
    //     return;

    if (!odom_received_ || !gimbal_received_ || !map_received_) {
        if(!odom_received_)
            ROS_INFO("no odom!");
        if(!gimbal_received_)
            ROS_INFO("no gimbal!");
        if(!map_received_)
            ROS_INFO("no map!");
      return;
    }
    // obtain state of odom
    while (odom_lock_.test_and_set())
      ;
    auto odom_msg = odom_msg_;
    odom_lock_.clear();
    while (gimbal_lock_.test_and_set())
        ;
    Eigen::Vector2d gimbal_state;
    gimbal_state(0) = gimbal_msg_.angle;
    gimbal_state(1) = gimbal_msg_.rate;
    gimbal_lock_.clear();

    Eigen::Vector3d odom_p(odom_msg.pose.pose.position.x,
                           odom_msg.pose.pose.position.y,
                           odom_msg.pose.pose.position.z);
    Eigen::Vector3d odom_v(odom_msg.twist.twist.linear.x,
                           odom_msg.twist.twist.linear.y,
                           odom_msg.twist.twist.linear.z);
    Eigen::Quaterniond odom_q(odom_msg.pose.pose.orientation.w,
                              odom_msg.pose.pose.orientation.x,
                              odom_msg.pose.pose.orientation.y,
                              odom_msg.pose.pose.orientation.z);
    if (!triger_received_) {
      ROS_INFO("no trigger!");
      return;
    }
    if (!target_received_) {
      ROS_INFO("no target!");
      return;
    }
    // NOTE obtain state of target
    while (target_lock_.test_and_set())
      ;
    replanStateMsg_.target = target_msg_;
    target_lock_.clear();
    Eigen::Vector3d target_p(replanStateMsg_.target.pose.pose.position.x,
                             replanStateMsg_.target.pose.pose.position.y,
                             replanStateMsg_.target.pose.pose.position.z);
    Eigen::Vector3d target_v(replanStateMsg_.target.twist.twist.linear.x,
                             replanStateMsg_.target.twist.twist.linear.y,
                             replanStateMsg_.target.twist.twist.linear.z);


    // NOTE force-hover: waiting for the speed of drone small enough
    if (force_hover_ && odom_v.norm() > 0.1) {
      return;
    }

    Eigen::Vector3d project_yaw = odom_q.toRotationMatrix().col(0);  // NOTE ZYX
    double now_yaw = std::atan2(project_yaw.y(), project_yaw.x());
    
    // NOTE obtain map
    while (gridmap_lock_.test_and_set())
      ;
    gridmapPtr_->from_msg(map_msg_);
    replanStateMsg_.occmap = map_msg_;
    gridmap_lock_.clear();
    prePtr_->setMap(*gridmapPtr_);

    // visualize the ray from drone to target
    if (envPtr_->checkRayValid(odom_p, target_p)) {
      visPtr_->visualize_arrow(odom_p, target_p, "ray", visualization::yellow);
    } else {
      visPtr_->visualize_arrow(odom_p, target_p, "ray", visualization::red);
    }

    // NOTE prediction
    std::vector<Eigen::Vector3d> target_predcit, target_vels;
    std::vector<double> target_headings;
    // ros::Time t_start = ros::Time::now();
    ros::Time t_start5 = ros::Time::now();
    bool generate_new_traj_success = prePtr_->predict(target_p, target_v, target_predcit, target_vels);
    // ros::Time t_stop = ros::Time::now();
    // std::cout << "predict costs: " << (t_stop - t_start).toSec() * 1e3 << "ms" << std::endl;

    // predict orientation
    static double last_estimate_orientation = 0.0;
    int predict_n = target_predcit.size();   
    if(generate_new_traj_success)
    {
      double predict_yaw;
      // NOTE orientation may change quickly when the velocity is too slow
      if(target_v.norm() < 0.1)
      {
          predict_yaw = last_estimate_orientation;
      }
      // NOTE a simple filter to orientation
      else{
          double vel_yaw = atan2(target_v(1), target_v(0));
          double d_yaw = fabs(vel_yaw - last_estimate_orientation);
          if(d_yaw > M_PI)
              d_yaw = 2 * M_PI - d_yaw;
          if(d_yaw > 0.1)  // 0.5
          {
              predict_yaw = vel_yaw;
              last_estimate_orientation = vel_yaw;
          }
          else
          {
              predict_yaw = last_estimate_orientation;
          }
      }
      target_headings.clear();
      predict_yaw = 0.0;  // comment this line so that the target velocity direction is its heading
      for(int i = 0; i < predict_n; i++)
      {
        target_headings.emplace_back(predict_yaw);
      }
    }

    if (generate_new_traj_success) {
      Eigen::Vector3d observable_p = target_predcit.back();
      visPtr_->visualize_path(target_predcit, "car_predict");
    }

    // NOTE replan state
    Eigen::MatrixXd iniState, iniAngle; // yaw, gimbal
    iniState.setZero(3, 4);
    iniAngle.setZero(2, 2);
    ros::Time replan_stamp = ros::Time::now() + ros::Duration(0.03);
    double replan_t = (replan_stamp - replan_stamp_).toSec();
    if (wait_hover_ || force_hover_ || replan_t > traj_poly_.getTotalDuration()) {
      // should replan from the hover state
      iniState.col(0) = odom_p;
      // iniState.col(1) = odom_v;
      iniState.col(1) = Eigen::Vector3d::Zero();
      iniAngle(0, 0) = now_yaw;
      iniAngle(0, 1) = 0.0;
      iniAngle.row(1) = gimbal_state.transpose();
    } else {
      // should replan from the last trajectory
      iniState.col(0) = traj_poly_.getPos(replan_t);
      iniState.col(1) = traj_poly_.getVel(replan_t);
      iniState.col(2) = traj_poly_.getAcc(replan_t);
      iniState.col(3) = traj_poly_.getJer(replan_t);
      iniAngle.col(0) = traj_poly_.getAngle(replan_t);
      iniAngle.col(1) = traj_poly_.getAngleRate(replan_t);
    }
    replanStateMsg_.header.stamp = ros::Time::now();
    replanStateMsg_.iniState.resize(12);
    Eigen::Map<Eigen::MatrixXd>(replanStateMsg_.iniState.data(), 3, 4) = iniState;
    replanStateMsg_.iniAngle.resize(4);
    Eigen::Map<Eigen::MatrixXd>(replanStateMsg_.iniAngle.data(), 2, 2) = iniAngle;

    //NOTE shot params
    if(shot_received_)
    {
        while(shot_lock_.test_and_set())
            ;
        shotPtr_ -> loadmsg(shot_msg_);
        shot_lock_.clear();
    }
    static double t_shot_ = 0;
    ros::Time t_front1 = ros::Time::now();
    std::vector<shot::ShotConfig> shotcfg_list;
                                                                                                                                                                                                                            
    shotPtr_ -> generate(shotcfg_list, predict_n);
    std::vector<Eigen::Vector2d> des_pos_image,  des_vel_image;
    for(int i = 0; i < predict_n; i++)
    {
        des_pos_image.emplace_back(shotcfg_list[i].image_p);
        des_vel_image.emplace_back(shotcfg_list[i].image_v);
    }
    ros::Time t_end1 = ros::Time::now();
    t_shot_ = (t_end1 - t_front1).toSec() * 1e3;

    replanStateMsg_.shot_d.resize(predict_n);
    replanStateMsg_.shot_angle.resize(predict_n);
    replanStateMsg_.shot_img_x.resize(predict_n);
    replanStateMsg_.shot_img_y.resize(predict_n);
    for(int i=0; i < predict_n; i++)
    {
        replanStateMsg_.shot_d[i] = shotcfg_list[i].distance;
        replanStateMsg_.shot_angle[i] = shotcfg_list[i].view_angle;
        replanStateMsg_.shot_img_x[i] = shotcfg_list[i].image_p.x();
        replanStateMsg_.shot_img_y[i] = shotcfg_list[i].image_p.y();
    }

    //NOTE kynodynamic astar
    double t_path = 0;
    std::vector<Eigen::Vector3d> kAstar_path, visible_ps;
    std::vector<Eigen::MatrixXd> view_polys;
    static int path_time =0, path_fail_time = 0;
    if (generate_new_traj_success) 
    {

        ros::Time t_front2 = ros::Time::now();
        generate_new_traj_success = pathPtr_->findPath(target_predcit, target_headings, shotcfg_list, 
                                                                                    iniState.col(0), iniState.col(1), kAstar_path);
        ros::Time t_end2 = ros::Time::now();
        path_time ++;
        if(!generate_new_traj_success)
            path_fail_time ++;
        t_path += (t_end2 - t_front2).toSec() * 1e3;

    }

    // NOTE region construct
    static int region_time = 0, region_fail_time = 0;
    static double t_region_avr = 0.0;
    if(generate_new_traj_success)
    {
        visPtr_->visualize_path(kAstar_path, "kAstar");
        ros::Time t_region1 = ros::Time::now();
        generate_new_traj_success = envPtr_->generate_view_regions(target_predcit, target_headings, shotcfg_list, 
                                            kAstar_path, visible_ps, view_polys);
        ros::Time t_region2 = ros::Time::now();
        if(!generate_new_traj_success)
            region_fail_time ++;
        double t_region = (t_region2 - t_region1).toSec() * 1e3;
        t_region_avr = (t_region_avr * region_time + t_region) / (++region_time);
        // std::cout << "t_region: " << t_region << std::endl;
        // std::cout << region_fail_time << "/" << region_time << std::endl;
        // std::cout << "average region time: " << t_region_avr << std::endl;
    }

    // NOTE determine whether to replan
    if (generate_new_traj_success
            && odom_v.norm() < 0.2 && target_v.norm() < 0.1 
            && (odom_p - visible_ps.back()).norm() < 0.3 )
    {
      if (!wait_hover_) {
        pub_hover_p(odom_p, ros::Time::now());
        wait_hover_ = true;
      }
      ROS_WARN("[planner] HOVERING...");
      replanStateMsg_.state = -1;
      replanState_pub_.publish(replanStateMsg_);
      return;
    } else {
      wait_hover_ = false;
    }

    Eigen::Vector3d p_start = iniState.col(0);
    std::vector<Eigen::Vector3d> path;

    static double t_path_ = 0;
    static double t_corridor_ = 0;
    static double t_optimization_ = 0;
    static double t_total_ = 0;
    static int times_path_ = 0;
    static int times_corridor_ = 0;
    static int times_optimization_ = 0;
    static int times_total_ = 0;

    // trajectory generation
    Trajectory traj;
    if (generate_new_traj_success) {
      visPtr_->visualize_pointcloud(visible_ps, "visible_ps");

      // NOTE corridor generating
      std::vector<Eigen::MatrixXd> hPolys;
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> keyPts;
      
      ros::Time t_front3 = ros::Time::now();
      envPtr_->pts2path(visible_ps, path);
      visPtr_->visualize_path(path, "astar");

      envPtr_->generateSFC(path, 2.0, hPolys, keyPts);
      ros::Time t_end3 = ros::Time::now();
      double t_corridor = (t_end3 - t_front3).toSec() * 1e3;

      envPtr_->visCorridor(hPolys);
      visPtr_->visualize_pairline(keyPts, "keyPts");

      // vis target -- visible_ps
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> visPts;
      visPts.clear();
      for(int i=0; i<predict_n; i++ )
      {
        std::pair<Eigen::Vector3d, Eigen::Vector3d> pairline(visible_ps[i], target_predcit[i]);
        visPts.emplace_back(pairline);
      }
      visPtr_->visualize_pairline(visPts, "visPts");

      // NOTE trajectory optimization
      Eigen::MatrixXd finState;
      finState.setZero(3, 4);
      finState.col(0) = path.back();
      finState.col(1) = target_v;
      // final angle: generated in optimization
    //   Eigen::Matrix2d finAngle; 
      ros::Time t_front4 = ros::Time::now();

      generate_new_traj_success = trajOptPtr_->generate_traj(iniState, 
                                                             finState,
                                                             iniAngle,
                                                             target_predcit,
                                                             target_vels,
                                                             view_polys,
                                                             des_pos_image,
                                                             des_vel_image,
                                                             hPolys, traj);
      ros::Time t_end4 = ros::Time::now();
      double t_optimization = (t_end4 - t_front4).toSec() * 1e3;
      ros::Time t_end5 = ros::Time::now();
      double t_total = (t_end5 - t_start5).toSec() * 1e3;

      // calculate time of path searching, corridor generation and optimization
      t_path_ = (t_path_ * times_path_ + t_path) / (++times_path_);
      t_corridor_ = (t_corridor_ * times_corridor_ + t_corridor) / (++times_corridor_);
      t_optimization_ = (t_optimization_ * times_optimization_ + t_optimization) / (++times_optimization_);
      t_total_ = t_path_ + t_region_avr + t_corridor_ + t_optimization_;

      // std::cout << "t_shot_: " << t_shot_ << "ms" << std::endl;
      // std::cout << "t_path_: " << t_path_ << " ms" << std::endl;
      // std::cout << "t_region_: " << t_region_avr << " ms" << std::endl;
      // std::cout << "t_corridor_: " << t_corridor_ << " ms" << std::endl;
      // std::cout << "t_optimization_: " << t_optimization_ << " ms" << std::endl;
      // std::cout << "t_total_: " << t_total_ << " ms" << std::endl;

      visPtr_->visualize_traj(traj, "traj");
      // for debug
      plan_once_flag = false;
    }

    // NOTE collision check
    static int replan_count = 5;
    bool valid = false;
    if (generate_new_traj_success) {
      valid = validcheck(traj, replan_stamp);
    } else {
      replanStateMsg_.state = -2;
      replanState_pub_.publish(replanStateMsg_);
    }
    if (valid) {
      replan_count = 25;  // 5
      force_hover_ = false;
      ROS_WARN("[planner] REPLAN SUCCESS");
      replanStateMsg_.state = 0;
      replanState_pub_.publish(replanStateMsg_);
      pub_traj(traj, replan_stamp);
      traj_poly_ = traj;
      replan_stamp_ = replan_stamp;
    } else if (force_hover_) {
      ROS_ERROR("[planner] REPLAN FAILED, HOVERING...");
      replanStateMsg_.state = 1;
      replanState_pub_.publish(replanStateMsg_);
      return;
    } else if (validcheck(traj_poly_, replan_stamp_) && replan_count > 0) {
      ROS_ERROR("[planner] REPLAN FAILED, EXECUTE LAST TRAJ...");
      replan_count --;
      replanStateMsg_.state = 3;
      replanState_pub_.publish(replanStateMsg_);
      return;  // current generated traj invalid but last is valid
    } else {
      force_hover_ = true;
      ROS_FATAL("[planner] EMERGENCY STOP!!!");
      replanStateMsg_.state = 2;
      replanState_pub_.publish(replanStateMsg_);
      pub_hover_p(iniState.col(0), replan_stamp);
      return;
    }
    visPtr_->visualize_traj(traj, "traj");
  }

  void fake_timer_callback(const ros::TimerEvent& event) {
      ROS_ERROR("This function is no longer in use!");
  }

  void debug_timer_callback(const ros::TimerEvent& event) {
    // ROS_WARN("This function is for debugs!");  
    heartbeat_pub_.publish(std_msgs::Empty());
    if(!replan_received_) return;

    ros::Time replan_stamp = ros::Time::now() + ros::Duration(0.03);
    Eigen::Vector3d target_p(replanStateMsg_.target.pose.pose.position.x,
                             replanStateMsg_.target.pose.pose.position.y,
                             replanStateMsg_.target.pose.pose.position.z);
    Eigen::Vector3d target_v(replanStateMsg_.target.twist.twist.linear.x,
                             replanStateMsg_.target.twist.twist.linear.y,
                             replanStateMsg_.target.twist.twist.linear.z);

    gridmapPtr_->from_msg(replanStateMsg_.occmap);
    prePtr_->setMap(*gridmapPtr_);
    map_pub_.publish(replanStateMsg_.occmap);

    std::vector<Eigen::Vector3d> target_predcit, target_vels;
    std::vector<double> target_headings;
    bool generate_new_traj_success = prePtr_->predict(target_p, target_v, target_predcit, target_vels);
    if (generate_new_traj_success) {
      visPtr_->visualize_path(target_predcit, "car_predict");
    }
    int predict_n = target_predcit.size();   
     // predict orientation
    static double last_estimate_orientation = 0.0;
    target_headings.clear();
    for(int i = 0; i < predict_n; i++)
    {
            double predict_yaw;
            // NOTE orientation may change quickly when the velocity is too slow
            if(target_vels[i].norm() < 0.1)
            {
                predict_yaw = last_estimate_orientation;
            }
            // NOTE a simple filter to orientation
            else{
                double vel_yaw = atan2(target_vels[i](1), target_vels[i](0));
                double d_yaw = fabs(vel_yaw - last_estimate_orientation);
                if(d_yaw > M_PI)
                    d_yaw = 2 * M_PI - d_yaw;
                if(d_yaw > 0.0)
                {
                    predict_yaw = vel_yaw;
                    last_estimate_orientation = vel_yaw;
                }
                else
                {
                    predict_yaw = last_estimate_orientation;
                }
            }
            // debug
            predict_yaw = 0.0;
            target_headings.emplace_back(predict_yaw);
    }
    // NOTE for the next planning
    last_estimate_orientation = target_headings[1];

    // NOTE replan state
    Eigen::MatrixXd iniState, iniAngle; // yaw, gimbal
    iniState.setZero(3, 4);
    iniAngle.setZero(2, 2);
    iniState = Eigen::Map<Eigen::MatrixXd>(replanStateMsg_.iniState.data(), 3, 4);
    iniAngle = Eigen::Map<Eigen::MatrixXd>(replanStateMsg_.iniAngle.data(), 2, 2);

    if(shot_received_)
    {
        while(shot_lock_.test_and_set())
            ;
        shotPtr_ -> loadmsg(shot_msg_);
        shot_lock_.clear();
    }
    std::vector<shot::ShotConfig> shotcfg_list;
    for(int i=0; i < predict_n; i++)
    {
        shot::ShotConfig cfg;
        cfg.distance = replanStateMsg_.shot_d[i];
        cfg.view_angle = replanStateMsg_.shot_angle[i];
        cfg.image_p.x() = replanStateMsg_.shot_img_x[i];
        cfg.image_p.y() = replanStateMsg_.shot_img_y[i];
        cfg.image_v.setZero();
        shotcfg_list.emplace_back(cfg);
    }

    // shotPtr_ -> generate(shotcfg_list, predict_n);
    std::vector<Eigen::Vector2d> des_pos_image,  des_vel_image;
    for(int i = 0; i < predict_n; i++)
    {
        des_pos_image.emplace_back(shotcfg_list[i].image_p);
        des_vel_image.emplace_back(shotcfg_list[i].image_v);
    }

    double t_path = 0;
    std::vector<Eigen::Vector3d> kAstar_path, visible_ps;
    std::vector<Eigen::MatrixXd> view_polys;
    if (generate_new_traj_success) 
    {
        ros::Time t_front2 = ros::Time::now();
        generate_new_traj_success = pathPtr_->findPath(target_predcit, target_headings, shotcfg_list, 
                                                                                    iniState.col(0), iniState.col(1), kAstar_path); 
        ros::Time t_end2 = ros::Time::now();
        t_path += (t_end2 - t_front2).toSec() * 1e3;
        std::cout << "t_path: " << t_path << std::endl;
    }


    // NOTE region construct
    if(generate_new_traj_success)
    {
        visPtr_->visualize_path(kAstar_path, "kAstar");
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> pairlines;
        for(int i = 0; i < predict_n; i++)
        {
            pairlines.emplace_back(std::pair<Eigen::Vector3d, Eigen::Vector3d>(target_predcit[i], kAstar_path[i]));
        }
        visPtr_->visualize_pairline(pairlines, "debug_pairline");
        ros::Time t_region1 = ros::Time::now();
        generate_new_traj_success = envPtr_->generate_view_regions(target_predcit, target_headings, shotcfg_list, 
                                            kAstar_path, visible_ps, view_polys);
        ros::Time t_region2 = ros::Time::now();
        double t_region = (t_region2 - t_region1).toSec() * 1e3;
        std::cout << "t_region: " << t_region << std::endl;
   }

    std::vector<Eigen::Vector3d> path;

    Trajectory traj;

    if (generate_new_traj_success) {
      visPtr_->visualize_pointcloud(visible_ps, "visible_ps");

      // NOTE corridor generating
      std::vector<Eigen::MatrixXd> hPolys;
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> keyPts;
      envPtr_->pts2path(visible_ps, path);
    //   visPtr_->visualize_path(kAstar_path, "astar");

      envPtr_->generateSFC(path, 2.0, hPolys, keyPts);

      envPtr_->visCorridor(hPolys);
      visPtr_->visualize_pairline(keyPts, "keyPts");

    // vis target -- visible_ps
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> visPts;
      visPts.clear();
      for(int i=0; i<predict_n; i++ )
      {
        std::pair<Eigen::Vector3d, Eigen::Vector3d> pairline(visible_ps[i], target_predcit[i]);
        visPts.emplace_back(pairline);
      }
      visPtr_->visualize_pairline(visPts, "visPts");

      // NOTE trajectory optimization
      Eigen::MatrixXd finState;
      finState.setZero(3, 4);
      finState.col(0) = path.back();
      finState.col(1) = target_v;
      // final angle: generated in optimization
    //   Eigen::Matrix2d finAngle; 
      ros::Time t_front4 = ros::Time::now();

      generate_new_traj_success = trajOptPtr_->generate_traj(iniState, 
                                                             finState,
                                                             iniAngle,
                                                             target_predcit,
                                                             target_vels,
                                                             view_polys,
                                                             des_pos_image,
                                                             des_vel_image,
                                                             hPolys, traj);
      ros::Time t_end4 = ros::Time::now();
      double t_optimization = (t_end4 - t_front4).toSec() * 1e3;
      std::cout << "t_optimization_: " << t_optimization << " ms" << std::endl;

      visPtr_->visualize_traj(traj, "traj");

    }
    if(generate_new_traj_success)
    {  
      pub_traj(traj, replan_stamp);
      ROS_WARN("[planner] REPLAN SUCCESS");
      replan_received_ = false;
    }
    else
    {
        ROS_ERROR("[planner] REPLAN FAILED");
    }
    // debug once
    replan_received_ = false;
  }

  void benchmark_timer_callback(const ros::TimerEvent& event) 
  {
    if (!odom_received_ || !gimbal_received_ || !map_received_) 
        return;
    if(!triger_received_)
        return;
    if(!target_received_)
        return;
    while (odom_lock_.test_and_set())
    ;
    auto odom_msg = odom_msg_;
    odom_lock_.clear();
    Eigen::Vector3d odom_p(odom_msg.pose.pose.position.x,
                           odom_msg.pose.pose.position.y,
                           odom_msg.pose.pose.position.z);
                           
    Eigen::Vector3d target_p;
    while (target_lock_.test_and_set())
      ;
    target_p <<  target_msg_.pose.pose.position.x, target_msg_.pose.pose.position.y, target_msg_.pose.pose.position.z;
    target_lock_.clear();

    // NOTE obtain map
    while (gridmap_lock_.test_and_set())
      ;
    gridmapPtr_->from_msg(map_msg_);
    gridmap_lock_.clear();

    static int fail_time = 0;
    static int block_time = 0;
    static int collide_time = 0;
    // static int out_fov_time = 0;
    static int total_time = 0;
    total_time ++;
    gridmapPtr_->isOccupied(odom_p);
    checkRayValid(odom_p, target_p);
    if(gridmapPtr_->isOccupied(odom_p))
    {
        collide_time ++;
        fail_time ++;
    }
    else if(!checkRayValid(odom_p, target_p))
    {
        block_time ++;
        fail_time ++;
    }
    // std::cout << "\033[32m"<< "[pn]total_time: " << total_time << " collide: " << double(collide_time)/total_time *100.0 
    //                     << " block: " << double(block_time)/total_time * 100.0 << "\033[0m" << std::endl;
  }

  bool validcheck(const Trajectory& traj, const ros::Time& t_start, const double& check_dur = 1.0) {
    double t0 = (ros::Time::now() - t_start).toSec();
    t0 = t0 > 0.0 ? t0 : 0.0;
    double delta_t = check_dur < traj.getTotalDuration() ? check_dur : traj.getTotalDuration();
    for (double t = t0; t < t0 + delta_t; t += 0.01) {
      Eigen::Vector3d p = traj.getPos(t);
      if (gridmapPtr_->isOccupied(p)) {
        return false;
      }
    }
    return true;
  }

  // benchamrk
  bool inline checkRayValid(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1) const {
    Eigen::Vector3d dp = p1 - p0;
    Eigen::Vector3i idx0 = gridmapPtr_->pos2idx(p0);
    Eigen::Vector3i idx1 = gridmapPtr_->pos2idx(p1);
    Eigen::Vector3i d_idx = idx1 - idx0;
    Eigen::Vector3i step = d_idx.array().sign().cast<int>();
    Eigen::Vector3d delta_t;
    for (int i = 0; i < 3; ++i) {
      delta_t(i) = dp(i) == 0 ? std::numeric_limits<double>::max() : 1.0 / std::fabs(dp(i));
    }
    Eigen::Vector3d t_max;
    for (int i = 0; i < 3; ++i) {
      t_max(i) = step(i) > 0 ? (idx0(i) + 1) - p0(i) / gridmapPtr_->resolution : p0(i) / gridmapPtr_->resolution - idx0(i);
    }
    t_max = t_max.cwiseProduct(delta_t);
    Eigen::Vector3i rayIdx = idx0;
    while ((rayIdx - idx1).squaredNorm() > 1) {
      if (gridmapPtr_->isOccupied(rayIdx)) {
        return false;
      }
      // find the shortest t_max
      int s_dim = 0;
      for (int i = 1; i < 3; ++i) {
        s_dim = t_max(i) < t_max(s_dim) ? i : s_dim;
      }
      rayIdx(s_dim) += step(s_dim);
      t_max(s_dim) += delta_t(s_dim);
    }
    return true;
  }

  void init(ros::NodeHandle& nh) {
    // set parameters of planning
    int plan_hz = 10;
    nh.getParam("plan_hz", plan_hz);
    nh.getParam("tracking_dur", tracking_dur_);
    nh.getParam("tracking_dist", tracking_dist_);
    nh.getParam("tolerance_d", tolerance_d_);
    nh.getParam("debug", debug_);
    nh.getParam("fake", fake_);

    gridmapPtr_ = std::make_shared<mapping::OccGridMap>();
    envPtr_ = std::make_shared<env::Env>(nh, gridmapPtr_);
    pathPtr_ = std::make_shared<path::PathSearch>(nh, gridmapPtr_);
    visPtr_ = std::make_shared<visualization::Visualization>(nh);
    trajOptPtr_ = std::make_shared<traj_opt::TrajOpt>(nh);
    prePtr_ = std::make_shared<prediction::Predict>(nh);
    shotPtr_ = std::make_shared<shot::ShotGenerator>(nh);

    heartbeat_pub_ = nh.advertise<std_msgs::Empty>("heartbeat", 10);
    traj_pub_ = nh.advertise<quadrotor_msgs::PolyTraj>("trajectory", 1);
    replanState_pub_ = nh.advertise<quadrotor_msgs::ReplanState>("replanState", 1);

    benchmark_timer_ = nh.createTimer(ros::Duration(0.05), &Nodelet::benchmark_timer_callback, this);

    if (debug_) {
      plan_timer_ = nh.createTimer(ros::Duration(1.0 / plan_hz), &Nodelet::debug_timer_callback, this);
      debug_sub_ = nh.subscribe<quadrotor_msgs::ReplanState>("replanState", 10, &Nodelet::replanMsg_callback, this, ros::TransportHints().tcpNoDelay());
      map_pub_ = nh.advertise<quadrotor_msgs::OccMap3d>("gridmap_inflate_dbg", 10);

    } else if (fake_) {
      plan_timer_ = nh.createTimer(ros::Duration(1.0 / plan_hz), &Nodelet::fake_timer_callback, this);
    } else {
      plan_timer_ = nh.createTimer(ros::Duration(1.0 / plan_hz), &Nodelet::plan_timer_callback, this);
    }
    gridmap_sub_ = nh.subscribe<quadrotor_msgs::OccMap3d>("gridmap_inflate", 1, &Nodelet::gridmap_callback, this, ros::TransportHints().tcpNoDelay());
    odom_sub_ = nh.subscribe<nav_msgs::Odometry>("odom", 10, &Nodelet::odom_callback, this, ros::TransportHints().tcpNoDelay());
    target_sub_ = nh.subscribe<nav_msgs::Odometry>("target", 10, &Nodelet::target_callback, this, ros::TransportHints().tcpNoDelay());
    triger_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("triger", 10, &Nodelet::triger_callback, this, ros::TransportHints().tcpNoDelay());
    shot_sub = nh.subscribe<quadrotor_msgs::ShotParams>("shot", 10, &Nodelet::shot_callback, this);
    gimbal_sub = nh.subscribe<quadrotor_msgs::GimbalState>("gimbal", 10, &Nodelet::gimbal_state_callback, this, ros::TransportHints().tcpNoDelay());

    ROS_WARN("Planning node initialized!");
  }

 public:
  void onInit(void) {
    ros::NodeHandle nh(getMTPrivateNodeHandle());
    initThread_ = std::thread(std::bind(&Nodelet::init, this, nh));
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace planning

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(planning::Nodelet, nodelet::Nodelet);
