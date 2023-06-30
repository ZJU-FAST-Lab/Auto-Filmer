// #include <eigen3/Eigen/src/Core/Matrix.h>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <ros/forwards.h>
#include <ros/publisher.h>
#include <ros/time.h>
#include <ros/transport_hints.h>
#include <sensor_msgs/Image.h>
#include <vector>
//include ros dep.
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <image_transport/image_transport.h>
#include <dynamic_reconfigure/server.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>

#include "tf/tf.h"
#include "tf/transform_datatypes.h"
#include <tf/transform_broadcaster.h>
//include pcl dep
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
 #include <pcl_conversions/pcl_conversions.h>
//include opencv and eigen
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <cv_bridge/cv_bridge.h>

//#include <cloud_banchmark/cloud_banchmarkConfig.h>
#include "depth_render.cuh"
#include "quadrotor_msgs/PositionCommand.h"
// temp use shot params
#include "quadrotor_msgs/GimbalState.h"
#include "quadrotor_msgs/ShotParams.h"
#include "shot.hpp"

// benchmark
#include <fstream>
#include <iostream>


using namespace cv;
using namespace std;
using namespace Eigen;

int *depth_hostptr;
cv::Mat depth_mat;

//camera param
int width, height;
double fx,fy,cx,cy;

DepthRender depthrender;
ros::Publisher pub_depth;
ros::Publisher pub_color;
ros::Publisher pub_pose;
ros::Publisher pub_pcl_wolrd;
ros::Publisher pub_gimbal;
ros::Publisher pub_best_view_p;

std::shared_ptr<shot::ShotGenerator> shotPtr_;

sensor_msgs::PointCloud2 local_map_pcl;
sensor_msgs::PointCloud2 local_depth_pcl;

ros::Subscriber odom_sub, target_sub, trigger_sub;
ros::Subscriber global_map_sub, local_map_sub, gimbal_cmd_sub, shot_sub;

ros::Timer local_sensing_timer, estimation_timer, target_render_timer;

bool has_global_map(false);
bool has_local_map(false);
bool has_odom(false);
bool has_target(false);

Matrix4d cam02body;
Matrix4d cam2world;
Eigen::Quaterniond cam2world_quat;
nav_msgs::Odometry _odom;

// gimbal camera
Matrix3d Rbc_theta, Rbc1, Rbw1;
Vector3d tbc1, twb1, pw_center1;

Vector3d target_pos, target_vel;

double sensing_horizon, sensing_rate, estimation_rate; 
double _x_size, _y_size, _z_size;
double _gl_xl, _gl_yl, _gl_zl;
double _resolution, _inv_resolution;
int _GLX_SIZE, _GLY_SIZE, _GLZ_SIZE;

ros::Time last_odom_stamp = ros::TIME_MAX;
Eigen::Vector3d last_pose_world; 

bool is_tracker = false;
bool in_tracking = false;
bool has_trigger = false;
int track_success = 0, track_total_time = 0, total_pixel_x = 0, total_pixel_y = 0;

void render_currentpose();
void render_pcl_world();

inline Eigen::Vector3d gridIndex2coord(const Eigen::Vector3i & index) 
{
    Eigen::Vector3d pt;
    pt(0) = ((double)index(0) + 0.5) * _resolution + _gl_xl;
    pt(1) = ((double)index(1) + 0.5) * _resolution + _gl_yl;
    pt(2) = ((double)index(2) + 0.5) * _resolution + _gl_zl;

    return pt;
};

inline Eigen::Vector3i coord2gridIndex(const Eigen::Vector3d & pt)
{
    Eigen::Vector3i idx;
    idx(0) = std::min( std::max( int( (pt(0) - _gl_xl) * _inv_resolution), 0), _GLX_SIZE - 1);
    idx(1) = std::min( std::max( int( (pt(1) - _gl_yl) * _inv_resolution), 0), _GLY_SIZE - 1);
    idx(2) = std::min( std::max( int( (pt(2) - _gl_zl) * _inv_resolution), 0), _GLZ_SIZE - 1);              
  
    return idx;
};

void rcvOdometryCallbck(const nav_msgs::Odometry& odom)
{
  /*if(!has_global_map)
    return;*/
  has_odom = true;
  _odom = odom;
  Matrix4d Pose_receive = Matrix4d::Identity();

  Eigen::Vector3d request_position;
  Eigen::Quaterniond request_pose;
  request_position.x() = odom.pose.pose.position.x;
  request_position.y() = odom.pose.pose.position.y;
  request_position.z() = odom.pose.pose.position.z;
  request_pose.x() = odom.pose.pose.orientation.x;
  request_pose.y() = odom.pose.pose.orientation.y;
  request_pose.z() = odom.pose.pose.orientation.z;
  request_pose.w() = odom.pose.pose.orientation.w;
  Pose_receive.block<3,3>(0,0) = request_pose.toRotationMatrix();
  Pose_receive(0,3) = request_position(0);
  Pose_receive(1,3) = request_position(1);
  Pose_receive(2,3) = request_position(2);

  Matrix4d body_pose = Pose_receive;
  //convert to cam pose
//  cout << "[pcl_render_node]: drone_pos " << request_position.transpose() << endl;
//  cout << "[pcl_render_node]: drone R matrix " << request_pose.toRotationMatrix() << endl;
  cam2world = body_pose * cam02body;
  cam2world_quat = cam2world.block<3,3>(0,0);

  last_odom_stamp = odom.header.stamp;

  last_pose_world(0) = odom.pose.pose.position.x;
  last_pose_world(1) = odom.pose.pose.position.y;
  last_pose_world(2) = odom.pose.pose.position.z;

  Rbw1 = request_pose.toRotationMatrix().inverse();
  twb1 = request_position;
  //publish tf
  /*static tf::TransformBroadcaster br;
  tf::Transform transform;
  transform.setOrigin( tf::Vector3(cam2world(0,3), cam2world(1,3), cam2world(2,3) ));
  transform.setRotation(tf::Quaternion(cam2world_quat.x(), cam2world_quat.y(), cam2world_quat.z(), cam2world_quat.w()));
  br.sendTransform(tf::StampedTransform(transform, last_odom_stamp, "world", "camera")); //publish transform from world frame to quadrotor frame.*/
}

void rcvTargetOdomCallback(const nav_msgs::Odometry& odom)
{
  has_target = true;
  target_pos(0) = odom.pose.pose.position.x;
  target_pos(1) = odom.pose.pose.position.y;
  target_pos(2) = odom.pose.pose.position.z;
  target_vel(0) = odom.twist.twist.linear.x;
  target_vel(1) = odom.twist.twist.linear.y;
  target_vel(2) = odom.twist.twist.linear.z;
  double v = target_vel.norm();
  double v_thresh = 0.01;
  
  pw_center1 = target_pos;
}

void rcvTriggerCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{
    //clear all data
    has_trigger = true;
    track_success = 0;
    track_total_time = 0;
    total_pixel_x = 0;
    total_pixel_y = 0;
}

void rcvGimbalCallback(const quadrotor_msgs::GimbalStateConstPtr& msg)
{
    double theta = msg->angle;
    Matrix3d R_theta = AngleAxisd(theta, Vector3d::UnitZ()).toRotationMatrix();
    // std::cout << "theta: " << theta << std::endl;
    // std::cout << "R_theta: " << R_theta << std::endl;
    Eigen::Quaterniond q_theta(R_theta);
    // std::cout << "q_theta: wxyz" << q_theta.w() << " " << q_theta.x() << " " << q_theta.y() << " " << q_theta.z() << std::endl;  
    Rbc_theta = R_theta * Rbc1;
}

void pubCameraPose(const ros::TimerEvent & event)
{ 
  //cout<<"pub cam pose"
  geometry_msgs::PoseStamped camera_pose;
  camera_pose.header = _odom.header;
  camera_pose.header.frame_id = "/map";
  camera_pose.pose.position.x = cam2world(0,3);
  camera_pose.pose.position.y = cam2world(1,3);
  camera_pose.pose.position.z = cam2world(2,3);
  camera_pose.pose.orientation.w = cam2world_quat.w();
  camera_pose.pose.orientation.x = cam2world_quat.x();
  camera_pose.pose.orientation.y = cam2world_quat.y();
  camera_pose.pose.orientation.z = cam2world_quat.z();
  pub_pose.publish(camera_pose);
}

void renderSensedPoints(const ros::TimerEvent & event)
{ 
  //if(! has_global_map || ! has_odom) return;
  if( !has_global_map && !has_local_map) return;
  if( !has_odom ) return;
  render_currentpose();
  render_pcl_world();
}

vector<float> cloud_data;
void rcvGlobalPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map )
{
  if(has_global_map)
    return;

  ROS_WARN("Global Pointcloud received..");
  //load global map
  pcl::PointCloud<pcl::PointXYZ> cloudIn;
  pcl::PointXYZ pt_in;
  //transform map to point cloud format
  pcl::fromROSMsg(pointcloud_map, cloudIn);
  for(int i = 0; i < int(cloudIn.points.size()); i++){
    pt_in = cloudIn.points[i];
    cloud_data.push_back(pt_in.x);
    cloud_data.push_back(pt_in.y);
    cloud_data.push_back(pt_in.z);
  }
  printf("global map has points: %d.\n", (int)cloud_data.size() / 3 );
  //pass cloud_data to depth render
  depthrender.set_data(cloud_data);
  depth_hostptr = (int*) malloc(width * height * sizeof(int));

  has_global_map = true;
}

void rcvLocalPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map )
{
  //ROS_WARN("Local Pointcloud received..");
  //load local map
  pcl::PointCloud<pcl::PointXYZ> cloudIn;
  pcl::PointXYZ pt_in;
  //transform map to point cloud format
  pcl::fromROSMsg(pointcloud_map, cloudIn);

  if(cloudIn.points.size() == 0) return;
  for(int i = 0; i < int(cloudIn.points.size()); i++){
    pt_in = cloudIn.points[i];
    Eigen::Vector3d pose_pt(pt_in.x, pt_in.y, pt_in.z);
    //pose_pt = gridIndex2coord(coord2gridIndex(pose_pt));
    cloud_data.push_back(pose_pt(0));
    cloud_data.push_back(pose_pt(1));
    cloud_data.push_back(pose_pt(2));
  }
  //printf("local map has points: %d.\n", (int)cloud_data.size() / 3 );
  //pass cloud_data to depth render
  depthrender.set_data(cloud_data);
  depth_hostptr = (int*) malloc(width * height * sizeof(int));

  has_local_map = true;
}

void render_pcl_world()
{
  //for debug purpose
  pcl::PointCloud<pcl::PointXYZ> localMap;
  pcl::PointXYZ pt_in;

  Eigen::Vector4d pose_in_camera;
  Eigen::Vector4d pose_in_world;
  Eigen::Vector3d pose_pt;

  for(int u = 0; u < width; u++)
    for(int v = 0; v < height; v++){
      float depth = depth_mat.at<float>(v,u);
      
      if(depth == 0.0)
        continue;

      pose_in_camera(0) = (u - cx) * depth / fx;
      pose_in_camera(1) = (v - cy) * depth / fy;
      pose_in_camera(2) = depth; 
      pose_in_camera(3) = 1.0;
      
      pose_in_world = cam2world * pose_in_camera;

      if( (pose_in_world.segment(0,3) - last_pose_world).norm() > sensing_horizon )
          continue; 

      pose_pt = pose_in_world.head(3);
      //pose_pt = gridIndex2coord(coord2gridIndex(pose_pt));
      pt_in.x = pose_pt(0);
      pt_in.y = pose_pt(1);
      pt_in.z = pose_pt(2);

      localMap.points.push_back(pt_in);
    }

  localMap.width = localMap.points.size();
  localMap.height = 1;
  localMap.is_dense = true;

  pcl::toROSMsg(localMap, local_map_pcl);
  local_map_pcl.header.frame_id  = "/map";
  local_map_pcl.header.stamp     = last_odom_stamp;

  pub_pcl_wolrd.publish(local_map_pcl);
}

void render_currentpose()
{
  double this_time = ros::Time::now().toSec();

  Matrix4d cam_pose = cam2world.inverse();

  double pose[4 * 4];

  for(int i = 0; i < 4; i ++)
    for(int j = 0; j < 4; j ++)
      pose[j + 4 * i] = cam_pose(i, j);

  depthrender.render_pose(pose, depth_hostptr);
  //depthrender.render_pose(cam_pose, depth_hostptr);

  depth_mat = cv::Mat::zeros(height, width, CV_32FC1);
  double min = 0.5;
  double max = 1.0f;
  for(int i = 0; i < height; i++)
  	for(int j = 0; j < width; j++)
  	{
  		float depth = (float)depth_hostptr[i * width + j] / 1000.0f;
  		depth = depth < 500.0f ? depth : 0;
  		max = depth > max ? depth : max;
  		depth_mat.at<float>(i,j) = depth;
  	}

  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = last_odom_stamp;
  out_msg.header.frame_id = "camera";
  out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
//  out_msg.encoding = sensor_msgs::image_encodings::RGB8;
  out_msg.image = depth_mat.clone();
  pub_depth.publish(out_msg.toImageMsg());

  cv::Mat adjMap;
  // depth_mat.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);
  depth_mat.convertTo(adjMap,CV_8UC3, 255 /13.0, -min);
  cv::Mat falseColorsMap;
  cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
  cv_bridge::CvImage cv_image_colored;
  cv_image_colored.header.frame_id = "depthmap";
  cv_image_colored.header.stamp = last_odom_stamp;
  cv_image_colored.encoding = sensor_msgs::image_encodings::BGR8;
  cv_image_colored.image = falseColorsMap;
  pub_color.publish(cv_image_colored.toImageMsg());
  //cv::imshow("depth_image", adjMap);
}

void shot_callback(const quadrotor_msgs::ShotParams::ConstPtr& msgPtr)
  {
      shotPtr_ -> loadmsg(*msgPtr);
  }

void targetRenderCallback(const ros::TimerEvent& event)
{
    if( !has_global_map && !has_local_map) return;
    if(!has_target || !has_odom)
    {
        return;
    }

    //benchmark
    static int success_time, total_time, block_time, out_fov_time;
    static double ave_err_dis, ave_err_angle, ave_err_image;

    // best view point
    shot::ShotConfig config;
    shotPtr_ -> generate(config);
    Eigen::Vector3d best_view_p, vec;
    vec << config.distance * cos(config.view_angle), config.distance * sin(config.view_angle), 0.0;
    best_view_p = target_pos + vec;
    nav_msgs::Odometry best_v_odom = _odom;
    best_v_odom.pose.pose.position.x = best_view_p.x();
    best_v_odom.pose.pose.position.y = best_view_p.y();
    best_v_odom.pose.pose.position.z = best_view_p.z();
    pub_best_view_p.publish(best_v_odom);

    // double this_time = ros::Time::now().toSec();
    
    Matrix4d cam_pose, Twb, Tbc;
    // TODO: theta
    Tbc.setZero();
    Tbc.block<3,3>(0,0) = Rbc_theta;
    Tbc.block<3,1>(0,3) = tbc1;
    Tbc(3,3) = 1.0;
    Twb.setZero();
    Twb.block<3,3>(0,0) = Rbw1.transpose();
    Twb.block<3,1>(0,3) = twb1;
    Twb(3,3) = 1.0;
    cam_pose = (Twb * Tbc).inverse();
    // cout << "my : " << "cam_pose^-1:\n" << Twb * Tbc << "body_pose:\n" << Twb
    //     << "cam02body:\n" << Tbc << endl;

    double pose[4 * 4];
    for(int i = 0; i < 4; i ++)
        for(int j = 0; j < 4; j ++)
        pose[j + 4 * i] = cam_pose(i, j);
    
    depthrender.render_pose(pose, depth_hostptr);
    depth_mat = cv::Mat::zeros(height, width, CV_32FC1);
    double min = 0.5;
    double max = 1.0f;
    for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++)
        {
            float depth = (float)depth_hostptr[i * width + j] / 1000.0f;
            depth = depth < 500.0f ? depth : 0;
            max = depth > max ? depth : max;
            depth_mat.at<float>(i,j) = depth;
  	}
    bool show_target = true;
    int target_x, target_y;
    int write_x, write_y;
    // -1 out of fov
    // -2 occlusion
    double radius;
    Eigen::Vector4d point_world, point_cam;
    point_world.head(3) = target_pos;
    point_world(3) = 1.0;
    point_cam = cam_pose * point_world;
    if(point_cam(2) <= 0.0)
    {
        show_target = false;
        write_x = -1;
        write_y = -1;
    }
    target_x = point_cam(0) / point_cam(2) * fx + cx + 0.5;
    target_y = point_cam(1) / point_cam(2) * fy + cy + 0.5;
    if(target_x < 0 || target_x >= width || target_y < 0 || target_y >= height)
    {
        show_target = false;
        write_x = -1;
        write_y = -1;
    }
    else
    {
        float depth = point_cam(2);
        depth = (depth * 1000.0f + 0.5f) / 1000.0f;
        radius = 0.33 * fx / depth; // for 250 drone
        float depth_obs = depth_mat.at<float>(target_y,target_x);
        if(depth >= depth_obs && depth_obs >= 0.0001)
        {
            show_target = false;
            write_x = -2;
            write_y = -2;
        }
    }
  // show target in the depth image
    if(show_target)
    {
    //      cout << "drawing drone radius:" << radius << endl;
        cv::circle(depth_mat, cv::Point(target_x, target_y), radius, cv::Scalar(0, 0, 255), cv::FILLED);
        write_x = target_x;
        write_y = target_y;
    }
    cv::Mat adjMap;
    // depth_mat.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);
    depth_mat.convertTo(adjMap,CV_8UC3, 255 /13.0, -min);
    cv::Mat falseColorsMap;
    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
    if(show_target)  //for better visualization
    {
        cv::circle(falseColorsMap, cv::Point(target_x, target_y), radius, cv::Scalar(255, 255, 255), cv::FILLED);
    }
    cv_bridge::CvImage cv_image_colored;
    cv_image_colored.header.frame_id = "usb_camera";
    cv_image_colored.header.stamp = last_odom_stamp;
    cv_image_colored.encoding = sensor_msgs::image_encodings::BGR8;
    cv_image_colored.image = falseColorsMap;
    pub_gimbal.publish(cv_image_colored.toImageMsg());

    if( has_trigger)
    {
        static double time_begin = ros::Time::now().toSec();
        // ofstream outfile("/home/zhiwei/benchmark_ws/data/af.txt",ios::app);
        double time_w = ros::Time::now().toSec();
        double image_err_print;
        if(show_target)
        {
            Eigen::Vector2d show_pos(write_x, write_y);
            image_err_print = (show_pos - config.image_p).norm();
        }
        else
        {
            image_err_print = 0.0;
        }
        // outfile << time_w - time_begin << " " << image_err_print << endl;
        // outfile.close();

        // benchmark
        total_time ++;
        if(!show_target)
        {
            if(write_x == -1)
                out_fov_time ++;
            else if(write_x == -2)
                block_time ++;
        }
        if(show_target)
        {
            Eigen::Vector3d pos, dp;
            pos << _odom.pose.pose.position.x, _odom.pose.pose.position.y, _odom.pose.pose.position.z;
            dp = pos - target_pos;
            double angle_err = std::fabs(std::atan2(dp(1), dp(0)) - config.view_angle);
            angle_err = angle_err > M_PI ? (2 * M_PI - angle_err) : angle_err;
            ave_err_angle = (ave_err_angle * success_time + angle_err) / (success_time + 1);
            double dis_err = std::fabs(dp.norm() - config.distance);
            ave_err_dis = (ave_err_dis * success_time + dis_err) / (success_time + 1);
            Eigen::Vector2d show_pos(write_x, write_y);
            double image_err = (show_pos - config.image_p).norm();
            ave_err_image = (ave_err_image * success_time + image_err) / (success_time + 1);
            success_time ++;
            // std:: cout << "config: " << config.view_angle << " d " <<  config.distance << " i " << config.image_p.transpose() << std::endl;
            // std::cout << "show_pos: " << show_pos.transpose() << std::endl;
            // std::cout << "angle_err: " << angle_err << " dis_err: " << dis_err << " image_err: " << image_err << std::endl;
        }
        // std::cout << "\033[32m" << "success: " << double(success_time)/total_time << " ave_err_image: " << ave_err_image << "\033[0m" << std::endl;
        // std::cout << "\033[32m" <<  "ave_err_image: " << ave_err_image << "\033[0m" << std::endl;
        // std::cout << "\033[32m"<< "[ls]total_time: " << total_time << " out fov: " << double(out_fov_time)/total_time *100.0 << "\033[0m" << std::endl;
    }


}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pcl_render");
  ros::NodeHandle nh("~");

   //benchmark
  // std::remove("/home/zhiwei/benchmark_ws/data/af.txt");

  nh.getParam("cam_width", width);
  nh.getParam("cam_height", height);
  nh.getParam("cam_fx", fx);
  nh.getParam("cam_fy", fy);
  nh.getParam("cam_cx", cx);
  nh.getParam("cam_cy", cy);
  nh.getParam("sensing_horizon", sensing_horizon);
  nh.getParam("sensing_rate",    sensing_rate);
  nh.getParam("estimation_rate", estimation_rate);
  nh.getParam("map/x_size",     _x_size);
  nh.getParam("map/y_size",     _y_size);
  nh.getParam("map/z_size",     _z_size);
  nh.getParam("is_tracker_", is_tracker);

  depthrender.set_para(fx, fy, cx, cy, width, height);

  // cam02body <<  0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
  //               0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
  //               -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
  //               0.0, 0.0, 0.0, 1.0;

  cam02body << 0.0, 0.0, 1.0, 0.0,
              -1.0, 0.0, 0.0, 0.0,
               0.0, -1.0,0.0, 0.0,
               0.0, 0.0, 0.0, 1.0;

  Rbc1 << 0.0, 0.0, 1.0,
                -1.0, 0.0, 0.0,
                0.0, -1.0, 0.0;

  tbc1 << 0.0, 0.0, 0.0;
  Rbc_theta = Rbc1;

  //init cam2world transformation
  cam2world = Matrix4d::Identity();

  shotPtr_ = std::make_shared<shot::ShotGenerator>(nh);
  //subscribe point cloud
  global_map_sub = nh.subscribe( "global_map", 1,  rcvGlobalPointCloudCallBack); 
  local_map_sub  = nh.subscribe( "local_map",  1,  rcvLocalPointCloudCallBack);  
  odom_sub       = nh.subscribe( "odometry",  1, rcvOdometryCallbck,ros::TransportHints().tcpNoDelay());

  //publisher depth image and color image
  pub_depth = nh.advertise<sensor_msgs::Image>("depth",1000);
  pub_color = nh.advertise<sensor_msgs::Image>("colordepth",1000);
  pub_pose  = nh.advertise<geometry_msgs::PoseStamped>("camera_pose",1000);
  pub_pcl_wolrd = nh.advertise<sensor_msgs::PointCloud2>("rendered_pcl",1);

  double sensing_duration  = 1.0 / sensing_rate;
  double estimate_duration = 1.0 / estimation_rate;

  local_sensing_timer = nh.createTimer(ros::Duration(sensing_duration),  renderSensedPoints); 
  estimation_timer    = nh.createTimer(ros::Duration(estimate_duration), pubCameraPose);
  //cv::namedWindow("depth_image",1);

  if(is_tracker)
  {
    target_sub = nh.subscribe("target", 1, rcvTargetOdomCallback,ros::TransportHints().tcpNoDelay());
    trigger_sub = nh.subscribe("trigger", 5, rcvTriggerCallback);
    pub_gimbal = nh.advertise<sensor_msgs::Image>("usb_camera", 1000);
    target_render_timer = nh.createTimer(ros::Duration(0.05), targetRenderCallback);
    gimbal_cmd_sub = nh.subscribe("gimbal_state", 1, rcvGimbalCallback, ros::TransportHints().tcpNoDelay());
    shot_sub = nh.subscribe("shot", 10, shot_callback);
    pub_best_view_p = nh.advertise<nav_msgs::Odometry>("best_view_odom", 10);
  }  

  _inv_resolution = 1.0 / _resolution;

  _gl_xl = -_x_size/2.0;
  _gl_yl = -_y_size/2.0;
  _gl_zl =   0.0;
  
  _GLX_SIZE = (int)(_x_size * _inv_resolution);
  _GLY_SIZE = (int)(_y_size * _inv_resolution);
  _GLZ_SIZE = (int)(_z_size * _inv_resolution);

  ros::Rate rate(200);
  bool status = ros::ok();
  while(status) 
  {
    ros::spinOnce();  
    status = ros::ok();
    rate.sleep();
  } 
}
