cmake_minimum_required(VERSION 2.8.3)
project(gui)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  quadrotor_msgs
)

catkin_package(
  CATKIN_DEPENDS quadrotor_msgs
)