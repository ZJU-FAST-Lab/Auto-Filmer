cmake_minimum_required(VERSION 2.8.3)
project(gimbal_simulator)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  quadrotor_msgs
  nav_msgs
  visualization_msgs
)
find_package(Eigen3 REQUIRED)

catkin_package(
    CATKIN_DEPENDS quadrotor_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable (gimbal_simulator src/fake_gimbal.cpp )
target_link_libraries(gimbal_simulator
  ${catkin_LIBRARIES})
