#pragma once

#include <cmath>
#include <ros/ros.h>
#include <quadrotor_msgs/ShotParams.h>
#include <Eigen/Core>

namespace shot {

struct  ShotConfig{
  Eigen::Vector2d image_p, image_v;
  double distance;
  double view_angle;
};

enum state
{
    STATIC, TRANSITION
};

struct ShotGenerator {
    private:

    bool rcv_msg;
    double cx_, cy_, tracking_dt_;
    double min_dis_, max_dis_;
    ShotConfig start_config_, goal_config_;
    state state_image, state_distance, state_view;
    Eigen::Vector2d time_image, time_distance, time_view;  // 0: start 1: duration
    const double IMG_THR = 5;
    const double DIS_THR = 0.1; 
    const double ANGLE_THR = 0.05; 

    public:
    inline ShotGenerator(ros::NodeHandle& nh)
    {
        nh.getParam("cam_cx", cx_);
        nh.getParam("cam_cy", cy_);
        nh.getParam("min_dis", min_dis_);
        nh.getParam("max_dis", max_dis_);
        nh.getParam("tracking_dt", tracking_dt_);
        rcv_msg = false;

        //default
        start_config_.image_p << cx_, cy_;
        start_config_.image_v.setZero();
        start_config_.distance = (min_dis_ + max_dis_)/2;
        // important! modify this before fly
        start_config_.view_angle = 0.0; // 3.14;  
        // start_config_.view_angle = 1.6039059469848036;
        state_image = STATIC;
        state_distance = STATIC;
        state_view = STATIC;
        time_image << 0.0, 0.0;
        time_distance << 0.0, 0.0;
        time_view << 0.0, 0.0;
        goal_config_ = start_config_;
    }

    inline void loadmsg(const quadrotor_msgs::ShotParams& shot_msg_)
    {
        double msg_transition, msg_dis, msg_angle;
        Eigen::Vector2d msg_pos_image;
        msg_pos_image << shot_msg_.image_x, shot_msg_.image_y;
        msg_dis = shot_msg_.distance;
        msg_angle = shot_msg_.theta;
        msg_transition = shot_msg_.transition;
        // double rcv_time = shot_msg_.time.toSec();
        double rcv_time = ros::Time::now().toSec();
        double now_time = ros::Time::now().toSec();
        // image
        if((goal_config_.image_p - msg_pos_image).norm() > IMG_THR)
        {
            if(state_image == STATIC)
            {
                time_image(0) = rcv_time;
                time_image(1) = msg_transition;
                goal_config_.image_p = msg_pos_image;
                state_image = TRANSITION;
            }
            else  //start a new transition
            {
                double time_from_start = now_time  - time_image(0);
                double ratio = time_from_start / time_image(1);
                start_config_.image_p = start_config_.image_p + ratio * (goal_config_.image_p - start_config_.image_p);
                time_image(0) = rcv_time;
                time_image(1) = msg_transition;
                goal_config_.image_p = msg_pos_image;
            }
        }

        // distance
        if(fabs(goal_config_.distance - msg_dis) > DIS_THR)
        {
            if(state_distance == STATIC)
            {
                time_distance(0) = rcv_time;
                time_distance(1) = msg_transition;
                goal_config_.distance = msg_dis;
                state_distance = TRANSITION;
            }
            else  //start a new transition
            {
                double time_from_start = now_time  - time_distance(0);
                double ratio = time_from_start / time_distance(1);
                start_config_.distance = start_config_.distance + ratio * (goal_config_.distance - start_config_.distance);
                time_distance(0) = rcv_time;
                time_distance(1) = msg_transition;
                goal_config_.distance = msg_dis;
            }
        }
        // view
        double d_angle = goal_config_.view_angle > msg_angle ?
                                         msg_angle + 2*M_PI - goal_config_.view_angle :
                                         goal_config_.view_angle + 2*M_PI - msg_angle;
        d_angle = std::min(std::fabs(goal_config_.view_angle - msg_angle), d_angle);
        if(d_angle > ANGLE_THR)
        {
            if(state_view == STATIC)
            {
                time_view(0) = rcv_time;
                time_view(1) = msg_transition;
                goal_config_.view_angle = msg_angle;
                state_view = TRANSITION;
            }
            else  //start a new transition
            {
                double time_from_start = now_time  - time_view(0);
                double ratio = time_from_start / time_view(1);
                double goal_view = goal_config_.view_angle;
                int direction;
                calcBestView(start_config_.view_angle, goal_view, direction);
                double angle  = start_config_.view_angle + ratio * (goal_view - start_config_.view_angle);
                start_config_.view_angle  = refineAngle(angle);
                time_view(0) = rcv_time;
                time_view(1) = msg_transition;
                goal_config_.view_angle = msg_angle;
            }
        }

    }

    inline void generate(std::vector<ShotConfig>& config_vec, const int& n)
    {
        // TODO: maybe generate from present drone and target state?
        config_vec.resize(n);
        // check whether exceeds duration
        double now_time = ros::Time::now().toSec();
        if(now_time - time_image(0) > time_image(1) && state_image == TRANSITION  )
        {
            start_config_.image_p = goal_config_.image_p;
            state_image = STATIC;
        }
        if(now_time - time_distance(0) > time_distance(1) && state_distance == TRANSITION)
        {
            start_config_.distance = goal_config_.distance;
            state_distance = STATIC;
        }
        if(now_time - time_view(0) > time_view(1) && state_view == TRANSITION)
        {
            start_config_.view_angle = goal_config_.view_angle;
            state_view = STATIC;
        }
        // assign parameters
        for(int i = 0; i < n; i ++)
        {
            ShotConfig config;
            // image
            if(state_image == STATIC)
            {
                config.image_p = start_config_.image_p;
                config.image_v.setZero();
            }
            else
            {
                double time_from_start = now_time + i * tracking_dt_ - time_image(0);
                if(time_from_start < time_image(1))
                {
                    double ratio = time_from_start / time_image(1);
                    config.image_p = start_config_.image_p + ratio * (goal_config_.image_p - start_config_.image_p);
                    config.image_v = (goal_config_.image_p - start_config_.image_p) / time_image(1);
                }
                else
                { 
                    config.image_p = goal_config_.image_p;
                    config.image_v.setZero();
                }
            }
            // distance
            if(state_distance == STATIC)
            {
                config.distance = start_config_.distance;
            }
            else
            {
                double time_from_start = now_time + i * tracking_dt_ - time_distance(0);
                if(time_from_start < time_distance(1))
                {
                    double ratio = time_from_start / time_distance(1);
                    config.distance = start_config_.distance + ratio * (goal_config_.distance - start_config_.distance);
                }
                else
                { 
                    config.distance = goal_config_.distance;
                }
            }
            //view
            if(state_view == STATIC)
            {
                config.view_angle = start_config_.view_angle;
            }
            else
            {
                double time_from_start = now_time + i * tracking_dt_ - time_view(0);
                if(time_from_start < time_view(1))
                {
                    double ratio = time_from_start / time_view(1);
                    double goal_view = goal_config_.view_angle;  // -pi to pi
                    int direction;
                    calcBestView(start_config_.view_angle, goal_view, direction);
                    double angle  = start_config_.view_angle + ratio * (goal_view - start_config_.view_angle);
                    config.view_angle  = refineAngle(angle);
                }
                else
                { 
                    config.view_angle= goal_config_.view_angle;
                }
            }
            // std::cout << "distance: " << config.distance << "\nimage: " << config.image_p.transpose() << " v: " << config.image_v << "\nangle: " << config.view_angle << std::endl; 
            config_vec[i] = config;
        }  //end for 
    
    }   

    inline double refineAngle(const double& theta)
    {
        double ret = theta;
        if(theta > M_PI)
            ret = theta - 2 * M_PI;
        else if(theta < -M_PI)
            ret = theta + 2 * M_PI;
        return ret;
    }

    // direction = 1: +, unclockwisw; direction = -1: -, clockwise
    inline void calcBestView(const double& start, double& goal, int& direction)
    {
        bool bigger = goal > start;
        double angle_add, angle_minus;
        if(bigger)
        {
            angle_add = goal - start;
            angle_minus = start + 2*M_PI - goal;
        }
        else{  // start > goal
            angle_add = goal + 2*M_PI - start;
            angle_minus = start - goal;
        }

        if(angle_add <= angle_minus)
        {
            direction = 1;
            goal = start + angle_add;
        }
        else
        {
            direction = -1;
            goal = start - angle_minus;
        }
    }

};  // end struct


}  // namespace shot
