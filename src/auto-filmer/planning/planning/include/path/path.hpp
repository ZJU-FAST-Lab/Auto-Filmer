#pragma once

#include <mapping/mapping.h>
#include <queue>
#include <shot/shot.hpp>
#include <ros/ros.h>

#include <Eigen/Core>
// #include <queue>
#include <algorithm>
#include <unordered_map>
#include <stack>

namespace path {

enum State { OPEN,
             CLOSE,
             UNVISITED };
struct Node {
  int n;
  Eigen::Vector4i idx;  // no prune 
  Eigen::Vector3d p; // v, a;
  double cost, h;
  State state = UNVISITED;
  Node* parent = nullptr;
};
typedef Node* NodePtr;
class NodeComparator {
 public:
  bool operator()(NodePtr& lhs, NodePtr& rhs) {
    // return lhs->cost > rhs->cost ;
    return lhs->cost + lhs->h > rhs->cost + rhs->h;
  }
};
class PathSearch {
 private:
  static constexpr int MAX_MEMORY = 1 << 22;
  // searching

  int N, node_num;
  const int no_prune_n = 2;
  double dt;
  double dur;
  double vmax, amax, input_max; 
  double v_res;
  double clearance_d_, cy_, fy_;
  std::vector<Eigen::Vector3d> target_ps_;
  std::shared_ptr<mapping::OccGridMap> mapPtr_;
  NodePtr data[MAX_MEMORY];
  double lambda_cost, lambda_theta, lambda_z;
  std::unordered_map<Eigen::Vector4i, NodePtr> visited_nodes_;  

  inline NodePtr visit(const Eigen::Vector3d& p, const int& n)
  {
      if(n <= no_prune_n)
      {
          NodePtr ptr= data[node_num++];
          ptr->p = p;
          ptr->n = n;
          ptr->idx = Eigen::Vector4i::Zero();
          bool valid = checkValid(mapPtr_->pos2idx(p), n);
          ptr->state = valid? UNVISITED : CLOSE;
          return ptr;
      }
      else
      {
          Eigen::Vector4i idx;
          idx << mapPtr_->pos2idx(p), n;
          auto iter = visited_nodes_.find(idx);
          if(iter == visited_nodes_.end())
          {
              NodePtr ptr = data[node_num++];
              ptr->idx =idx;
              ptr->p = p;
              ptr->n = n;
              bool valid = checkValid(mapPtr_->pos2idx(p), n);
              ptr->state = valid? UNVISITED : CLOSE;
              visited_nodes_[idx] = ptr;
              return ptr;
          }
          else
          {
              return iter->second;
          }
      }
  }

  inline const Eigen::Vector3i vel2idx(const Eigen::Vector3d& v)
  {
      Eigen::Vector3i ret;
      for(int i = 0; i < v.size(); i ++)
      {
        if (v(i) > v_res)
            ret(i) = 1;
        else if (v(i) < -v_res)
            ret(i) = -1;
        else
            ret(i) = 0;
      }
      return ret;
    // return Eigen::Vector3i::Zero();
  }  

  bool inline checkValid(const Eigen::Vector3i& idx, const int& n )
  {
      Eigen::Vector3d p = mapPtr_->idx2pos(idx);
      Eigen::Vector3d target = target_ps_[n];
      bool valid = checkRayValid(p, target);
      // region check visibility
      // if(valid)
      // {
      //   Eigen::Vector3d dp = p - target;
      //   static constexpr double max_arc_length = 0.45;  // avoid exceed the region
      //   double d_angle = max_arc_length / dp.norm();  
      //   Eigen::Vector3d p1, p2;
      //   p1 = target + Eigen::AngleAxisd(d_angle, Eigen::Vector3d::UnitZ()) * dp;
      //   p2 = target + Eigen::AngleAxisd(-d_angle, Eigen::Vector3d::UnitZ()) * dp;
      //   valid = checkRayValid(p1, target) && checkRayValid(p2, target);
      // }
      return valid;

  }

  bool inline checkRayValid(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1) {
    Eigen::Vector3d dp = p1 - p0;
    if(dp.norm() < 1.5)
    {
      return false;
    }
    Eigen::Vector3i idx0 = mapPtr_->pos2idx(p0);
    Eigen::Vector3i idx1 = mapPtr_->pos2idx(p1);
    Eigen::Vector3i d_idx = idx1 - idx0;
    Eigen::Vector3i step = d_idx.array().sign().cast<int>();
    Eigen::Vector3d delta_t;
    for (int i = 0; i < 3; ++i) {
      delta_t(i) = dp(i) == 0 ? std::numeric_limits<double>::max() : 1.0 / std::fabs(dp(i));
    }
    Eigen::Vector3d t_max;
    for (int i = 0; i < 3; ++i) {
      t_max(i) = step(i) > 0 ? (idx0(i) + 1) - p0(i) / mapPtr_->resolution : p0(i) / mapPtr_->resolution - idx0(i);
    }
    t_max = t_max.cwiseProduct(delta_t);
    Eigen::Vector3i rayIdx = idx0;
    while ((rayIdx - idx1).squaredNorm() > 1) {
      if (mapPtr_->isOccupied(rayIdx)) {
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

 public:
  PathSearch(ros::NodeHandle& nh, 
        std::shared_ptr<mapping::OccGridMap>& mapPtr) : mapPtr_(mapPtr) {
    nh.getParam("tracking_dur", dur);
    nh.getParam("tracking_dt", dt);
    nh.getParam("vmax", vmax);
    v_res =  vmax / 3.0;  // NOTE: better from vmax
    nh.getParam("amax", amax);
    input_max = vmax * 0.6;  //sqrt(3)/3
    nh.getParam("path/lambda_cost", lambda_cost);
    nh.getParam("path/lambda_theta", lambda_theta);
    nh.getParam("path/lambda_z", lambda_z);
    nh.getParam("cam_cy", cy_);
    nh.getParam("cam_fy", fy_);
    nh.getParam("clearance_d", clearance_d_);
    for (int i = 0; i < MAX_MEMORY; ++i) {
      data[i] = new Node;
    }
  }

  ~PathSearch()
  {
    for (int i = 0; i < MAX_MEMORY; ++i) {
      delete data[i];
    }
  }

    inline double calcH(const int &n)
    {
        return (N - n);
    }

    inline double score(const Eigen::Vector3d& p, const Eigen::Vector3d& target, const double& target_yaw, const shot::ShotConfig& shotcfg)
    {
        double d, angle, des_d, des_angle, delta_angle, best_tilt, des_z;
        Eigen::Vector3d dp = p - target;
        // 1. distance
        des_d = shotcfg.distance;
        d = dp.norm();
        // 2. tilt
        best_tilt = atan((shotcfg.image_p(1) - cy_) / fy_);
        des_z = target.z() + shotcfg.distance * sin(best_tilt);
        // if ground limits, then best_z is clearance_d_
        // if(des_z < clearance_d_)
        //     des_z = clearance_d_;

        // des_tilt = asin((des_z - target.z()) / shotcfg.distance);
        // tilt = asin(dp(2) / d);
        // 3. pan
        angle = std::atan2(dp(1), dp(0));
        des_angle = shotcfg.view_angle + target_yaw;
        delta_angle = fabs(des_angle - angle);
        while(delta_angle > 2*M_PI)
            delta_angle -= 2*M_PI;
        delta_angle = std::min(delta_angle , 2*M_PI - delta_angle);
        return lambda_cost * (fabs(d - des_d) + lambda_theta * delta_angle + lambda_z * fabs(p.z() - des_z)); 
    }

    inline bool findPath(const std::vector<Eigen::Vector3d> target_ps,
                        const std::vector<double> target_headings,
                        const std::vector<shot::ShotConfig> shotconfigs,
                        const Eigen::Vector3d& init_pos,
                        const Eigen::Vector3d& init_vel,
                        std::vector<Eigen::Vector3d>& visible_ps,
                        const double&  max_time = 0.2)
    {
        int visit_n = 0;
        node_num = 0;
        N = target_ps.size();
        ros::Time t_start = ros::Time::now();
        std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator> open_set;
        Eigen::Vector3d input;
        target_ps_ = target_ps;
        visited_nodes_.clear();

        NodePtr curPtr = visit(init_pos, 0);
        curPtr->state = CLOSE;
        curPtr->parent = nullptr;
        curPtr->cost = 0;  
        curPtr->h = 0;
        if(!checkValid(mapPtr_->pos2idx(init_pos), 0)) // || (init_vel.norm() > vmax))
        {
            std::cout << "[path search] start node invalid, may cause failure !" << std::endl;
        }
        while(curPtr->n < N - 1)
        {
            visit_n++;  // debug
            Eigen::Vector3d target_parent = target_ps_[curPtr->n];
            double target_yaw_parent = target_headings[curPtr->n];
            shot::ShotConfig cfg_parent = shotconfigs[curPtr->n];

            int n = curPtr->n + 1;
            Eigen::Vector3d target = target_ps_[n];
            double target_yaw = target_headings[n];
            shot::ShotConfig cfg = shotconfigs[n];
            
            for (input(0) = -input_max; input(0) <= input_max; input(0) += input_max)
            {
                for (input(1) = -input_max; input(1) <= input_max; input(1) += input_max) 
                {
                    for(input(2) = -input_max; input(2) <= input_max; input(2) += input_max)
                    {
                        Eigen::Vector3d p;  // v;
                        p = curPtr->p + input * dt;

                        bool valid = true;
                        // path check
                        int check_sum = 2;
                        double temp_t;
                        Eigen::Vector3d temp_p;
                        for(int check_n = 1; check_n < check_sum;  check_n++)
                        {
                            temp_t = dt * check_n / check_sum;
                            temp_p = curPtr->p + input * temp_t;
                            if(mapPtr_->isOccupied(temp_p))
                                valid = false;
                        }
                        if(!valid)
                        {
                            continue;
                        }
                        NodePtr ptr = visit(p, n);
                        if(ptr->state == UNVISITED)
                        {
                            ptr->parent = curPtr;
                            // ptr->cost = curPtr->cost + score(p, target, target_yaw, cfg);
                            ptr->cost = curPtr->cost + score(p, target, target_yaw, cfg) - score(curPtr->p, target_parent, target_yaw_parent, cfg_parent); //shooting lost
                            ptr->state = OPEN;
                            ptr->h = calcH(ptr->n);
                            open_set.push(ptr);
                        }
                        else if(ptr->state == OPEN)
                        {
                            // if(ptr->cost > curPtr->cost + score(p, target, target_yaw, cfg))
                            if(ptr->cost > curPtr->cost + score(p, target, target_yaw, cfg) - score(curPtr->p, target_parent, target_yaw_parent, cfg_parent))
                            {
                                ptr->p = p;
                                // ptr->cost = curPtr->cost + score(p, target, target_yaw, cfg);
                                ptr->cost = curPtr->cost + score(p, target, target_yaw, cfg) - score(curPtr->p, target_parent, target_yaw_parent, cfg_parent);
                                ptr->h = calcH(ptr->n);
                                ptr->parent = curPtr;
                            }
                        }
                        else if(ptr->state == CLOSE)
                        {
                            continue;
                        }
                        
                    }
                }
            } // end for

            if(open_set.empty())
            {
                ROS_ERROR("[path search] no way!visit nodes: %d", visit_n);
                return false;
            }
            if (visited_nodes_.size() >= MAX_MEMORY) {
                ROS_ERROR("[path search] out of memory! visit nodes: %d", visit_n);
                return false;
            }
            double t_cost = (ros::Time::now() - t_start).toSec();
            if (t_cost > max_time) {
                std::cout << "curPtr->n: " << curPtr->n << std::endl;
                ROS_ERROR("[path search] too slow!visit nodes: %d", visit_n);
                return false;
            }

            curPtr = open_set.top();
            open_set.pop();
            curPtr->state = CLOSE;
        }  // end while

        // path
        for (NodePtr ptr = curPtr; ptr != nullptr; ptr = ptr->parent) {
            visible_ps.push_back(ptr->p);
        }

        std::reverse(visible_ps.begin(), visible_ps.end());
        // std::cout << "[path search] success. node_num: " << visit_n << std::endl; 

        return true;
    }


};

}  // namespace path
