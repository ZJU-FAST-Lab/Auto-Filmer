#pragma once
#include "decomp_geometry/polyhedron.h"
#include "decomp_ros_msgs/PolyhedronArray.h"
#include <decomp_ros_utils/data_ros_utils.h>
#include <decomp_util/ellipsoid_decomp.h>
#include <mapping/mapping.h>
#include <math.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>
#include <memory>
#include <queue>
#include <sys/types.h>
#include <traj_opt/geoutils.hpp>
#include <unordered_map>
#include <shot/shot.hpp>

namespace std {
template <typename Scalar, int Rows, int Cols>
struct hash<Eigen::Matrix<Scalar, Rows, Cols>> {
  size_t operator()(const Eigen::Matrix<Scalar, Rows, Cols>& matrix) const {
    size_t seed = 0;
    for (size_t i = 0; i < (size_t)matrix.size(); ++i) {
      Scalar elem = *(matrix.data() + i);
      seed ^=
          std::hash<Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
}  // namespace std

namespace env {

enum State { OPEN,
             CLOSE,
             UNVISITED };
struct Node {
  Eigen::Vector3i idx;
  bool valid = false;
  State state = UNVISITED;
  double g, h;
  Node* parent = nullptr;
};
typedef Node* NodePtr;
class NodeComparator {
 public:
  bool operator()(NodePtr& lhs, NodePtr& rhs) {
    return lhs->g + lhs->h > rhs->g + rhs->h;
  }
};

class Env {
  static constexpr int MAX_MEMORY = 1 << 18;
  static constexpr double MAX_DURATION = 0.2;

 private:
  ros::Publisher hPolyPub_, viewRegionPub_, dbgRegionPub_;
  ros::Time t_start_;

  std::unordered_map<Eigen::Vector3i, NodePtr> visited_nodes_;
  std::shared_ptr<mapping::OccGridMap> mapPtr_;
  NodePtr data_[MAX_MEMORY];
  double desired_dist_, alpha_clearance_, clearance_d_, tolerance_d_, kill_d_, tracking_dt_, ratemax_, cy_, fy_;
  double lambda_dist, lambda_theta;
  const double MAX = 10000.0;
  double last_estimate_orientation;

  inline NodePtr visit(const Eigen::Vector3i& idx) {
    auto iter = visited_nodes_.find(idx);
    if (iter == visited_nodes_.end()) {
      auto ptr = data_[visited_nodes_.size()];
      ptr->idx = idx;
      ptr->valid = !mapPtr_->isOccupied(idx);
      ptr->state = UNVISITED;
      visited_nodes_[idx] = ptr;
      return ptr;
    } else {
      return iter->second;
    }
  }

 public:
  Env(ros::NodeHandle& nh,
      std::shared_ptr<mapping::OccGridMap>& mapPtr) : mapPtr_(mapPtr) {
    hPolyPub_ = nh.advertise<decomp_ros_msgs::PolyhedronArray>("polyhedra", 1);
    viewRegionPub_ =nh.advertise<decomp_ros_msgs::PolyhedronArray>("view_region", 1);
    dbgRegionPub_ = nh.advertise<sensor_msgs::PointCloud2>("dbg_region", 1);
    nh.getParam("tracking_dist", desired_dist_);
    nh.getParam("tolerance_d", tolerance_d_);
    nh.getParam("ratemax", ratemax_);
    nh.getParam("alpha_clearance", alpha_clearance_);
    nh.getParam("clearance_d", clearance_d_);
    nh.getParam("tracking_dt", tracking_dt_);
    nh.getParam("path/lambda_cost", lambda_dist);
    nh.getParam("path/lambda_theta", lambda_theta);
    nh.getParam("kill_d", kill_d_);
    nh.getParam("cam_cy", cy_);
    nh.getParam("cam_fy", fy_);
    last_estimate_orientation = 0.0;
    for (int i = 0; i < MAX_MEMORY; ++i) {
      data_[i] = new Node;
    }
  }
  ~Env() {
    for (int i = 0; i < MAX_MEMORY; ++i) {
      delete data_[i];
    }
  }

  bool inline checkRayValid(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, double max_dist) const {
    Eigen::Vector3d dp = p1 - p0;
    double dist = dp.norm();
    if (dist > max_dist) {
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
  bool inline checkRayValid(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1) const {
    Eigen::Vector3d dp = p1 - p0;
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

  // get the occupied point, or the end point
  double inline getValidRay(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1) const {
    Eigen::Vector3d dp = p1 - p0;
    double dist = dp.norm();
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
    while ((rayIdx - idx1).squaredNorm() >= 1) {
      if (mapPtr_->isOccupied(rayIdx)) {
        return (mapPtr_->idx2pos(rayIdx) - p0).norm();
      }
      // find the shortest t_max
      int s_dim = 0;
      for (int i = 1; i < 3; ++i) {
        s_dim = t_max(i) < t_max(s_dim) ? i : s_dim;
      }
      rayIdx(s_dim) += step(s_dim);
      t_max(s_dim) += delta_t(s_dim);
    }
    return dist;
  }

  void compressPoly(Polyhedron3D& poly, double dx) {
    vec_E<Hyperplane3D> hyper_planes = poly.hyperplanes();
    for (uint j = 0; j < hyper_planes.size(); j++) {
      hyper_planes[j].p_ = hyper_planes[j].p_ - hyper_planes[j].n_ * dx;
    }
    poly = Polyhedron3D(hyper_planes);
  }
  void compressPoly(Eigen::MatrixXd& poly, double dx) {
    for (int i = 0; i < poly.cols(); ++i) {
      poly.col(i).tail(3) = poly.col(i).tail(3) - poly.col(i).head(3) * dx;
    }
  }

  void getPointCloudAroundLine(const vec_Vec3f& line,
                               const int maxWidth,
                               vec_Vec3f& pc) {
    pc.clear();
    Eigen::Vector3d p0 = line.front();
    Eigen::Vector3d p1 = line.back();
    Eigen::Vector3i idx0 = mapPtr_->pos2idx(p0);
    Eigen::Vector3i idx1 = mapPtr_->pos2idx(p1);
    Eigen::Vector3i d_idx = idx1 - idx0;
    Eigen::Vector3i step = d_idx.array().sign().cast<int>();
    Eigen::Vector3d delta_t;
    Eigen::Vector3i tmp_p, margin;
    margin.setConstant(maxWidth);
    for (tmp_p.x() = idx0.x() - margin.x(); tmp_p.x() <= idx0.x() + margin.x(); ++tmp_p.x()) {
      for (tmp_p.y() = idx0.y() - margin.y(); tmp_p.y() <= idx0.y() + margin.y(); ++tmp_p.y()) {
        for (tmp_p.z() = idx0.z() - margin.z(); tmp_p.z() <= idx0.z() + margin.z(); ++tmp_p.z()) {
          if (mapPtr_->isOccupied(tmp_p)) {
            pc.push_back(mapPtr_->idx2pos(tmp_p));
          }
        }
      }
    }
    for (int i = 0; i < 3; ++i) {
      delta_t(i) = d_idx(i) == 0 ? 2.0 : 1.0 / std::abs(d_idx(i));
    }
    Eigen::Vector3d t_max;
    for (int i = 0; i < 3; ++i) {
      t_max(i) = step(i) > 0 ? std::ceil(p0(i)) - p0(i) : p0(i) - std::floor(p0(i));
    }
    t_max = t_max.cwiseProduct(delta_t);
    Eigen::Vector3i rayIdx = idx0;
    // ray casting
    while (rayIdx != idx1) {
      // find the shortest t_max
      int s_dim = 0;
      for (int i = 1; i < 3; ++i) {
        s_dim = t_max(i) < t_max(s_dim) ? i : s_dim;
      }
      rayIdx(s_dim) += step(s_dim);
      t_max(s_dim) += delta_t(s_dim);
      margin.setConstant(maxWidth);
      margin(s_dim) = 0;
      Eigen::Vector3i center = rayIdx;
      center(s_dim) += maxWidth * step(s_dim);
      for (tmp_p.x() = center.x() - margin.x(); tmp_p.x() <= center.x() + margin.x(); ++tmp_p.x()) {
        for (tmp_p.y() = center.y() - margin.y(); tmp_p.y() <= center.y() + margin.y(); ++tmp_p.y()) {
          for (tmp_p.z() = center.z() - margin.z(); tmp_p.z() <= center.z() + margin.z(); ++tmp_p.z()) {
            if (mapPtr_->isOccupied(tmp_p)) {
              pc.push_back(mapPtr_->idx2pos(tmp_p));
            }
          }
        }
      }
    }
  }

  bool filterCorridor(std::vector<Eigen::MatrixXd>& hPolys) {
    // return false;
    bool ret = false;
    if (hPolys.size() <= 2) {
      return ret;
    }
    std::vector<Eigen::MatrixXd> ret_polys;
    Eigen::MatrixXd hPoly0 = hPolys[0];
    Eigen::MatrixXd curIH;
    Eigen::Vector3d interior;
    for (int i = 2; i < (int)hPolys.size(); i++) {
      curIH.resize(6, hPoly0.cols() + hPolys[i].cols());
      curIH << hPoly0, hPolys[i];
      if (geoutils::findInteriorDist(curIH, interior) < 1.0) {
        ret_polys.push_back(hPoly0);
        hPoly0 = hPolys[i - 1];
      } else {
        ret = true;
      }
    }
    ret_polys.push_back(hPoly0);
    ret_polys.push_back(hPolys.back());
    hPolys = ret_polys;
    return ret;
  }

  void generateOneCorridor(const std::pair<Eigen::Vector3d, Eigen::Vector3d>& l,
                           const double bbox_width,
                           Eigen::MatrixXd& hPoly) {
    vec_Vec3f obs_pc;
    EllipsoidDecomp3D decomp_util;
    decomp_util.set_local_bbox(Eigen::Vector3d(bbox_width, bbox_width, bbox_width));
    int maxWidth = bbox_width / mapPtr_->resolution;

    vec_Vec3f line;
    line.push_back(l.first);
    line.push_back(l.second);
    getPointCloudAroundLine(line, maxWidth, obs_pc);
    decomp_util.set_obs(obs_pc);
    decomp_util.dilate(line);
    Polyhedron3D poly = decomp_util.get_polyhedrons()[0];
    compressPoly(poly, 0.1);

    vec_E<Hyperplane3D> current_hyperplanes = poly.hyperplanes();
    hPoly.resize(6, current_hyperplanes.size());
    for (uint j = 0; j < current_hyperplanes.size(); j++) {
      hPoly.col(j) << current_hyperplanes[j].n_, current_hyperplanes[j].p_;
    }
    return;
  }

  void generateSFC(const std::vector<Eigen::Vector3d>& path,
                   const double bbox_width,
                   std::vector<Eigen::MatrixXd>& hPolys,
                   std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& keyPts) {
    assert(path.size() > 1);
    vec_Vec3f obs_pc;
    EllipsoidDecomp3D decomp_util;
    decomp_util.set_local_bbox(Eigen::Vector3d(bbox_width, bbox_width, bbox_width));


    int maxWidth = bbox_width / mapPtr_->resolution;

    vec_E<Polyhedron3D> decompPolys;

    int path_len = path.size();

    int idx = 0;
    keyPts.clear();

    while (idx < path_len - 1) {
      int next_idx = idx;
      // looking forward -> get a farest next_idx
      while (next_idx + 1 < path_len && checkRayValid(path[idx], path[next_idx + 1], bbox_width)) {
        next_idx++;
      }
      // avoid same position
      if(next_idx == (path_len - 1))
      {
        while((path[next_idx] - path[idx]).norm() < 1e-4 && next_idx > 0)
        {
          next_idx --;
        }
      }
      // generate corridor with idx and next_idx
      vec_Vec3f line;
      line.push_back(path[idx]);
      line.push_back(path[next_idx]);
      keyPts.emplace_back(path[idx], path[next_idx]);
      getPointCloudAroundLine(line, maxWidth, obs_pc);
      decomp_util.set_obs(obs_pc);
      decomp_util.dilate(line);
      Polyhedron3D poly = decomp_util.get_polyhedrons()[0];
      decompPolys.push_back(poly);

      // find a farest idx in current corridor
      idx = next_idx;
      while (idx + 1 < path_len && decompPolys.back().inside(path[idx + 1])) {
        idx++;
      }
    }

    hPolys.clear();
    Eigen::MatrixXd current_poly;
    for (uint i = 0; i < decompPolys.size(); i++) {
      vec_E<Hyperplane3D> current_hyperplanes = decompPolys[i].hyperplanes();
      current_poly.resize(6, current_hyperplanes.size());
      for (uint j = 0; j < current_hyperplanes.size(); j++) {
        current_poly.col(j) << current_hyperplanes[j].n_, current_hyperplanes[j].p_;
        //outside
      }
      hPolys.push_back(current_poly);
    }
    filterCorridor(hPolys);
    // check again
    Eigen::MatrixXd curIH;
    Eigen::Vector3d interior;
    std::vector<int> inflate(hPolys.size(), 0);
    for (int i = 0; i < (int)hPolys.size(); i++) {     
      if (geoutils::findInteriorDist(current_poly, interior) < 0.1) {
        inflate[i] = 1;
      } else {
        compressPoly(hPolys[i], 0.1);
      }
    }
    for (int i = 1; i < (int)hPolys.size(); i++) {
      curIH.resize(6, hPolys[i - 1].cols() + hPolys[i].cols());
      curIH << hPolys[i - 1], hPolys[i];
      if (!geoutils::findInterior(curIH, interior)) {
        if (!inflate[i - 1]) {
          compressPoly(hPolys[i - 1], -0.1);
          inflate[i - 1] = 1;
        }
      } else {
        continue;
      }
      curIH << hPolys[i - 1], hPolys[i];
      if (!geoutils::findInterior(curIH, interior)) {
        if (!inflate[i]) {
          compressPoly(hPolys[i], -0.1);
          inflate[i] = 1;
        }
      }
    }
  }

  inline void visCorridor(const vec_E<Polyhedron3D>& polyhedra) {
    decomp_ros_msgs::PolyhedronArray poly_msg = DecompROS::polyhedron_array_to_ros(polyhedra);
    poly_msg.header.frame_id = "world";
    poly_msg.header.stamp = ros::Time::now();
    hPolyPub_.publish(poly_msg);
  }
  inline void visCorridor(const std::vector<Eigen::MatrixXd>& hPolys) {
    vec_E<Polyhedron3D> decompPolys;
    for (const auto& poly : hPolys) {
      vec_E<Hyperplane3D> hyper_planes;
      hyper_planes.resize(poly.cols());
      for (int i = 0; i < poly.cols(); ++i) {
        hyper_planes[i].n_ = poly.col(i).head(3);
        hyper_planes[i].p_ = poly.col(i).tail(3);
      }
      decompPolys.emplace_back(hyper_planes);
    }
    visCorridor(decompPolys);
  }

  bool rayValid(const Eigen::Vector3i& idx0, const Eigen::Vector3i& idx1) {
    Eigen::Vector3i d_idx = idx1 - idx0;
    Eigen::Vector3i step = d_idx.array().sign().cast<int>();
    Eigen::Vector3d delta_t;
    for (int i = 0; i < 3; ++i) {
      delta_t(i) = d_idx(i) == 0 ? std::numeric_limits<double>::max() : 1.0 / std::fabs(d_idx(i));
    }
    Eigen::Vector3d t_max(0.5, 0.5, 0.5);
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
  };

  inline bool findAstarPath(const Eigen::Vector3i& start_idx,
                              const Eigen::Vector3i& end_idx,
                              std::vector<Eigen::Vector3i>& idx_path) {
    auto stopCondition = [&](const NodePtr& ptr) -> bool {
      return ptr->idx == end_idx;
    };
    auto calulateHeuristic = [&](const NodePtr& ptr) {
      Eigen::Vector3i dp = end_idx - ptr->idx;
      double dx = dp.x();
      double dy = dp.y();
      double dz = dp.z();
      ptr->h = abs(dp.x()) + abs(dp.y()) + abs(dp.z());
      double dx0 = (start_idx - end_idx).x();
      double dy0 = (start_idx - end_idx).y();
      double cross = fabs(dx * dy0 - dy * dx0) + abs(dz);
      ptr->h += 0.001 * cross;
    };
    // initialization of datastructures
    std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator> open_set;
    std::vector<std::pair<Eigen::Vector3i, double>> neighbors;
    // NOTE 6-connected graph
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3i neighbor(0, 0, 0);
      neighbor[i] = 1;
      neighbors.emplace_back(neighbor, 1);
      neighbor[i] = -1;
      neighbors.emplace_back(neighbor, 1);
    }
    bool ret = false;
    NodePtr curPtr = visit(start_idx);
    // NOTE we should permit the start pos invalid! (for corridor generation)
    if (!curPtr->valid) {
      visited_nodes_.clear();
      std::cout << "start postition invalid!" << std::endl;
      return false;
    }
    curPtr->parent = nullptr;
    curPtr->g = 0;
    calulateHeuristic(curPtr);
    curPtr->state = CLOSE;

    double t_cost = (ros::Time::now() - t_start_).toSec();
    if (t_cost > MAX_DURATION) {
      std::cout << "[env] search costs more than " << MAX_DURATION << "s!" << std::endl;
    }
    while (visited_nodes_.size() < MAX_MEMORY && t_cost <= MAX_DURATION) {
      for (const auto& neighbor : neighbors) {
        auto neighbor_idx = curPtr->idx + neighbor.first;
        auto neighbor_dist = neighbor.second;
        NodePtr neighborPtr = visit(neighbor_idx);
        if (neighborPtr->state == CLOSE) {
          continue;
        }
        if (neighborPtr->state == OPEN) {
          // check neighbor's g score
          // determine whether to change its parent to current
          if (neighborPtr->g > curPtr->g + neighbor_dist) {
            neighborPtr->parent = curPtr;
            neighborPtr->g = curPtr->g + neighbor_dist;
          }
          continue;
        }
        if (neighborPtr->state == UNVISITED) {
          if (neighborPtr->valid) {
            neighborPtr->parent = curPtr;
            neighborPtr->state = OPEN;
            neighborPtr->g = curPtr->g + neighbor_dist;
            calulateHeuristic(neighborPtr);
            open_set.push(neighborPtr);
          }
        }
      }  // for each neighbor
      if (open_set.empty()) {
        // std::cout << "start postition invalid!" << std::endl;
        std::cout << "[env] no way!" << std::endl;
        break;
      }
      curPtr = open_set.top();
      open_set.pop();
      curPtr->state = CLOSE;
      if (stopCondition(curPtr)) {
        ret = true;
        break;
      }
      if (visited_nodes_.size() == MAX_MEMORY) {
        std::cout << "[env] out of memory!" << std::endl;
      }
    }
    if (ret) {
      for (NodePtr ptr = curPtr; ptr != nullptr; ptr = ptr->parent) {
        idx_path.push_back(ptr->idx);
      }
      // idx_path.push_back(start_idx);
      std::reverse(idx_path.begin(), idx_path.end());
    }
    visited_nodes_.clear();

    return ret;
  }

  inline bool findPath(const Eigen::Vector3d& start_p,
                              std::vector<Eigen::Vector3d>& way_pts,
                              std::vector<Eigen::Vector3d>& path) {
    t_start_ = ros::Time::now();
    Eigen::Vector3i start_idx = mapPtr_->pos2idx(start_p);
    std::vector<Eigen::Vector3i> idx_path;
    path.push_back(start_p);
    for (const auto& way_p : way_pts) {
      Eigen::Vector3i end_idx = mapPtr_->pos2idx(way_p);
      idx_path.clear();
      if (!findAstarPath(start_idx, end_idx, idx_path)) {
        return false;
      }
      start_idx = end_idx;
      for (const auto& idx : idx_path) {
        path.push_back(mapPtr_->idx2pos(idx));
      }
    }
    return true;
  }

  inline void visible_pair(const Eigen::Vector3d& center,
                           Eigen::Vector3d& seed,
                           Eigen::Vector3d& visible_p,
                           double& theta) {
    Eigen::Vector3d dp = seed - center;
    double theta0 = atan2(dp.y(), dp.x());
    double d_theta = mapPtr_->resolution / desired_dist_ / 2;
    double t_l, t_r;
    for (t_l = theta0 - d_theta; t_l > theta0 - M_PI; t_l -= d_theta) {
      Eigen::Vector3d p = center;
      p.x() += desired_dist_ * cos(t_l);
      p.y() += desired_dist_ * sin(t_l);
      if (!checkRayValid(p, center)) {
        t_l += d_theta;
        break;
      }
    }
    for (t_r = theta0 + d_theta; t_r < theta0 + M_PI; t_r += d_theta) {
      Eigen::Vector3d p = center;
      p.x() += desired_dist_ * cos(t_r);
      p.y() += desired_dist_ * sin(t_r);
      if (!checkRayValid(p, center)) {
        t_r -= d_theta;
        break;
      }
    }
    double theta_v = (t_l + t_r) / 2;
    visible_p = center;
    visible_p.x() += desired_dist_ * cos(theta_v);
    visible_p.y() += desired_dist_ * sin(theta_v);
    theta = (t_r - t_l) / 2;
    double theta_c = theta < alpha_clearance_ ? theta : alpha_clearance_;
    if (theta0 - t_l < theta_c) {
      seed = center;
      seed.x() += desired_dist_ * cos(t_l + theta_c);
      seed.y() += desired_dist_ * sin(t_l + theta_c);
    } else if (t_r - theta0 < theta_c) {
      seed = center;
      seed.x() += desired_dist_ * cos(t_r - theta_c);
      seed.y() += desired_dist_ * sin(t_r - theta_c);
    }
    return;
  }

  inline void generate_visible_regions(const std::vector<Eigen::Vector3d>& targets,
                                       std::vector<Eigen::Vector3d>& seeds,
                                       std::vector<Eigen::Vector3d>& visible_ps,
                                       std::vector<double>& thetas) {
    assert(targets.size() == seeds.size());
    visible_ps.clear();
    thetas.clear();
    Eigen::Vector3d visible_p;
    double theta = 0;
    int M = targets.size();
    for (int i = 0; i < M; ++i) {
      visible_pair(targets[i], seeds[i], visible_p, theta);
      visible_ps.push_back(visible_p);
      thetas.push_back(theta);
    }
    return;
  }

  // NOTE
  // predict -> generate a series of circles
  // from drone to circle one by one search visible
  // put them together to generate corridor
  // for each center of circle, generate a visible region
  // optimization penalty:  <a,b>-cos(theta0)*|a|*|b| <= 0

  inline bool short_astar(const Eigen::Vector3d& start_p,
                          const Eigen::Vector3d& end_p,
                          std::vector<Eigen::Vector3d>& path) {
    Eigen::Vector3i start_idx = mapPtr_->pos2idx(start_p);
    Eigen::Vector3i end_idx = mapPtr_->pos2idx(end_p);
    auto stopCondition = [&](const NodePtr& ptr) -> bool {
      return ptr->idx == end_idx;
    };
    auto calulateHeuristic = [&](const NodePtr& ptr) {
      Eigen::Vector3i dp = end_idx - ptr->idx;
      int dx = dp.x();
      int dy = dp.y();
      int dz = dp.z();
      ptr->h = abs(dx) + abs(dy) + abs(dz);
      double dx0 = (start_idx - end_idx).x();
      double dy0 = (start_idx - end_idx).y();
      double cross = fabs(dx * dy0 - dy * dx0) + abs(dz);
      ptr->h += 0.001 * cross;
    };
    // initialization of datastructures
    std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator> open_set;
    std::vector<std::pair<Eigen::Vector3i, double>> neighbors;
    // NOTE 6-connected graph
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3i neighbor(0, 0, 0);
      neighbor[i] = 1;
      neighbors.emplace_back(neighbor, 1);
      neighbor[i] = -1;
      neighbors.emplace_back(neighbor, 1);
    }
    bool ret = false;
    NodePtr curPtr = visit(start_idx);
    // NOTE we should permit the start pos invalid! (for corridor generation)
    if (!curPtr->valid) {
      visited_nodes_.clear();
      std::cout << "[short astar]start postition invalid! start_p: " << start_p.transpose() << "start_idx: " << start_idx.transpose() << std::endl;
      return false;
    }
    curPtr->parent = nullptr;
    curPtr->g = 0;
    calulateHeuristic(curPtr);
    curPtr->state = CLOSE;

    while (visited_nodes_.size() < MAX_MEMORY) {
      for (const auto& neighbor : neighbors) {
        auto neighbor_idx = curPtr->idx + neighbor.first;
        auto neighbor_dist = neighbor.second;
        NodePtr neighborPtr = visit(neighbor_idx);
        if (neighborPtr->state == CLOSE) {
          continue;
        }
        if (neighborPtr->state == OPEN) {
          // check neighbor's g score
          // determine whether to change its parent to current
          if (neighborPtr->g > curPtr->g + neighbor_dist) {
            neighborPtr->parent = curPtr;
            neighborPtr->g = curPtr->g + neighbor_dist;
          }
          continue;
        }
        if (neighborPtr->state == UNVISITED) {
          if (neighborPtr->valid) {
            neighborPtr->parent = curPtr;
            neighborPtr->state = OPEN;
            neighborPtr->g = curPtr->g + neighbor_dist;
            calulateHeuristic(neighborPtr);
            open_set.push(neighborPtr);
          }
        }
      }  // for each neighbor
      if (open_set.empty()) {
        // std::cout << "start postition invalid!" << std::endl;
        std::cout << "[short astar] no way!" << std::endl;
        break;
      }
      curPtr = open_set.top();
      open_set.pop();
      curPtr->state = CLOSE;
      if (stopCondition(curPtr)) {
        ret = true;
        break;
      }
      if (visited_nodes_.size() == MAX_MEMORY) {
        std::cout << "[short astar] out of memory!" << std::endl;
      }
    }
    if (ret) {
      for (NodePtr ptr = curPtr->parent; ptr->parent != nullptr; ptr = ptr->parent) {
        path.push_back(mapPtr_->idx2pos(ptr->idx));
      }
      std::reverse(path.begin(), path.end());
    }
    visited_nodes_.clear();
    return ret;
  }

  inline void pts2path(const std::vector<Eigen::Vector3d>& wayPts, std::vector<Eigen::Vector3d>& path) {
    path.clear();
    path.push_back(wayPts.front());
    int M = wayPts.size();
    std::vector<Eigen::Vector3d> short_path;
    for (int i = 0; i < M - 1; ++i) {
      const Eigen::Vector3d& p0 = path.back();
      const Eigen::Vector3d& p1 = wayPts[i + 1];
      if (mapPtr_->pos2idx(p0) == mapPtr_->pos2idx(p1)) {
        continue;
      }
      if (!checkRayValid(p0, p1, 1.5)) {
        short_path.clear();
        short_astar(p0, p1, short_path);
        for (const auto& p : short_path) {
          path.push_back(p);
        }
      }
      path.push_back(p1);
    }
    if (path.size() < 2) {
      Eigen::Vector3d p = path.front();
      p.z() += 0.1;
      path.push_back(p);
    }
  }

    inline bool generate_view_regions(const std::vector<Eigen::Vector3d>& predict,
                                                                        const std::vector<double>& predict_yaws,
                                                                        const std::vector<shot::ShotConfig>& shotcfg_list,
                                                                        const std::vector<Eigen::Vector3d>& init_points,
                                                                        std::vector<Eigen::Vector3d>& visible_ps,
                                                                        std::vector<Eigen::MatrixXd>& visible_regions)
    {
        std::vector<Eigen::MatrixXd> Polys_for_vis;  //for visualization only
        visible_ps.clear();
        visible_regions.clear();
        Polys_for_vis.clear();
        for(uint i = 1; i < predict.size() * 4 / 5; i++)
        {
            Eigen::Vector3d vis_p;
            Eigen::MatrixXd Poly;
            if(!get_view_point_region(predict[i], init_points[i], predict_yaws[i], shotcfg_list[i], vis_p, Poly))
            {
                ROS_WARN("[plan env]: no available view region! fail node: %d", i);
                return false;
            }
            // std::cout << "***************************************" << std::endl << Poly << std::endl;
            Eigen::MatrixXd Poly_vis;
            Poly_vis.resize(6, 5 + 2);
            Poly_vis.block<6, 5>(0, 0) = Poly;
            Eigen::Vector3d p1, p2, p5, v;
            p1 = Poly.block<3,1>(3,0);
            p2 = Poly.block<3,1>(3,1);
            p5 = Poly.block<3,1>(3,4);
            v = (p1- p5).cross(p2 - p1);
            v = v.normalized();
            Poly_vis.block<3,1>(0, 5) = v;
            Poly_vis.block<3,1>(3, 5) = p1 + 0.01 * v;
            Poly_vis.block<3,1>(0, 6) = -v;
            Poly_vis.block<3,1>(3, 6) = p1 - 0.01 * v;
            Polys_for_vis.emplace_back(Poly_vis);
            visible_ps.emplace_back(vis_p);
            visible_regions.emplace_back(Poly);
            // add z
            // Eigen::MatrixXd Poly_z;
            // Poly_z.resize(6, 5 + 2);
            // Poly_z.block<6, 5>(0, 0) = Poly;
            // Poly_z.block<3,1>(0, 5) = v;
            // Poly_z.block<3,1>(3, 5) = p1 + 0.5 * v;
            // Poly_z.block<3,1>(0, 6) = -v;
            // Poly_z.block<3,1>(3, 6) = p1 - 0.5 * v;
            // visible_regions.emplace_back(Poly_z);
        }
        // since initial state is not constrained, simply copy i = 1 for i = 0
        visible_ps.insert(visible_ps.begin(), init_points[0]);
        visible_regions.insert(visible_regions.begin(), visible_regions[0]);
        Polys_for_vis.insert(Polys_for_vis.begin(), Polys_for_vis[0]);
        visViewRegion(Polys_for_vis);
        // for stamps behind, keep astar points
        for(uint i = predict.size() * 4 / 5; i <predict.size(); i++)
        {
            visible_ps.emplace_back(init_points[i]);
        } 
        return true;
    }

    inline double getOcclusion(const Eigen::Vector3d& center,
                                                                        const double& theta,
                                                                        const double& tilt,
                                                                        const double& d)
    {
        Eigen::Vector3d goal;
        double horizon_d = d * cos(tilt);
        goal<< horizon_d * cos(theta), horizon_d * sin(theta), d * sin(tilt);
        goal += center;
        return getValidRay(center, goal);
    }

    inline double calc_f(const std::deque<std::pair<double, double>>& q,
                                                    const std::deque<int>& vis,
                                                    const double& des_theta,
                                                    const double& des_d,
                                                    double &center_theta,
                                                    double &min_d,
                                                    double &dis_thresh_low,
                                                    double &dis_thresh_up)
    {
        center_theta = 0.0;
        min_d = q[0].second;
        int max_vis = 0;
        for(uint32_t i = 0; i < q.size(); i++)
        {
            center_theta += q[i].first;
            if(q[i].second < min_d)
                min_d = q[i].second;
            if(vis[i] > max_vis)
                max_vis = vis[i];
        }
        center_theta = center_theta / q.size();
        // std::cout << "min_d: " <<  min_d << std::endl;
        // std::cout << "center_theta" << center_theta << std::endl;
        if(min_d < dis_thresh_low)
            return MAX;
        
        double delta_theta = fabs(center_theta - des_theta);
        while(delta_theta > 2* M_PI)
            delta_theta -= 2*M_PI;
        delta_theta = delta_theta > M_PI ? 2*M_PI - delta_theta : delta_theta;
        double f = lambda_dist * (des_d - (min_d - clearance_d_)) + lambda_theta * delta_theta + max_vis;  
        return f;
    }

    inline bool get_view_point_region(const Eigen::Vector3d& target,
                                                                        const Eigen::Vector3d& init_p,
                                                                        const double& yaw,
                                                                        const shot::ShotConfig& param,
                                                                        Eigen::Vector3d& view_p,
                                                                        Eigen::MatrixXd& view_region)
    {
        // parameters
        double visibility_thresh = 1.0; // visibility detect range, in radius
        double distance_thresh = 0.3; // distance adjustment
        int region_grid = 3;  // how large the view region is 
        double angle_thresh = 1.5 / param.distance;  // angle adjustment

        std::vector<double> ray_buffer;
        std::vector<int> occ_index;
        Eigen::VectorXi vis_cost;
        double theta, des_theta, theta_l, theta_u,  d_theta,  best_theta, theta0;
        double dis, des_d, min_d, max_d, best_dis, min_f;
        double best_tilt, tilt_z, des_tilt, tilt;

        std::vector<Eigen::Vector3d> err_region;
        err_region.emplace_back(target);

        // desired params
        best_tilt = atan((param.image_p(1) - cy_) / fy_);
        tilt_z = target.z() + param.distance * sin(best_tilt);
        if(tilt_z < clearance_d_)
            tilt_z = clearance_d_;
        des_tilt = asin((tilt_z - target.z()) / param.distance);
        des_d = param.distance * cos(des_tilt);
        des_theta = yaw + param.view_angle;  
        des_theta = des_theta > M_PI ? des_theta - 2 * M_PI : des_theta;
        des_theta = des_theta < -M_PI ? des_theta + 2 * M_PI : des_theta;

        // params by astar
        Eigen::Vector3d dp = init_p - target;
        dis = dp.norm();
        tilt = asin(dp(2) / dis);
        theta = atan2(dp(1), dp(0));
        min_d = std::max(dis - distance_thresh + clearance_d_, kill_d_);
        max_d = dis + distance_thresh + clearance_d_;
        theta_l = theta - angle_thresh;
        theta_u = theta + angle_thresh;
        theta0 = theta_l - visibility_thresh;
        d_theta = mapPtr_->resolution / des_d / 2;  

        // raycast 
        ray_buffer.clear();
        occ_index.clear();
        int ray_n = ceil((2 * angle_thresh + 2 * visibility_thresh) / d_theta);
        vis_cost.resize(ray_n);
        vis_cost.setZero();
        for(int t = 0; t < ray_n; t ++ )
        {
            double ray_length = getOcclusion(target, theta0 + t * d_theta, tilt, max_d);
            ray_buffer.emplace_back(ray_length);
            if(ray_length < min_d)
                occ_index.emplace_back(t);
        }

        // visbility cost: each grid
        int clear_n = visibility_thresh / d_theta;
        for(uint i = 0; i < occ_index.size(); i++)
        {
            int angle = occ_index[i];
            int begin = std::max(angle - clear_n, 0);
            int end = std::min(angle + clear_n, ray_n - 1);
            for(int t = begin; t <= end; t++)
            {
                int cost = clear_n - abs(t - angle);
                vis_cost(t) = std::max(vis_cost(t), cost);
            }
        }

        // regions
        int region_i = (int)(visibility_thresh  / d_theta);  // index theta_l
        std::deque<std::pair<double, double>> theta_dis_queue;
        std::deque<int> vis_queue;
        for(int i = 0; i < region_grid; i++)
        {
            std::pair<double, double> p;
            p.first = theta0 + region_i * d_theta;
            p.second = ray_buffer[region_i];
            theta_dis_queue.emplace_back(p);
            vis_queue.emplace_back(vis_cost(region_i));
            err_region.emplace_back(target + p.second * Eigen::Vector3d(cos(tilt) * cos(p.first), cos(tilt) *sin(p.first), sin(tilt)));
            region_i++;
        }
        min_f = calc_f(theta_dis_queue, vis_queue, des_theta, des_d, best_theta, best_dis, min_d, max_d);  
        while(region_i <= (visibility_thresh + 2 * angle_thresh) / d_theta)
        {
            double theta_p, dis_p;
            std::pair<double, double> p;
            p.first = theta0 + region_i * d_theta;
            p.second = ray_buffer[region_i];
            theta_dis_queue.pop_front();
            theta_dis_queue.push_back(p);
            vis_queue.pop_front();
            vis_queue.push_back(vis_cost(region_i));
            err_region.emplace_back(target + p.second * Eigen::Vector3d(cos(tilt) * cos(p.first), cos(tilt) *sin(p.first), sin(tilt)));
            double ft = calc_f(theta_dis_queue, vis_queue, des_theta, des_d, theta_p, dis_p, min_d, max_d);  
            if(ft < min_f)
            {
                min_f = ft;
                best_theta = theta_p;
                best_dis = dis_p;
            }
            region_i ++;
        }

        if(min_f > MAX - 0.1)
        {
            // std::cout << "[Env]: no available view region!" << std::endl;
            std::vector<Eigen::Vector3d> vis_points;
            uint32_t len = err_region.size();
            for(uint32_t i = 0; i < len; i++)
            {
                Eigen::Vector3d start, end;
                start = err_region[i];
                end = (i == (len-1)) ? err_region[0] : err_region[i+1];
                int n = 50; // interpolation number
                for(int j = 0; j < n; j ++)
                    vis_points.emplace_back(start + (end - start) * j / n);
            }
            visualize_error_region(vis_points);
            // failure, return init p
            // return false;
            view_p = init_p;
            best_theta = theta;
            best_dis = dis;
        }
        else
        {
          best_theta = best_theta < -M_PI ? best_theta + 2 * M_PI : best_theta;
          best_theta = best_theta > M_PI ? best_theta - 2 * M_PI : best_theta;
          best_dis -= clearance_d_;
          // best view_p
          view_p = target + best_dis * Eigen::Vector3d(cos(tilt) * cos(best_theta), cos(tilt) * sin(best_theta), sin(tilt));
        }
        // view region polyhedra
        double dis_low, dis_upper, alpha_low, alpha_upper;
        dis_low = best_dis - tolerance_d_; dis_upper = best_dis + tolerance_d_;
        alpha_low = best_theta - d_theta * region_grid /2; alpha_upper = best_theta + d_theta * region_grid /2;
        Eigen::Vector3d p1, p2, p3, p4, p5, v1, v2, v3, v4, v5, v;
        p1 = target + dis_low * Eigen::Vector3d(cos(tilt) * cos(alpha_low), cos(tilt) * sin(alpha_low), sin(tilt));
        p2 = target + dis_upper * Eigen::Vector3d(cos(tilt) * cos(alpha_low),  cos(tilt) * sin(alpha_low), sin(tilt));
        p3 = target + dis_upper * Eigen::Vector3d(cos(tilt) * cos(best_theta), cos(tilt) * sin(best_theta), sin(tilt));
        p4 = target + dis_upper * Eigen::Vector3d(cos(tilt) * cos(alpha_upper), cos(tilt) * sin(alpha_upper), sin(tilt));
        p5 = target + dis_low * Eigen::Vector3d(cos(tilt) * cos(alpha_upper), cos(tilt) * sin(alpha_upper), sin(tilt));
        v = (p1- p5).cross(p2 - p1);
        // v = v.normalized();
        normalized_v_vec(p1, p2, v, v1); 
        normalized_v_vec(p2, p3, v, v2);
        normalized_v_vec(p3, p4, v, v3);
        normalized_v_vec(p4, p5, v, v4);
        normalized_v_vec(p5, p1, v, v5);  
        view_region.resize(6, 5);
        view_region.block<3, 1>(0, 0) = v1; view_region.block<3, 1>(3, 0) = p1;
        view_region.block<3, 1>(0, 1) = v2; view_region.block<3, 1>(3, 1) = p2;
        view_region.block<3, 1>(0, 2) = v3; view_region.block<3, 1>(3, 2) = p3;
        view_region.block<3, 1>(0, 3) = v4; view_region.block<3, 1>(3, 3) = p4;
        view_region.block<3, 1>(0, 4) = v5; view_region.block<3, 1>(3, 4) = p5;
        return true;
    }

    inline void visualize_error_region(const std::vector<Eigen::Vector3d> path)
    {
        pcl::PointCloud<pcl::PointXYZ> point_cloud;
        sensor_msgs::PointCloud2 point_cloud_msg;
        point_cloud.reserve(path.size());
        for (const auto& pt : path) {
            point_cloud.points.emplace_back(pt[0], pt[1], pt[2]);
        }
        pcl::toROSMsg(point_cloud, point_cloud_msg);
        point_cloud_msg.header.frame_id = "world";
        point_cloud_msg.header.stamp = ros::Time::now();
        dbgRegionPub_.publish(point_cloud_msg);
    }

    inline void normalized_v_vec(const Eigen::Vector3d& p1,
                                                        const Eigen::Vector3d& p2,
                                                        const Eigen::Vector3d& v,
                                                        Eigen::Vector3d& v_ret)
    {
        v_ret = (p2 - p1).cross(v);
        v_ret = v_ret.normalized();
    }

    inline void visViewRegion(const std::vector<Eigen::MatrixXd>& hPolys)
    {
        vec_E<Polyhedron3D> decompPolys;
        bool show_one = false;
        if(show_one)
        {
            Eigen::MatrixXd poly = hPolys[0];
            vec_E<Hyperplane3D> hyper_planes;
            hyper_planes.resize(poly.cols());
            for (int i = 0; i < poly.cols(); ++i) {
            hyper_planes[i].n_ = poly.col(i).head(3);
            hyper_planes[i].p_ = poly.col(i).tail(3);
            }
            decompPolys.emplace_back(hyper_planes);
        }
        else
        {
            for (const auto& poly : hPolys) {
                vec_E<Hyperplane3D> hyper_planes;
                hyper_planes.resize(poly.cols());
                for (int i = 0; i < poly.cols(); ++i) {
                hyper_planes[i].n_ = poly.col(i).head(3);
                hyper_planes[i].p_ = poly.col(i).tail(3);
                }
                decompPolys.emplace_back(hyper_planes);
            }           
        }
        decomp_ros_msgs::PolyhedronArray poly_msg = DecompROS::polyhedron_array_to_ros(decompPolys);
        poly_msg.header.frame_id = "world";
        poly_msg.header.stamp = ros::Time::now();
        viewRegionPub_.publish(poly_msg);
    }

};

}  // namespace env