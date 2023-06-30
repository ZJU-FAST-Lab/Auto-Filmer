#pragma once
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Core>
#include <vector>

namespace mapping {

template <typename _Datatype>
struct RingBuffer {
 public:
  double resolution;
  int size_x, size_y, size_z;
  int offset_x, offset_y, offset_z;
  std::vector<_Datatype> data;

  inline const int idx2add(int x, int N) const {
    // return x % N >= 0 ? x % N : x % N + N;
    // NOTE this is much faster than before!!
    return (x & N) >= 0 ? (x & N) : (x & N) + N;
  }
  inline const Eigen::Vector3i idx2add(const Eigen::Vector3i& id) const {
    return Eigen::Vector3i(idx2add(id.x(), size_x - 1),
                           idx2add(id.y(), size_y - 1),
                           idx2add(id.z(), size_z - 1));
  }
  // NOTE dangerous!! ad should be the address in the data
  inline const _Datatype& at(const Eigen::Vector3i& ad) const {
    return data[(ad.z() * size_y + ad.y()) * size_x + ad.x()];
  }
  inline _Datatype& at(const Eigen::Vector3i& ad) {
    return data[(ad.z() * size_y + ad.y()) * size_x + ad.x()];
  }
  inline _Datatype* atPtr(const Eigen::Vector3i& ad) {
    return &(data[(ad.z() * size_y + ad.y()) * size_x + ad.x()]);
  }
  inline const _Datatype& atId(const Eigen::Vector3i& id) const {
    return at(idx2add(id));
  }
  inline _Datatype& atId(const Eigen::Vector3i& id) {
    return at(idx2add(id));
  }
  inline _Datatype* atIdPtr(const Eigen::Vector3i& id) {
    return atPtr(idx2add(id));
  }
  inline const Eigen::Vector3i pos2idx(const Eigen::Vector3d& pt) const {
    return (pt / resolution).array().floor().cast<int>();
  }
  inline const Eigen::Vector3d idx2pos(const Eigen::Vector3i& id) const {
    return (id.cast<double>() + Eigen::Vector3d(0.5, 0.5, 0.5)) * resolution;
  }
  inline const bool isInMap(const Eigen::Vector3i& id) const {
    return !((id.x() - offset_x) & (~(size_x - 1))) &&
           !((id.y() - offset_y) & (~(size_y - 1))) &&
           !((id.z() - offset_z) & (~(size_z - 1)));
  }
  inline const bool isInMap(const Eigen::Vector3d& p) const {
    return isInMap(pos2idx(p));
  }
  template <typename _Msgtype>
  inline void to_msg(_Msgtype& msg) {
    msg.resolution = resolution;
    msg.size_x = size_x;
    msg.size_y = size_y;
    msg.size_z = size_z;
    msg.offset_x = offset_x;
    msg.offset_y = offset_y;
    msg.offset_z = offset_z;
    msg.data = data;
  }
  template <typename _Msgtype>
  inline void from_msg(const _Msgtype& msg) {
    resolution = msg.resolution;
    size_x = msg.size_x;
    size_y = msg.size_y;
    size_z = msg.size_z;
    offset_x = msg.offset_x;
    offset_y = msg.offset_y;
    offset_z = msg.offset_z;
    data = msg.data;
  }
};

struct OccGridMap : public RingBuffer<int8_t> {
 private:
  // std::vector<int8_t> data;  // 1 for occupied, 0 for known, -1 for free
  // NOTE only used for update map
  std::vector<int8_t> vis;  // 1 for occupied, -1 for raycasted, 0 for free or unvisited
  std::vector<int> pro;
  bool init_finished = false;  // just for initialization
  int p_min, p_max, p_hit, p_mis, p_occ, p_def;
  double sensor_range;

 public:
  std::vector<Eigen::Vector3i> v0, v1;
  inline void setup(const double& res,
                    const Eigen::Vector3d& map_size,
                    const double& cam_range,
                    bool use_global_map = false) {
    resolution = res;
    size_x = exp2(int(log2(map_size.x() / res)));
    size_y = exp2(int(log2(map_size.y() / res)));
    size_z = exp2(int(log2(map_size.z() / res)));
    if (use_global_map) {
      offset_x = -size_x;
      offset_y = -size_y;
    //   offset_z = -size_z;  //benchmark map
      size_x *= 2;
      size_y *= 2;
      size_z *= 2;
    }
    data.resize(size_x * size_y * size_z);
    std::fill(data.begin(), data.end(), 0);
    if (use_global_map) {
      return;
    }
    vis.resize(size_x * size_y * size_z);
    pro.resize(size_x * size_y * size_z);
    v0.reserve(size_x * size_y * size_z);
    v1.reserve(size_x * size_y * size_z);
    std::fill(pro.begin(), pro.end(), p_def);
    sensor_range = cam_range;
  }
  inline void setOcc(const Eigen::Vector3d& p) {
    at(idx2add(pos2idx(p))) = 1;
  }
  inline void setupP(const int& _p_min,
                     const int& _p_max,
                     const int& _p_hit,
                     const int& _p_mis,
                     const int& _p_occ,
                     const int& _p_def) {
    // NOTE logit(x) = log(x/(1-x))
    p_min = _p_min;  // 0.12 -> -199
    p_max = _p_max;  // 0.90 ->  220
    p_hit = _p_hit;  // 0.65 ->   62
    p_mis = _p_mis;  // 0.35 ->   62
    p_occ = _p_occ;  // 0.80 ->  139
    p_def = _p_def;  // 0.12 -> -199
  }

 private:
  // NOTE x must be in map
  inline const int& prob(const Eigen::Vector3i& ad) const {
    return pro[(ad.z() * size_x + ad.y()) * size_y + ad.x()];
  }
  inline int& prob(const Eigen::Vector3i& ad) {
    return pro[(ad.z() * size_x + ad.y()) * size_y + ad.x()];
  }
  inline const int8_t& visited(const Eigen::Vector3i& ad) const {
    return vis[(ad.z() * size_x + ad.y()) * size_y + ad.x()];
  }
  inline int8_t& visited(const Eigen::Vector3i& ad) {
    return vis[(ad.z() * size_x + ad.y()) * size_y + ad.x()];
  }
  inline void resetVisited() {
    std::fill(vis.begin(), vis.end(), 0);
  }
  // return true if in range; id_filtered will be limited in range
  inline bool filter(const Eigen::Vector3d& sensor_p,
                     const Eigen::Vector3d& p,
                     Eigen::Vector3d& pt) const {
    Eigen::Vector3i id = pos2idx(p);
    Eigen::Vector3d dp = p - sensor_p;
    double dist = dp.norm();
    pt = p;
    if (dist < sensor_range && isInMap(id)) {
      return true;
    } else if (dist >= sensor_range) {
      pt = sensor_range / dist * dp + sensor_p;
    }
    if (isInMap(pos2idx(pt))) {
      return false;
    } else {
      dp = pt - sensor_p;
      Eigen::Array3d v = dp.array().abs() / resolution;
      Eigen::Array3d d;
      d.x() = v.x() <= size_x / 2 - 1 ? 0 : v.x() - size_x / 2 + 1;
      d.y() = v.y() <= size_y / 2 - 1 ? 0 : v.y() - size_y / 2 + 1;
      d.z() = v.z() <= size_z / 2 - 1 ? 0 : v.z() - size_z / 2 + 1;
      double t_max = 0;
      for (int i = 0; i < 3; ++i) {
        t_max = (d[i] > 0 && d[i] / v[i] > t_max) ? d[i] / v[i] : t_max;
      }
      pt = pt - dp * t_max;
      return false;
    }
  }
  inline void hit(const Eigen::Vector3i& ad) {
    prob(ad) = prob(ad) + p_hit > p_max ? p_max : prob(ad) + p_hit;
    at(ad) = prob(ad) > p_occ ? 1 : -1;
    visited(ad) = 1;  // set occupied
  }
  inline void mis(const Eigen::Vector3i& ad) {
    prob(ad) = prob(ad) - p_mis < p_min ? p_min : prob(ad) - p_mis;
    at(ad) = prob(ad) > p_occ ? 1 : -1;
    visited(ad) = -1;  // set raycasted
  }
  inline void setUnKnown(const Eigen::Vector3i& ad) {
    prob(ad) = p_def;
    at(ad) = 0;
  }

 public:
  inline const bool isOccupied(const Eigen::Vector3i& id) const {
    return isInMap(id) && at(idx2add(id)) == 1;
  }
  inline const bool isOccupied(const Eigen::Vector3d& p) const {
    return isOccupied(pos2idx(p));
  }
  inline const bool isUnKnown(const Eigen::Vector3i& id) const {
    return (!isInMap(id)) || at(idx2add(id)) == 0;
  }
  inline const bool isUnKnown(const Eigen::Vector3d& p) const {
    return isUnKnown(pos2idx(p));
  }
  inline void setFree(const Eigen::Vector3i& id) {
    if (isInMap(id)) {
      Eigen::Vector3i ad = idx2add(id);
      at(ad) = -1;
      prob(ad) = p_min;
    }
  }
  inline void setFree(const Eigen::Vector3d& p) {
    Eigen::Vector3i id = pos2idx(p);
    setFree(id);
  }
  inline void setFree(const Eigen::Vector3d& ld, const Eigen::Vector3d& ru) {
    Eigen::Vector3i id_ld = pos2idx(ld);
    Eigen::Vector3i id_ru = pos2idx(ru);
    Eigen::Vector3i id;
    for (id.x() = id_ld.x(); id.x() <= id_ru.x(); ++id.x())
      for (id.y() = id_ld.y(); id.y() <= id_ru.y(); ++id.y())
        for (id.z() = id_ld.z(); id.z() <= id_ru.z(); ++id.z()) {
          setFree(id);
        }
  }
  void updateMap(const Eigen::Vector3d& sensor_p,
                 const std::vector<Eigen::Vector3d>& pc);
  void occ2pc(sensor_msgs::PointCloud2& msg);
  void occ2pc(sensor_msgs::PointCloud2& msg, double floor, double ceil);
  void inflate_once();
  void inflate_xy();
  void inflate_last();
  void inflate(int inflate_size);
};

}  // namespace mapping