#include <mapping/mapping.h>
#include <pcl_conversions/pcl_conversions.h>

#include <iostream>

namespace mapping {
void OccGridMap::updateMap(const Eigen::Vector3d& sensor_p,
                           const std::vector<Eigen::Vector3d>& pc) {
  resetVisited();

  Eigen::Vector3i sensor_idx = pos2idx(sensor_p);
  Eigen::Vector3i offset = sensor_idx - Eigen::Vector3i(size_x / 2, size_y / 2, size_z / 2);
  // TODO clear the updated part
  if (init_finished) {
    Eigen::Vector3i move = offset - Eigen::Vector3i(offset_x, offset_y, offset_z);
    Eigen::Vector3i from, to;
    from.setZero();
    to.setZero();
    from.x() = move.x() >= 0 ? 0 : move.x() + size_x;
    from.y() = move.y() >= 0 ? 0 : move.y() + size_y;
    from.z() = move.z() >= 0 ? 0 : move.z() + size_z;
    to.x() = move.x() >= 0 ? move.x() : size_x;
    to.y() = move.y() >= 0 ? move.y() : size_y;
    to.z() = move.z() >= 0 ? move.z() : size_z;
    for (int x = from.x(); x < to.x(); ++x) {
      for (int y = 0; y < size_y; ++y) {
        for (int z = 0; z < size_z; ++z) {
          Eigen::Vector3i id = Eigen::Vector3i(offset_x + x, offset_y + y, offset_z + z);
          setUnKnown(idx2add(id));
        }
      }
    }
    offset_x = offset.x();
    for (int y = from.y(); y < to.y(); ++y) {
      for (int x = 0; x < size_x; ++x) {
        for (int z = 0; z < size_z; ++z) {
          Eigen::Vector3i id = Eigen::Vector3i(offset_x + x, offset_y + y, offset_z + z);
          setUnKnown(idx2add(id));
        }
      }
    }
    offset_y = offset.y();
    for (int z = from.z(); z < to.z(); ++z) {
      for (int x = 0; x < size_x; ++x) {
        for (int y = 0; y < size_y; ++y) {
          Eigen::Vector3i id = Eigen::Vector3i(offset_x + x, offset_y + y, offset_z + z);
          setUnKnown(idx2add(id));
        }
      }
    }
    offset_z = offset.z();
  } else {
    offset_x = offset.x();
    offset_y = offset.y();
    offset_z = offset.z();
    init_finished = true;
  }
  // set occupied
  for (const auto& p : pc) {
    Eigen::Vector3d pt;
    bool inrange = filter(sensor_p, p, pt);
    Eigen::Vector3i idx = pos2idx(pt);
    Eigen::Vector3i add = idx2add(idx);
    if (visited(add) != 1) {
      if (inrange) {
        hit(add);
      } else {
        mis(add);
      }
    } else {
      continue;
    }
    // ray casting

    Eigen::Vector3i d_idx = sensor_idx - idx;
    Eigen::Vector3i step = d_idx.array().sign().cast<int>();
    Eigen::Vector3d delta_t;
    Eigen::Vector3d dp = sensor_p - pt;
    for (int i = 0; i < 3; ++i) {
      delta_t(i) = dp(i) == 0 ? std::numeric_limits<double>::max() : 1.0 / std::fabs(dp(i));
    }
    Eigen::Vector3d t_max;
    for (int i = 0; i < 3; ++i) {
      t_max(i) = step(i) > 0 ? (idx(i) + 1) - pt(i) / resolution : pt(i) / resolution - idx(i);
    }
    t_max = t_max.cwiseProduct(delta_t);
    Eigen::Vector3i rayIdx = idx;
    while ((rayIdx - sensor_idx).squaredNorm() != 1) {
      // find the shortest t_max
      int s_dim = 0;
      for (int i = 1; i < 3; ++i) {
        s_dim = t_max(i) < t_max(s_dim) ? i : s_dim;
      }
      rayIdx(s_dim) += step(s_dim);
      t_max(s_dim) += delta_t(s_dim);
      Eigen::Vector3i rayAdd = idx2add(rayIdx);
      if (visited(rayAdd) == -1) {
        break;
      }
      if (visited(rayAdd) != 1) {
        mis(rayAdd);
      }
    }
  }
}

void OccGridMap::occ2pc(sensor_msgs::PointCloud2& msg) {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> pcd;
  for (int x = 0; x < size_x; ++x) {
    for (int y = 0; y < size_y; ++y) {
      for (int z = 0; z < size_z; ++z) {
        Eigen::Vector3i idx(offset_x + x, offset_y + y, offset_z + z);
        if (atId(idx) == 1) {
          pt.x = (offset_x + x + 0.5) * resolution;
          pt.y = (offset_y + y + 0.5) * resolution;
          pt.z = (offset_z + z + 0.5) * resolution;
          pcd.push_back(pt);
        }
      }
    }
  }
  pcd.width = pcd.points.size();
  pcd.height = 1;
  pcd.is_dense = true;
  pcl::toROSMsg(pcd, msg);
  msg.header.frame_id = "world";
}

void OccGridMap::occ2pc(sensor_msgs::PointCloud2& msg, double floor, double ceil) {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> pcd;
  for (int x = 0; x < size_x; ++x) {
    for (int y = 0; y < size_y; ++y) {
      for (int z = 0; z < size_z; ++z) {
        Eigen::Vector3i idx(offset_x + x, offset_y + y, offset_z + z);
        if (atId(idx) == 1) {
          pt.x = (offset_x + x + 0.5) * resolution;
          pt.y = (offset_y + y + 0.5) * resolution;
          pt.z = (offset_z + z + 0.5) * resolution;
          if (pt.z > floor && pt.z < ceil) {
            pcd.push_back(pt);
          }
        }
      }
    }
  }
  pcd.width = pcd.points.size();
  pcd.height = 1;
  pcd.is_dense = true;
  pcl::toROSMsg(pcd, msg);
  msg.header.frame_id = "world";
}

void OccGridMap::inflate_once() {
  static Eigen::Vector3i p;
  for (const auto& id : v0) {
    int x0 = id.x() - 1 >= offset_x ? id.x() - 1 : offset_x;
    int y0 = id.y() - 1 >= offset_y ? id.y() - 1 : offset_y;
    int z0 = id.z() - 1 >= offset_z ? id.z() - 1 : offset_z;
    int x1 = id.x() + 1 <= offset_x + size_x - 1 ? id.x() + 1 : offset_x + size_x - 1;
    int y1 = id.y() + 1 <= offset_y + size_y - 1 ? id.y() + 1 : offset_y + size_y - 1;
    int z1 = id.z() + 1 <= offset_z + size_z - 1 ? id.z() + 1 : offset_z + size_z - 1;
    for (p.x() = x0; p.x() <= x1; p.x()++)
      for (p.y() = y0; p.y() <= y1; p.y()++)
        for (p.z() = z0; p.z() <= z1; p.z()++) {
          auto ptr = atIdPtr(p);
          if ((*ptr) != 1) {
            *ptr = 1;
            v1.push_back(p);
          }
        }
  }
}

void OccGridMap::inflate_xy() {
  static Eigen::Vector3i p;
  for (const auto& id : v0) {
    int x0 = id.x() - 1 >= offset_x ? id.x() - 1 : offset_x;
    int y0 = id.y() - 1 >= offset_y ? id.y() - 1 : offset_y;
    int x1 = id.x() + 1 <= offset_x + size_x - 1 ? id.x() + 1 : offset_x + size_x - 1;
    int y1 = id.y() + 1 <= offset_y + size_y - 1 ? id.y() + 1 : offset_y + size_y - 1;
    p.z() = id.z();
    for (p.x() = x0; p.x() <= x1; p.x()++)
      for (p.y() = y0; p.y() <= y1; p.y()++) {
        auto ptr = atIdPtr(p);
        if ((*ptr) != 1) {
          *ptr = 1;
          v1.push_back(p);
        }
      }
  }
}

void OccGridMap::inflate_last() {
  static Eigen::Vector3i p;
  for (const auto& id : v1) {
    int x0 = id.x() - 1 >= offset_x ? id.x() - 1 : offset_x;
    int y0 = id.y() - 1 >= offset_y ? id.y() - 1 : offset_y;
    int z0 = id.z() - 1 >= offset_z ? id.z() - 1 : offset_z;
    int x1 = id.x() + 1 <= offset_x + size_x - 1 ? id.x() + 1 : offset_x + size_x - 1;
    int y1 = id.y() + 1 <= offset_y + size_y - 1 ? id.y() + 1 : offset_y + size_y - 1;
    int z1 = id.z() + 1 <= offset_z + size_z - 1 ? id.z() + 1 : offset_z + size_z - 1;
    for (p.x() = x0; p.x() <= x1; p.x()++)
      for (p.y() = y0; p.y() <= y1; p.y()++)
        for (p.z() = z0; p.z() <= z1; p.z()++)
          atId(p) = 1;
  }
}

void OccGridMap::inflate(int inflate_size) {
  if (inflate_size < 1) {
    return;
  }
  Eigen::Vector3i idx;
  v1.clear();
  for (idx.x() = offset_x; idx.x() < offset_x + size_x; ++idx.x())
    for (idx.y() = offset_y; idx.y() < offset_y + size_y; ++idx.y())
      for (idx.z() = offset_z; idx.z() < offset_z + size_z; ++idx.z()) {
        if (atId(idx) == 1) {
          v1.push_back(idx);
        }
      }
  for (int i = 0; i < inflate_size - 1; ++i) {
    std::swap(v0, v1);
    v1.clear();
    inflate_once();
  }
  inflate_last();
}

}  // namespace mapping