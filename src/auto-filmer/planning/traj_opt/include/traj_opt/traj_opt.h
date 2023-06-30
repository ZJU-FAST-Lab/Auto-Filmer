#pragma once
#include <ros/ros.h>

#include "minco.hpp"
#include <traj_opt/flatness.hpp>
#include <quadrotor_msgs/ShotParams.h>
#include <quadrotor_msgs/OptDebug.h>

namespace traj_opt {

class TrajOpt {
 public:
  ros::NodeHandle nh_;
  ros::Publisher dbg_pub_;
  // # pieces and # key points
  int N_, K_, dim_t_, dim_p_, dim_psi_, dim_theta_;
  // weight for time regularization term
  double rhoT_;
  // collision avoiding and dynamics paramters
  double vmax_, amax_, ratemax_frame_, ratemax_;
  double rhoSmooth_, rhoAngleSmooth_, rhoMapping_;
  double rhoP_, rhoV_, rhoA_, rhoOmg_;
  double rhoNoKilling_, rhoVisibilityPos_, rhoVisibilityVel_, rhoOcclusion_, rhoViewPos_;
  double clearance_d_, tolerance_d_, alpha_clearance_, kill_d_;
  quadrotor_msgs::OptDebug dbg_msg;
  // corridor
  std::vector<Eigen::MatrixXd> cfgVs_;
  std::vector<Eigen::MatrixXd> cfgHs_;
  // Minimum Jerk Optimizer
  minco::MINCO mincoOpt_;
  // flatness map
  flatness::FlatnessMap flatmap_;
  // weight for each vertex
  Eigen::VectorXd p_;
  // duration of each piece of the trajectory
  Eigen::VectorXd t_;
  // yaw trajectory
  Eigen::VectorXd psi_;
  // gimbal trajectory
  Eigen::VectorXd theta_;
  double* x_;
  double sum_T_;

  std::vector<Eigen::Vector3d> tracking_ps_;
  std::vector<Eigen::Vector3d> tracking_vs_;
  std::vector<Eigen::MatrixXd> view_polys_;
  double tracking_dur_;
  double tracking_dist_;
  double tracking_dt_;

  //camera params
  Eigen::Matrix3d cam2body_R_;
  Eigen::Vector3d cam2body_p_;
  double fx_, fy_, cx_, cy_;
  int cam_width_, cam_height_;

  //shot params
  Eigen::Vector2d pos_image, vel_image;
  std::vector<Eigen::Vector2d> plan_pos_image, plan_vel_image;

  Eigen::Matrix3Xd gradByPoints;
  Eigen::Matrix2Xd gradByAngles;
  Eigen::VectorXd gradByTimes;
  Eigen::MatrixX3d partialGradByCoeffs;
  Eigen::MatrixX2d partialGradByCoeffsAngle;
  Eigen::VectorXd partialGradByTimes;

  // polyH utils
  bool extractVs(const std::vector<Eigen::MatrixXd>& hPs,
                 std::vector<Eigen::MatrixXd>& vPs) const;

 public:
  TrajOpt(ros::NodeHandle& nh);
  ~TrajOpt() {}

  void setBoundConds(const Eigen::MatrixXd& iniState, const Eigen::MatrixXd& finState,
                     const Eigen::MatrixXd& iniAngle);
  void setAngleInitValue(const Eigen::MatrixXd& iniAngle,
                                                const Eigen::VectorXd& T,
                                                const Eigen::MatrixXd& P, 
                                                const Eigen::Vector3d& finPos,
                                                Eigen::Matrix2d& iniAngleAdjust,
                                                Eigen::Matrix2d& finAngleAdjust,
                                                Eigen::MatrixXd& interAngle);
  void adjustAngleValue(double& begin, 
                    Eigen::VectorXd& intermediate,
                    double& end);
  int optimize(const double& delta = 1e-4);
  bool generate_traj(const Eigen::MatrixXd& iniState,
                     const Eigen::MatrixXd& finState,
                     const Eigen::MatrixXd& iniAngle,
                     const std::vector<Eigen::Vector3d>& target_predcit,
                     const std::vector<Eigen::Vector3d>& target_vels,
                     const std::vector<Eigen::MatrixXd>& view_polys,
                     const std::vector<Eigen::Vector2d>& des_pos_image,
                     const std::vector<Eigen::Vector2d>& des_vel_image,
                     const std::vector<Eigen::MatrixXd>& hPolys,
                     Trajectory& traj);

  void addTimeIntPenalty(const Eigen::VectorXd &T1,
                            const Eigen::MatrixX3d &b,
                            const Eigen::MatrixX2d &b_angle,
                            double &cost,
                            Eigen::VectorXd &gradT,
                            Eigen::MatrixX3d &gradC,
                            Eigen::MatrixX2d &gradC_angle,
                            flatness::FlatnessMap &flatmap);
  void addTimeCost(const Eigen::VectorXd &T,
                            const Eigen::MatrixX3d &b,
                            const Eigen::MatrixX2d &b_angle,
                            double &cost,
                            Eigen::VectorXd &gradT,
                            Eigen::MatrixX3d &gradC,
                            Eigen::MatrixX2d &gradC_angle,
                            flatness::FlatnessMap &flatmap);
  void grad_cost_p_corridor(const Eigen::Vector3d& p,
                            const Eigen::MatrixXd& hPoly,
                            Eigen::Vector3d& gradp,
                            double& costp);
  void grad_cost_v(const Eigen::Vector3d& v,
                   Eigen::Vector3d& gradv,
                   double& costv);
  void grad_cost_a(const Eigen::Vector3d& a,
                   Eigen::Vector3d& grada,
                   double& costa);
  void grad_cost_rate(const Eigen::Vector3d& omg,
                            const double& thetaDot,
                            Eigen::Vector3d& gradomg,
                            double& gradthetaDot,
                            double& cost);
  void grad_cost_safety(const Eigen::Vector3d& v,
                            const double& psi,
                            Eigen::Vector3d& gradv,
                            double& gradpsi,
                            double& cost);
  void grad_cost_actor_safety(const Eigen::Vector3d& p,
                            const Eigen::Vector3d& target_p,
                            Eigen::Vector3d& gradp,
                            double& costp);
  void grad_cost_view(const Eigen::Vector3d& p, 
                            const Eigen::MatrixXd& poly,
                            Eigen::Vector3d& gradp,
                            double& costp);
  void grad_cost_visibility(const Eigen::Vector3d& pos,
                            const Eigen::Vector3d& vel,
                            const Eigen::Vector3d& center,
                            const Eigen::Vector3d& v_center,
                            const Eigen::Vector4d& q,
                            const Eigen::Vector3d& omg,
                            const double& theta,
                            const double& dTheta,
                            const Eigen::Vector2d& des_image_p,
                            const Eigen::Vector2d& des_image_v,
                            Eigen::Vector3d& gradPos,
                            Eigen::Vector3d& gradVel,
                            Eigen::Vector4d& gradQuat,
                            Eigen::Vector3d& gradOmg,
                            double& gradTheta,
                            double& gradThetaDot,
                            double& cost);

  void getJacobian(const Eigen::Vector3d& p,
                            const Eigen::Quaterniond& q,
                            Eigen::MatrixXd& Jacobian);

  void softmax(const Eigen::VectorXd& x,
                            double &ret,
                            Eigen::VectorXd& grad);

};

}  // namespace traj_opt