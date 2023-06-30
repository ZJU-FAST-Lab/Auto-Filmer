#include "ros/time.h"
#include <iostream>
#include <traj_opt/traj_opt.h>

#include <random>
#include <traj_opt/geoutils.hpp>
#include <traj_opt/lbfgs.hpp>

namespace traj_opt {

// SECTION  variables transformation and gradient transmission
static double expC2(double t) {
  return t > 0.0 ? ((0.5 * t + 1.0) * t + 1.0)
                 : 1.0 / ((0.5 * t - 1.0) * t + 1.0);
}
static double logC2(double T) {
  return T > 1.0 ? (sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T - 1.0));
}
static void forwardT(const Eigen::Ref<const Eigen::VectorXd>& t, const double& sT, Eigen::Ref<Eigen::VectorXd> vecT) {
  int M = t.size();
  for (int i = 0; i < M; ++i) {
    vecT(i) = expC2(t(i));
  }
  // TODO: why?
  vecT(M) = 0.0;
  vecT /= 1.0 + vecT.sum();
  vecT(M) = 1.0 - vecT.sum();
  vecT *= sT;
  return;
}
static void backwardT(const Eigen::Ref<const Eigen::VectorXd>& vecT, Eigen::Ref<Eigen::VectorXd> t) {
  int M = t.size();
  // useless
  t = vecT.head(M) / vecT(M);
  for (int i = 0; i < M; ++i) {
    t(i) = logC2(vecT(i));
  }
  return;
}
static void addLayerTGrad(const Eigen::Ref<const Eigen::VectorXd>& t,
                          const double& sT,
                          const Eigen::Ref<const Eigen::VectorXd>& gradT,
                          Eigen::Ref<Eigen::VectorXd> gradt) {
  int Ms1 = t.size();
  Eigen::VectorXd gFree = sT * gradT.head(Ms1);
  double gTail = sT * gradT(Ms1);
  Eigen::VectorXd dExpTau(Ms1);
  double expTauSum = 0.0, gFreeDotExpTau = 0.0;
  double denSqrt, expTau;
  for (int i = 0; i < Ms1; i++) {
    if (t(i) > 0) {
      expTau = (0.5 * t(i) + 1.0) * t(i) + 1.0;
      dExpTau(i) = t(i) + 1.0;
      expTauSum += expTau;
      gFreeDotExpTau += expTau * gFree(i);
    } else {
      denSqrt = (0.5 * t(i) - 1.0) * t(i) + 1.0;
      expTau = 1.0 / denSqrt;
      dExpTau(i) = (1.0 - t(i)) / (denSqrt * denSqrt);
      expTauSum += expTau;
      gFreeDotExpTau += expTau * gFree(i);
    }
  }
  denSqrt = expTauSum + 1.0;
  gradt = (gFree.array() - gTail) * dExpTau.array() / denSqrt -
          (gFreeDotExpTau - gTail * expTauSum) * dExpTau.array() / (denSqrt * denSqrt);
}

static void forwardP(const Eigen::Ref<const Eigen::VectorXd>& p,
                     const std::vector<Eigen::MatrixXd>& cfgPolyVs,
                     Eigen::MatrixXd& inP) {
  int M = cfgPolyVs.size();
  Eigen::VectorXd q;
  int j = 0, k;
  for (int i = 0; i < M; ++i) {
    k = cfgPolyVs[i].cols() - 1;
    q = 2.0 / (1.0 + p.segment(j, k).squaredNorm()) * p.segment(j, k);
    inP.col(i) = cfgPolyVs[i].rightCols(k) * q.cwiseProduct(q) +
                 cfgPolyVs[i].col(0);
    j += k;
  }
  return;
}
static double objectiveNLS(void* ptrPOBs,
                           const double* x,
                           double* grad,
                           const int n) {
  const Eigen::MatrixXd& pobs = *(Eigen::MatrixXd*)ptrPOBs;
  Eigen::Map<const Eigen::VectorXd> p(x, n);
  Eigen::Map<Eigen::VectorXd> gradp(grad, n);

  double qnsqr = p.squaredNorm();
  double qnsqrp1 = qnsqr + 1.0;
  double qnsqrp1sqr = qnsqrp1 * qnsqrp1;
  Eigen::VectorXd r = 2.0 / qnsqrp1 * p;

  Eigen::Vector3d delta = pobs.rightCols(n) * r.cwiseProduct(r) +
                          pobs.col(1) - pobs.col(0);
  double cost = delta.squaredNorm();
  Eigen::Vector3d gradR3 = 2 * delta;

  Eigen::VectorXd gdr = pobs.rightCols(n).transpose() * gradR3;
  gdr = gdr.array() * r.array() * 2.0;
  gradp = gdr * 2.0 / qnsqrp1 -
          p * 4.0 * gdr.dot(p) / qnsqrp1sqr;

  return cost;
}

static void backwardP(const Eigen::Ref<const Eigen::MatrixXd>& inP,
                      const std::vector<Eigen::MatrixXd>& cfgPolyVs,
                      Eigen::VectorXd& p) {
  int M = inP.cols();
  int j = 0, k;

  // Parameters for tiny nonlinear least squares
  double minSqrD;
  lbfgs::lbfgs_parameter_t nls_params;
  lbfgs::lbfgs_load_default_parameters(&nls_params);
  nls_params.g_epsilon = FLT_EPSILON;
  nls_params.max_iterations = 128;

  Eigen::MatrixXd pobs;
  for (int i = 0; i < M; i++) {
    k = cfgPolyVs[i].cols() - 1;
    p.segment(j, k).setConstant(1.0 / (sqrt(k + 1.0) + 1.0));
    pobs.resize(3, k + 2);
    pobs << inP.col(i), cfgPolyVs[i];
    lbfgs::lbfgs_optimize(k,
                          p.data() + j,
                          &minSqrD,
                          &objectiveNLS,
                          nullptr,
                          nullptr,
                          &pobs,
                          &nls_params);
    j += k;
  }
  return;
}
static void addLayerPGrad(const Eigen::Ref<const Eigen::VectorXd>& p,
                          const std::vector<Eigen::MatrixXd>& cfgPolyVs,
                          const Eigen::Ref<const Eigen::MatrixXd>& gradInPs,
                          Eigen::Ref<Eigen::VectorXd> grad) {
  int M = gradInPs.cols();

  int j = 0, k;
  double qnsqr, qnsqrp1, qnsqrp1sqr;
  Eigen::VectorXd q, r, gdr;
  for (int i = 0; i < M; i++) {
    k = cfgPolyVs[i].cols() - 1;
    q = p.segment(j, k);
    qnsqr = q.squaredNorm();
    qnsqrp1 = qnsqr + 1.0;
    qnsqrp1sqr = qnsqrp1 * qnsqrp1;
    r = 2.0 / qnsqrp1 * q;
    gdr = cfgPolyVs[i].rightCols(k).transpose() * gradInPs.col(i);
    gdr = gdr.array() * r.array() * 2.0;

    grad.segment(j, k) = gdr * 2.0 / qnsqrp1 -
                         q * 4.0 * gdr.dot(q) / qnsqrp1sqr;
    j += k;
  }
  return;
}
// !SECTION variables transformation and gradient transmission

// SECTION object function
static inline double objectiveFunc(void* ptrObj,
                                   const double* x,
                                   double* grad,
                                   const int n) {
//   std::cout << "***************************" << std::endl;  //debug only
  TrajOpt& obj = *(TrajOpt*)ptrObj;

  Eigen::Map<const Eigen::VectorXd> t(x, obj.dim_t_);
  Eigen::Map<const Eigen::VectorXd> p(x + obj.dim_t_, obj.dim_p_);
  Eigen::Map<const Eigen::VectorXd> psi(x + obj.dim_t_ + obj.dim_p_, obj.dim_psi_);
  Eigen::Map<const Eigen::VectorXd> theta(x + obj.dim_t_ + obj.dim_p_ + obj.dim_psi_, obj.dim_theta_);
  Eigen::Map<Eigen::VectorXd> gradt(grad, obj.dim_t_);
  Eigen::Map<Eigen::VectorXd> gradp(grad + obj.dim_t_, obj.dim_p_);
  Eigen::Map<Eigen::VectorXd> gradpsi(grad + obj.dim_t_ + obj.dim_p_, obj.dim_psi_);
  Eigen::Map<Eigen::VectorXd> gradtheta(grad + obj.dim_t_ + obj.dim_p_ + obj.dim_psi_, obj.dim_theta_);
  double deltaT = x[obj.dim_t_ + obj.dim_p_ + obj.dim_psi_ + obj.dim_theta_];

  obj.dbg_msg.header.stamp = ros::Time::now();
  obj.dbg_msg.cost_smooth = 0.0;
  obj.dbg_msg.cost_smooth_angle = 0.0;
  obj.dbg_msg.cost_corridor = 0.0;
  obj.dbg_msg.cost_v = 0.0;
  obj.dbg_msg.cost_a = 0.0;
  obj.dbg_msg.cost_omg = 0.0;
  obj.dbg_msg.cost_mapping = 0.0;
  obj.dbg_msg.cost_view = 0.0;
  obj.dbg_msg.cost_actor_safe = 0.0;
  obj.dbg_msg.cost_visibility = 0.0;

  Eigen::VectorXd T(obj.N_);
  Eigen::MatrixXd P(3, obj.N_ - 1);
  Eigen::VectorXd Yaw(obj.N_ - 1);
  Eigen::VectorXd Theta(obj.N_ - 1);
  // T_sigma = T_s + deltaT^2
  double sumT = obj.sum_T_ + deltaT * deltaT;
  forwardT(t, sumT, T);
  forwardP(p, obj.cfgVs_, P);
  Yaw = psi;
  Theta = theta;

  //setup
  obj.mincoOpt_.setParameters(P, Yaw, Theta, T); 
  double cost = 0.0;
  obj.partialGradByCoeffs.resize(8 * obj.N_, 3);
  obj.partialGradByCoeffsAngle.resize(4 * obj.N_, 2);
  obj.partialGradByTimes.resize(obj.N_);
  obj.gradByPoints.resize(3, obj.N_ - 1);
  obj.gradByTimes.resize(obj.N_);
  obj.gradByAngles.resize(2, obj.N_ - 1);

  obj.partialGradByTimes.setZero();
  obj.partialGradByCoeffs.setZero();
  obj.partialGradByCoeffsAngle.setZero();
  obj.gradByPoints.setZero();
  obj.gradByAngles.setZero();
  obj.gradByTimes.setZero(); 

  obj.dbg_msg.cost_smooth = cost;
  obj.mincoOpt_.getEnergy(cost, obj.rhoSmooth_);
  obj.dbg_msg.cost_smooth = cost - obj.dbg_msg.cost_smooth;
  obj.mincoOpt_.getEnergyPartialGradByCoeffs(obj.partialGradByCoeffs, obj.rhoSmooth_);
  obj.mincoOpt_.getEnergyPartialGradByTimes(obj.partialGradByTimes, obj.rhoSmooth_);
//   std::cout << "spatial smooth: " << cost <<std::endl;
  obj.dbg_msg.cost_smooth_angle = cost;
  obj.mincoOpt_.getEnergyAngle(cost, obj.rhoAngleSmooth_);
  obj.dbg_msg.cost_smooth_angle = cost - obj.dbg_msg.cost_smooth_angle;
  obj.mincoOpt_.getEnergyPartialGradByCoeffsAngle(obj.partialGradByCoeffsAngle, obj.rhoAngleSmooth_);
  obj.mincoOpt_.getEnergeyPartialGradByTimesAngle(obj.partialGradByTimes, obj.rhoAngleSmooth_);
//   std::cout << "angular smooth: " << cost <<std::endl;

  Eigen::MatrixX3d b = obj.mincoOpt_.getCoeffs();
  Eigen::MatrixX2d b_angle = obj.mincoOpt_.getCoeffsAngle();
  
  // this function add intergral penalty like pos, vel and acc
  obj.addTimeIntPenalty(T, b, b_angle, cost, obj.partialGradByTimes, 
                            obj.partialGradByCoeffs, obj.partialGradByCoeffsAngle, obj.flatmap_);
  
  obj.addTimeCost(T, b, b_angle, cost, obj.partialGradByTimes, obj.partialGradByCoeffs, 
                            obj.partialGradByCoeffsAngle, obj.flatmap_);

  obj.mincoOpt_.propogateGrad(obj.partialGradByCoeffs, obj.partialGradByCoeffsAngle, obj.partialGradByTimes, 
                                        obj.gradByPoints, obj.gradByAngles, obj.gradByTimes);

  grad[obj.dim_t_ + obj.dim_p_ + obj.dim_psi_ + obj.dim_theta_] = obj.gradByTimes.dot(T) / sumT + obj.rhoT_;
  cost += obj.rhoT_ * deltaT * deltaT;
  grad[obj.dim_t_ + obj.dim_p_ + obj.dim_psi_ + obj.dim_theta_] *= 2 * deltaT;
  addLayerTGrad(t, sumT, obj.gradByTimes, gradt);
  addLayerPGrad(p, obj.cfgVs_, obj.gradByPoints, gradp);
  gradpsi = obj.gradByAngles.row(0).transpose();  //simply push grad yaws into it
  gradtheta = obj.gradByAngles.row(1).transpose();  //simply push grad thetas into it
  return cost;
}
// !SECTION object function

static inline int earlyExit(void* ptrObj,
                            const double* x,
                            const double* grad,
                            const double fx,
                            const double xnorm,
                            const double gnorm,
                            const double step,
                            int n,
                            int k,
                            int ls) {
  return k > 1e3;
}

bool TrajOpt::extractVs(const std::vector<Eigen::MatrixXd>& hPs,
                        std::vector<Eigen::MatrixXd>& vPs) const {
  const int M = hPs.size() - 1;

  vPs.clear();
  vPs.reserve(2 * M + 1);

  int nv;
  Eigen::MatrixXd curIH, curIV, curIOB;
  for (int i = 0; i < M; i++) {
    if (!geoutils::enumerateVs(hPs[i], curIV)) {
      return false;
    }
    nv = curIV.cols();
    curIOB.resize(3, nv);
    curIOB << curIV.col(0), curIV.rightCols(nv - 1).colwise() - curIV.col(0);
    vPs.push_back(curIOB);

    curIH.resize(6, hPs[i].cols() + hPs[i + 1].cols());
    curIH << hPs[i], hPs[i + 1];
    if (!geoutils::enumerateVs(curIH, curIV)) {
      return false;
    }
    nv = curIV.cols();
    curIOB.resize(3, nv);
    curIOB << curIV.col(0), curIV.rightCols(nv - 1).colwise() - curIV.col(0);
    vPs.push_back(curIOB);
  }

  if (!geoutils::enumerateVs(hPs.back(), curIV)) {
    return false;
  }
  nv = curIV.cols();
  curIOB.resize(3, nv);
  curIOB << curIV.col(0), curIV.rightCols(nv - 1).colwise() - curIV.col(0);
  vPs.push_back(curIOB);

  return true;
}

TrajOpt::TrajOpt(ros::NodeHandle& nh) : nh_(nh) {
  // nh.getParam("N", N_);
  nh.getParam("K", K_);
  // load dynamic paramters
  nh.getParam("vmax", vmax_);
  nh.getParam("amax", amax_);
  nh.getParam("ratemax", ratemax_);
  nh.getParam("rate_max_frame", ratemax_frame_);
  nh.getParam("rhoSmooth", rhoSmooth_);
  nh.getParam("rhoAngleSmooth", rhoAngleSmooth_);
  nh.getParam("rhoT", rhoT_);
  nh.getParam("rhoP", rhoP_);
  nh.getParam("rhoV", rhoV_);
  nh.getParam("rhoA", rhoA_);
  nh.getParam("rhoOmg", rhoOmg_);
  nh.getParam("rhoMapping", rhoMapping_);
  nh.getParam("rhoNoKilling", rhoNoKilling_);
  nh.getParam("rhoOcclusion", rhoOcclusion_);
  nh.getParam("rhoViewPos",rhoViewPos_);
  nh.getParam("rhoVisibilityPos", rhoVisibilityPos_);
  nh.getParam("rhoVisibilityVel", rhoVisibilityVel_);
  nh.getParam("alpha_clearance", alpha_clearance_);
  nh.getParam("tracking_dur", tracking_dur_);
  nh.getParam("tracking_dist", tracking_dist_);
  nh.getParam("tracking_dt", tracking_dt_);
  nh.getParam("clearance_d", clearance_d_);
  nh.getParam("tolerance_d", tolerance_d_);
  nh.getParam("kill_d", kill_d_);
  std::vector<double> tmp;
  if (nh.param<std::vector<double>>("cam2body_R", tmp, std::vector<double>())) {
    cam2body_R_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 3);
  }
  if (nh.param<std::vector<double>>("cam2body_p", tmp, std::vector<double>())) {
    cam2body_p_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 1);
  }
  nh.getParam("cam_fx", fx_);
  nh.getParam("cam_fy", fy_);
  nh.getParam("cam_cx", cx_);
  nh.getParam("cam_cy", cy_);
  nh.getParam("cam_width", cam_width_);
  nh.getParam("cam_height", cam_height_);

  dbg_pub_ = nh.advertise<quadrotor_msgs::OptDebug>("opt_debug", 10);

  pos_image << cx_, cy_;
  vel_image << 0.0, 0.0;
  int knots = static_cast<int>(tracking_dur_/tracking_dt_) + 1;
  plan_pos_image.resize(knots);
  plan_vel_image.resize(knots);
  for(int i = 0; i < knots; i++)
  {
      plan_pos_image[i] = pos_image;
      plan_vel_image[i] = vel_image;
  }
}

void TrajOpt::setBoundConds(const Eigen::MatrixXd& iniState,
                            const Eigen::MatrixXd& finState,
                            const Eigen::MatrixXd& iniAngle) {
  Eigen::MatrixXd initS = iniState;
  Eigen::MatrixXd finalS = finState;
  double tempNorm = initS.col(1).norm();
  initS.col(1) *= tempNorm > vmax_ ? (vmax_ / tempNorm) : 1.0;
  tempNorm = finalS.col(1).norm();
  finalS.col(1) *= tempNorm > vmax_ ? (vmax_ / tempNorm) : 1.0;
  tempNorm = initS.col(2).norm();
  initS.col(2) *= tempNorm > amax_ ? (amax_ / tempNorm) : 1.0;
  tempNorm = finalS.col(2).norm();
  finalS.col(2) *= tempNorm > amax_ ? (amax_ / tempNorm) : 1.0;

  Eigen::VectorXd T(N_);
  T.setConstant(sum_T_ / N_);
  backwardT(T, t_);
  Eigen::MatrixXd P(3, N_ - 1);
  for (int i = 0; i < N_ - 1; ++i) {
    int k = cfgVs_[i].cols() - 1;
    P.col(i) = cfgVs_[i].rightCols(k).rowwise().sum() / (1.0 + k) + cfgVs_[i].col(0);
  }
  backwardP(P, cfgVs_, p_);
  // init value of psi
  Eigen::Matrix2d iniAngleAdjust, finAngleAdjust;
  Eigen::MatrixXd interAngle;

  Eigen::Vector3d finPos;
  finPos = finState.col(0);
  setAngleInitValue(iniAngle, T, P, finPos, iniAngleAdjust, finAngleAdjust, interAngle);
  psi_ = interAngle.row(0).transpose();
  theta_ = interAngle.row(1).transpose();
  mincoOpt_.setConditions(initS, finalS, iniAngleAdjust, finAngleAdjust, N_);
//   std::cout << "init values" << std::endl;
//   std::cout << "P: " << initS.col(0) << P << finalS.col(0) << std::endl;
//   std::cout << "T: " << T << std::endl;
//   std::cout << "psi: " << iniAngleAdjust(0,0) << psi_ << finAngleAdjust(0,0) << std::endl;
//   std::cout << "theta: " << iniAngleAdjust(1,0) << theta_ << finAngleAdjust(1,0) << std::endl;

  Eigen::VectorXd physicalParams(6);
// params from px4 control
  physicalParams(0) = 0.61;
  physicalParams(1) = 9.8;
  physicalParams(2) = 0.10;
  physicalParams(3) = 0.23;
  physicalParams(4) = 0.01;
  physicalParams(5) = 0.02;  
  flatmap_.reset(physicalParams(0), physicalParams(1), physicalParams(2),
                      physicalParams(3), physicalParams(4), physicalParams(5));
  return;
}

void TrajOpt::setAngleInitValue(const Eigen::MatrixXd& iniAngle,
                                                const Eigen::VectorXd& T,
                                                const Eigen::MatrixXd& P, 
                                                const Eigen::Vector3d& finPos,
                                                Eigen::Matrix2d& iniAngleAdjust,
                                                Eigen::Matrix2d& finAngleAdjust,
                                                Eigen::MatrixXd& interAngle)
{
    Eigen::VectorXd interYaw, interTheta;
    interYaw.resize(N_ - 1);
    interTheta.resize(N_ - 1);
    // step 1. yaw: velocity direction
    double yaw;
    double yaw_before = iniAngle(0,0);
    for(int i = 0; i < N_ - 1 ;i ++)
    {
        Eigen::Vector3d pos, pos_next;
        pos = P.col(i);
        if (i == (N_ - 2) )
        {
            pos_next = finPos;
        }
        else
        {
            pos_next = P.col(i + 1);
        }
        // too close
        double thresh = 1e-5;
        if( std::fabs(pos(1) - pos_next(1)) < thresh &&
                std::fabs(pos(0) - pos_next(0)) < thresh)
        {
            yaw = yaw_before;
        }
        else 
        {
            yaw = std::atan2(pos_next(1) - pos(1), pos_next(0) - pos(0));     
        }
        interYaw(i) = yaw;
    }
    double end_yaw = interYaw(N_ - 2);
    // pi problem
    double begin_yaw = iniAngle(0, 0);
    adjustAngleValue(begin_yaw, interYaw, end_yaw);

    //step 2. theta: yaw + theta = pointing to the target
    int i_pre = 0;
    double t_pos = 0.0;
    double theta_abs;
    for(int i = 0; i < N_ - 1 ;i ++)
    {
        t_pos += T(i);
        Eigen::Vector3d pos, p_pre;
        pos = P.col(i);
        // theta: heading to the target
        while(fabs(tracking_dt_ * i_pre - t_pos) >= fabs(tracking_dt_ * (i_pre + 1) - t_pos) && (i_pre + 1) < tracking_ps_.size())
        {
            i_pre += 1;
        }
        p_pre = tracking_ps_.at(i_pre);
        theta_abs = std::atan2(p_pre(1) - pos(1), p_pre(0) - pos(0));
        interTheta(i) = theta_abs - interYaw(i);
    }
    Eigen::Vector3d fin_target = tracking_ps_.back();
    theta_abs = std::atan2(fin_target(1) - finPos(1) , fin_target(0) - finPos(0));
    double end_theta = theta_abs - end_yaw;
    double begin_theta = iniAngle(1, 0);
    adjustAngleValue(begin_theta, interTheta, end_theta);

    //step 3. load values
    interAngle.resize(2, N_ - 1);    
    iniAngleAdjust = iniAngle;
    iniAngleAdjust(0, 0) = begin_yaw;
    iniAngleAdjust(1, 0) = begin_theta;
    finAngleAdjust.setZero();
    finAngleAdjust(0, 0) = end_yaw;
    finAngleAdjust(1, 0) = end_theta;
    interAngle.row(0) = interYaw.transpose();
    interAngle.row(1) = interTheta.transpose();
}

void TrajOpt::adjustAngleValue(double& begin, 
                    Eigen::VectorXd& intermediate,
                    double& end)
{
    // function to avoid pi problem
    double before = begin;
    while(before > M_PI || before < -M_PI)
    {
        if(before > M_PI)
            before -= 2 * M_PI;
        else if(before < -M_PI)
            before += 2 * M_PI;
    }
    begin = before;
    for(int i = 0; i < intermediate.size(); i++)
    {
        double delta1 = fmod(intermediate(i) - before, 2  * M_PI);
        double delta2 = delta1 < 0 ? (delta1 + 2 * M_PI) : (delta1 - 2 * M_PI);
        double delta = fabs(delta1) < fabs(delta2) ? delta1 : delta2;
        intermediate(i) = before + delta;
        before = intermediate(i);
    }
    double delta1 = fmod(end - before, 2  * M_PI);
    double delta2 = delta1 < 0 ? (delta1 + 2 * M_PI) : (delta1 - 2 * M_PI);
    double delta = fabs(delta1) < fabs(delta2) ? delta1 : delta2;
    end = before + delta;
}

int TrajOpt::optimize(const double& delta) {
  // Setup for L-BFGS solver
  lbfgs::lbfgs_parameter_t lbfgs_params;
  lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
  lbfgs_params.mem_size = 16;
  lbfgs_params.past = 3;
  lbfgs_params.g_epsilon = 1e-10; 
  lbfgs_params.min_step = 1e-32;
  lbfgs_params.delta = delta;
  Eigen::Map<Eigen::VectorXd> t(x_, dim_t_);
  Eigen::Map<Eigen::VectorXd> p(x_ + dim_t_, dim_p_);
  Eigen::Map<Eigen::VectorXd> psi(x_ + dim_t_ + dim_p_, dim_psi_);
  Eigen::Map<Eigen::VectorXd> theta(x_ + dim_t_ + dim_p_ + dim_psi_, dim_theta_);
  t = t_;
  p = p_;
  psi = psi_;
  theta = theta_;
  double minObjective;
  ros::Time t1 = ros::Time::now();
  auto ret = lbfgs::lbfgs_optimize(dim_t_ + dim_p_ + dim_psi_ + dim_theta_ + 1, x_, &minObjective,
                                   &objectiveFunc, nullptr,
                                   &earlyExit, this, &lbfgs_params);
  std::cout << "\033[32m"
            << "ret: " << ret << "\033[0m" << std::endl;
  ros::Time t2 = ros::Time::now();
  double opt_t = (t2- t1).toSec() * 1e3;
  dbg_pub_.publish(dbg_msg);

  if (ret != lbfgs::LBFGS_CONVERGENCE &&
        ret != lbfgs::LBFGSERR_MAXIMUMITERATION &&
        ret != lbfgs::LBFGS_ALREADY_MINIMIZED &&
        ret != lbfgs::LBFGS_STOP)
    {
        std::cout <<  lbfgs::lbfgs_strerror(ret) << std::endl;
    }
  t_ = t;
  p_ = p;
  psi_ = psi;
  theta_ = theta;
  return ret;
}

bool TrajOpt::generate_traj(const Eigen::MatrixXd& iniState,
                            const Eigen::MatrixXd& finState,
                            const Eigen::MatrixXd& iniAngle,
                            const std::vector<Eigen::Vector3d>& target_predcit,
                            const std::vector<Eigen::Vector3d>& target_vels,
                            const std::vector<Eigen::MatrixXd>& view_polys,
                            const std::vector<Eigen::Vector2d>& des_pos_image,
                            const std::vector<Eigen::Vector2d>& des_vel_image,
                            const std::vector<Eigen::MatrixXd>& hPolys,
                            Trajectory& traj) {
  cfgHs_ = hPolys;
  if (cfgHs_.size() == 1) {
    cfgHs_.push_back(cfgHs_[0]);
  }
//   std::cout << "****************************" << std::endl;
//   for(const auto &Poly : hPolys)
//     {
//         std::cout << Poly << std::endl;
//     }
  if (!extractVs(cfgHs_, cfgVs_)) {
    ROS_ERROR("extractVs fail!");
    return false;
  }

  N_ = 2 * cfgHs_.size(); 
  // NOTE wonderful trick
  sum_T_ = tracking_dur_;

  // NOTE: one corridor two pieces
  dim_t_ = N_ - 1;
  dim_p_ = 0;
  for (const auto& cfgV : cfgVs_) {
    dim_p_ += cfgV.cols() - 1;  //cfgV matrixXd
  }
  dim_psi_ = N_ - 1;
  dim_theta_ = N_ - 1;
  p_.resize(dim_p_);
  t_.resize(dim_t_);
  psi_.resize(dim_psi_);
  theta_.resize(dim_theta_);

  x_ = new double[dim_t_ + dim_p_ + dim_psi_ + dim_theta_ + 1];
  Eigen::VectorXd T(N_);
  Eigen::MatrixXd P(3, N_ - 1);

  tracking_ps_ = target_predcit;
  tracking_vs_ = target_vels;
  plan_pos_image = des_pos_image;
  plan_vel_image = des_vel_image;
  view_polys_ = view_polys;

  setBoundConds(iniState, finState, iniAngle);

  x_[dim_t_ + dim_p_ + dim_psi_ + dim_theta_] = 0.1;

  int opt_ret = optimize();
  if (opt_ret < 0) {
    return false;
  }

  double sumT = sum_T_ + x_[dim_t_ + dim_p_ + dim_psi_ + dim_theta_] * x_[dim_t_ + dim_p_ + dim_psi_ + dim_theta_];
  forwardT(t_, sumT, T);
  forwardP(p_, cfgVs_, P);
  mincoOpt_.setParameters(P, psi_, theta_, T);
  // std::cout << "P: \n" << P << std::endl;
  // std::cout << "T: " << T.transpose() << std::endl;

  mincoOpt_.getTrajectory(traj);
  delete[] x_;
  return true;
}


void TrajOpt::addTimeIntPenalty(const Eigen::VectorXd &T1,
                                                const Eigen::MatrixX3d &b,
                                                const Eigen::MatrixX2d &b_angle,
                                                double &cost,
                                                Eigen::VectorXd &gradT,
                                                Eigen::MatrixX3d &gradC,
                                                Eigen::MatrixX2d &gradC_angle,
                                                flatness::FlatnessMap &flatmap) 
{
    // state variables
  Eigen::Vector3d pos, vel, acc, jer, sna;
  Eigen::Vector2d angle, angle_rate, angle_acc;
  double psi, dPsi, ddPsi, theta, dTheta, ddTheta, thr;
  Eigen::Vector4d quat;
  Eigen::Vector3d bodyrate;
  // partial gradient
  Eigen::Vector3d gradPos, gradVel, gradAcc, gradOmg;
  double gradThr, gradTheta, gradThetaDot;
  Eigen::Vector4d gradQuat;
  // total gradient
  Eigen::Vector3d totalGradPos, totalGradVel, totalGradAcc, totalGradJer;
  double totalGradPsi, totalGradPsiD;
  Eigen::Vector2d totalGradAngle, totalGradAngleRate;
  // assistant variables
  Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3, beta4;
  Eigen::Matrix<double, 4, 1> betaAngle0, betaAngle1, betaAngle2; 
  double s1, s2, s3, s4, s5, s6, s7;
  double step, alpha;
  double omg, pena;
  int innerLoop;
  for (int i = 0; i < N_; ++i) {
    const Eigen::Matrix<double, 8, 3> &c = b.block<8, 3>(i * 8, 0);
    const Eigen::Matrix<double, 4, 2> &c_angle = b_angle.block<4, 2>(i * 4, 0);
    step = T1(i) / K_;
    s1 = 0.0;
    innerLoop = K_ + 1;

    const auto& hPoly = cfgHs_[i / 2];
    for (int j = 0; j < innerLoop; ++j) {
      alpha = 1.0 / K_ * j;
      s1 = j * step;
      s2 = s1 * s1;
      s3 = s2 * s1;
      s4 = s2 * s2;
      s5 = s4 * s1;
      s6 = s4 * s2;
      s7 = s4 * s3;
      beta0 << 1.0, s1, s2, s3, s4, s5, s6, s7;
      beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4, 6 * s5, 7 * s6;
      beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3, 30 * s4, 42 * s5;
      beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2, 120 * s3, 210 * s4;
      beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * s1, 360 * s2, 840 * s3;
      pos = c.transpose() * beta0;
      vel = c.transpose() * beta1;
      acc = c.transpose() * beta2;
      jer = c.transpose() * beta3;
      sna = c.transpose() * beta4;
      betaAngle0 << 1.0, s1, s2, s3;
      betaAngle1 << 0.0, 1.0, 2.0*s1, 3.0*s2;
      betaAngle2 << 0.0, 0.0, 2.0, 6.0 * s1;
      angle = c_angle.transpose() * betaAngle0;
      angle_rate = c_angle.transpose() * betaAngle1;
      angle_acc = c_angle.transpose() * betaAngle2;
      psi = angle(0); theta = angle(1);
      dPsi = angle_rate(0); dTheta = angle_rate(1);
      ddPsi = angle_acc(0); ddTheta = angle_acc(1);

      flatmap.forward(vel, acc, jer, psi, dPsi, thr, quat, bodyrate);
      pena = 0.0;
      gradPos.setZero();
      gradVel.setZero();
      gradAcc.setZero();
      gradThr = 0.0;  //no use
      gradQuat.setZero();  // no use
      gradOmg.setZero();
      gradTheta = 0.0;  // no use
      gradThetaDot = 0.0;

      omg = (j == 0 || j == innerLoop - 1) ? 0.5 : 1.0;

      double cost_corridor, cost_v, cost_a, cost_omg, cost_mapping;
      // penalty p, v, a and omega
      cost_corridor = pena;
      grad_cost_p_corridor(pos, hPoly, gradPos, pena);
      cost_corridor = pena - cost_corridor;

      cost_v = pena;
      grad_cost_v(vel, gradVel, pena);
      cost_v = pena - cost_v;

      cost_a = pena;
      grad_cost_a(acc, gradAcc, pena);
      cost_a = pena - cost_a;

      cost_omg = pena;
      grad_cost_rate(bodyrate, dTheta, gradOmg, gradThetaDot, pena);  
      cost_omg = pena - cost_omg;

      flatmap.backward(gradPos, gradVel, gradAcc, gradThr, gradQuat, gradOmg,
                                     totalGradPos, totalGradVel, totalGradAcc, totalGradJer,
                                     totalGradPsi, totalGradPsiD);

      // some gradient are in flatmap.backward
      cost_mapping = pena;
      grad_cost_safety(vel, psi, totalGradVel, totalGradPsi, pena);
      cost_mapping = pena - cost_mapping;
      
      totalGradAngle << totalGradPsi, gradTheta;
      totalGradAngleRate << totalGradPsiD, gradThetaDot;
      
      gradC.block<8, 3>(i * 8, 0) += (beta0 * totalGradPos.transpose() +
                                                    beta1 * totalGradVel.transpose() +
                                                    beta2 * totalGradAcc.transpose() +
                                                    beta3 * totalGradJer.transpose()) 
                                                   * omg * step;
      
      gradC_angle.block<4, 2>(i * 4, 0) += (betaAngle0 * totalGradAngle.transpose() +
                                                    betaAngle1 * totalGradAngleRate.transpose())
                                                    * omg * step; 
      gradT(i) += (totalGradPos.dot(vel) +
                        totalGradVel.dot(acc) +
                        totalGradAcc.dot(jer) +
                        totalGradJer.dot(sna) +
                        totalGradAngle.dot(angle_rate) + 
                        totalGradAngleRate.dot(angle_acc))
                        * alpha * omg * step +
                        omg / K_ * pena;
      cost += omg * step * pena;

      dbg_msg.cost_corridor += omg * step * cost_corridor;
      dbg_msg.cost_v += omg * step * cost_v;
      dbg_msg.cost_a += omg * step * cost_a;
      dbg_msg.cost_omg += omg * step * cost_omg;
      dbg_msg.cost_mapping += omg * step * cost_mapping;
    }

    // if(i==0)
    // {
    //     const double supress_initial_jerk = 100.0;
    //     cost+=supress_initial_jerk*c.row(3).squaredNorm()/36.0;
    //     gradC.row(3)+=supress_initial_jerk*c.row(3)/18.0;
    // }
    
  }
}

void TrajOpt::addTimeCost(const Eigen::VectorXd &T,
                                        const Eigen::MatrixX3d &b,
                                        const Eigen::MatrixX2d &b_angle,
                                        double &cost,
                                        Eigen::VectorXd &gradT,
                                        Eigen::MatrixX3d &gradC,
                                        Eigen::MatrixX2d &gradC_angle,
                                        flatness::FlatnessMap &flatmap) 
{
//   std::cout << "\033[31m######################################################\033[0m" << std::endl;
  // assistant variables
  int piece = 0;
  int M = tracking_ps_.size() * 4 / 5;
  double t = 0;
  double t_pre = 0;
  double step = tracking_dt_;
  double pena;

  // state variables
  Eigen::Vector3d pos, vel, acc, jer, sna;
  Eigen::Vector2d angle, angle_rate, angle_acc;
  double psi, dPsi, ddPsi, theta, dTheta, ddTheta, thr;
  Eigen::Vector4d quat;
  Eigen::Vector3d bodyrate;
  // assistant variables
  Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3, beta4;
  Eigen::Matrix<double, 4, 1> betaAngle0, betaAngle1, betaAngle2; 
  double s1, s2, s3, s4, s5, s6, s7;
  // partial gradient
  Eigen::Vector3d gradPos, gradVel, gradAcc, gradOmg;
  double gradThr, gradTheta, gradThetaDot;
  Eigen::Vector4d gradQuat;
  // total gradient
  Eigen::Vector3d totalGradPos, totalGradVel, totalGradAcc, totalGradJer;
  double totalGradPsi, totalGradPsiD;
  Eigen::Vector2d totalGradAngle, totalGradAngleRate;

  for (int i = 0; i < M; ++i) {
    t = i * step;
    double rho = exp2(-3.0 * i / M);
    while (t - t_pre > T(piece)) {
      t_pre += T(piece);
      piece++;
    }
    s1 = t - t_pre;
    s2 = s1 * s1;
    s3 = s2 * s1;
    s4 = s2 * s2;
    s5 = s4 * s1;
    s6 = s4 * s2;
    s7 = s4 * s3;
    beta0 << 1.0, s1, s2, s3, s4, s5, s6, s7;
    beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4, 6 * s5, 7 * s6;
    beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3, 30 * s4, 42 * s5;
    beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2, 120 * s3, 210 * s4;
    beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * s1, 360 * s2, 840 * s3;
    betaAngle0 << 1.0, s1, s2, s3;
    betaAngle1 << 0.0, 1.0, 2.0*s1, 3.0*s2;
    betaAngle2 << 0.0, 0.0, 2.0, 6.0*s1;

    const Eigen::Matrix<double, 8,  3>& c = b.block<8, 3>(piece * 8, 0);
    const Eigen::Matrix<double, 4, 2> &c_angle = b_angle.block<4, 2>(piece * 4, 0);
    pos = c.transpose() * beta0;
    vel = c.transpose() * beta1;
    acc = c.transpose() * beta2;
    jer = c.transpose() * beta3;
    sna = c.transpose() * beta4;
    angle = c_angle.transpose() * betaAngle0;
    angle_rate = c_angle.transpose() * betaAngle1;
    angle_acc = c_angle.transpose() * betaAngle2;
    psi = angle(0); theta = angle(1);
    dPsi = angle_rate(0); dTheta = angle_rate(1);
    ddPsi = angle_acc(0); ddTheta = angle_acc(1);
    flatmap.forward(vel, acc, jer, psi, dPsi, thr, quat, bodyrate);

    pena = 0.0;
    gradPos.setZero();
    gradVel.setZero();
    gradAcc.setZero();
    gradThr = 0.0; 
    gradQuat.setZero(); 
    gradOmg.setZero();
    gradTheta = 0.0;
    gradThetaDot = 0.0;

    Eigen::Vector3d target_p = tracking_ps_[i];
    Eigen::Vector3d target_v = tracking_vs_[i];
    Eigen::MatrixXd view_region = view_polys_[i];

    double cost_view, cost_actor_safe, cost_visibility;

    //penalty tracking distance, visibility and observability
    cost_view = pena;
    grad_cost_view(pos, view_region, gradPos, pena);
    cost_view = pena - cost_view;

    cost_actor_safe = pena;
    grad_cost_actor_safety(pos, target_p, gradPos, pena);
    cost_actor_safe = pena - cost_actor_safe;

    //target position and velocity on image plane
    cost_visibility = pena;
    Eigen::Vector2d des_image_pos, des_image_vel;
    des_image_pos = plan_pos_image[i];
    des_image_vel = plan_vel_image[i];
    grad_cost_visibility(pos, vel, target_p, target_v, quat, bodyrate, theta, dTheta,
                                    des_image_pos, des_image_vel,
                                    gradPos, gradVel, gradQuat, gradOmg, gradTheta, gradThetaDot, pena);
    cost_visibility = pena - cost_visibility;

    flatmap.backward(gradPos, gradVel, gradAcc, gradThr, gradQuat, gradOmg,
                                     totalGradPos, totalGradVel, totalGradAcc, totalGradJer,
                                     totalGradPsi, totalGradPsiD);
    
    totalGradAngle << totalGradPsi, gradTheta;
    totalGradAngleRate << totalGradPsiD, gradThetaDot;

    gradC.block<8, 3>(piece * 8, 0) += (beta0 * totalGradPos.transpose() +
                                                    beta1 * totalGradVel.transpose() +
                                                    beta2 * totalGradAcc.transpose() +
                                                    beta3 * totalGradJer.transpose()) 
                                                    * rho * step;
    gradC_angle.block<4, 2>(piece * 4, 0) += (betaAngle0 * totalGradAngle.transpose() +
                                                    betaAngle1 * totalGradAngleRate.transpose())
                                                    * rho * step;
    if(piece > 0)
    {
        gradT.head(piece).array() += (totalGradPos.dot(vel) +
                                                    totalGradVel.dot(acc) +
                                                    totalGradAcc.dot(jer) +
                                                    totalGradJer.dot(sna) +
                                                    totalGradAngle.dot(angle_rate) +
                                                    totalGradAngleRate.dot(angle_acc))
                                                    * (-rho * step);
    } 
    cost += rho * step * pena; 

    dbg_msg.cost_view += rho * step * cost_view;
    dbg_msg.cost_actor_safe += rho * step * cost_actor_safe;
    dbg_msg.cost_visibility += rho * step * cost_visibility;
  }
}

void TrajOpt::grad_cost_p_corridor(const Eigen::Vector3d& p,
                                   const Eigen::MatrixXd& hPoly,
                                   Eigen::Vector3d& gradp,
                                   double& costp) {
  // return false;
  for (int i = 0; i < hPoly.cols(); ++i) {
    Eigen::Vector3d norm_vec = hPoly.col(i).head<3>();
    double pen = norm_vec.dot(p - hPoly.col(i).tail<3>() + clearance_d_ * norm_vec);
    if (pen > 0) {
      double pen2 = pen * pen;
      gradp += rhoP_ * 3 * pen2 * norm_vec;
      costp += rhoP_ * pen2 * pen;
    }
  }
  return;
}

static double penF(const double& x, double& grad) {
  static double eps = 0.05;
  static double eps2 = eps * eps;
  static double eps3 = eps * eps2;
  if (x < 2 * eps) {
    double x2 = x * x;
    double x3 = x * x2;
    double x4 = x2 * x2;
    grad = 12 / eps2 * x2 - 4 / eps3 * x3;
    return 4 / eps2 * x3 - x4 / eps3;
  } else {
    grad = 16;
    return 16 * (x - eps);
  }
}

static double sigmoidF(const double& x, const double &lb, const double& hb, double& grad){
    // y = 1 / (1 + exp(x))  lower: y(-5) = 0.0067
    // y = max / (1 + exp(- alpha * (x - offset)))

    assert(hb > lb);
    double offset = (lb + hb) / 2;
    double alpha = 10 / (hb - offset);
    double exp_part = exp(- alpha * (x - offset));
    double denom = 1 / (1 + exp_part);
    double ret = denom;
    grad = alpha * exp_part * denom * denom;
    return ret;
}

void TrajOpt::grad_cost_v(const Eigen::Vector3d& v,
                          Eigen::Vector3d& gradv,
                          double& costv) {
  double vpen = v.squaredNorm() - vmax_ * vmax_;
  if (vpen > 0) {
    gradv += rhoV_ * 6 * vpen * vpen * v;
    costv += rhoV_ * vpen * vpen * vpen;
  }
  return;
}

void TrajOpt::grad_cost_a(const Eigen::Vector3d& a,
                          Eigen::Vector3d& grada,
                          double& costa) {
  double apen = a.squaredNorm() - amax_ * amax_;
  if (apen > 0) {
    grada += rhoA_ * 6 * apen * apen * a;
    costa += rhoA_ * apen * apen * apen;
  }
  return;
}

void TrajOpt::grad_cost_rate(const Eigen::Vector3d& omg,
                                const double& thetaDot,
                                Eigen::Vector3d& gradomg,
                                double& gradthetaDot,
                                double& cost)
{
    // ratemax: thetaDot & yawDot
    // rateframe_max: roll, pitch, actual yaw
    Eigen::VectorXd max_vec, rate_vec;
    Eigen::MatrixXd gradtmp;
    max_vec.resize(5);
    rate_vec.resize(5);
    gradtmp.resize(5, 4);
    max_vec << 1.5 * ratemax_, 1.5 * ratemax_, ratemax_, ratemax_frame_, ratemax_frame_;
    rate_vec << omg(2), thetaDot, omg(2) + thetaDot, omg(0), omg(1);
    gradtmp.col(0) << 0.0, 0.0, 0.0, 2 * omg(0), 0.0;
    gradtmp.col(1) << 0.0, 0.0, 0.0, 0.0, 2 * omg(1);
    gradtmp.col(2) << 2 * omg(2), 0.0, 2 * (omg(2) + thetaDot), 0.0, 0.0;
    gradtmp.col(3) << 0.0, 2 * thetaDot, 2 * (omg(2) + thetaDot), 0.0, 0.0;
    for(int i = 0; i < 5; i++)
    {
        double pen = rate_vec(i) * rate_vec(i) - max_vec(i) * max_vec(i);
        if( pen > 0.0)
        {
            double pen2 = pen * pen;
            cost += rhoOmg_ * pen *  pen2;
            gradomg += rhoOmg_ * 3 * pen2 * gradtmp.block<1, 3>(i, 0).transpose();
            gradthetaDot += rhoOmg_ * 3 * pen2 * gradtmp(i, 3);
        }
    }
    return;
}

void TrajOpt::grad_cost_safety(const Eigen::Vector3d& v,
                        const double& psi,
                        Eigen::Vector3d& gradv,
                        double& gradpsi,
                        double& cost)
{
    double eps = 1e-5;
    Eigen::Vector2d a, b, grada, gradb, grada_psi;
    a << cos(psi), sin(psi);
    b << v(0), v(1);
    double inner_product = a.dot(b);
    double norm_a = a.norm();
    double norm_b = sqrt(b(0)*b(0) + b(1)*b(1) + eps);
    double pen = 1 - inner_product / norm_a /norm_b;

    cost += rhoMapping_ * pen;
    grada = -((norm_a * norm_b * b) - inner_product * norm_b / norm_a * a) / (norm_a * norm_a * norm_b * norm_b);
    gradb = -((norm_a * norm_b * a) - inner_product * norm_a / norm_b * b) / (norm_a * norm_a * norm_b * norm_b);
    gradv(0) += rhoMapping_ * gradb(0);
    gradv(1) += rhoMapping_ * gradb(1);
    grada_psi << -sin(psi), cos(psi);
    gradpsi += rhoMapping_ * grada_psi.dot(grada);
}

void TrajOpt::grad_cost_view(const Eigen::Vector3d& p, 
                                    const Eigen::MatrixXd& poly,
                                    Eigen::Vector3d& gradp,
                                    double& costp)
{
    for (int i = 0; i < poly.cols(); ++i) {
    Eigen::Vector3d norm_vec = poly.col(i).head<3>();
    double pen = norm_vec.dot(p - poly.col(i).tail<3>());
    if (pen > 0) {
      double pen2 = pen * pen;
      gradp += rhoViewPos_ * 3 * pen2 * norm_vec;
      costp += rhoViewPos_ * pen2 * pen;
    }
  }
  return;
}

void TrajOpt::grad_cost_actor_safety(const Eigen::Vector3d& p,
                                   const Eigen::Vector3d& target_p,
                                   Eigen::Vector3d& gradp,
                                   double& costp) {
  double lower = 1.0;  // 1m away
  lower = lower * lower;
  Eigen::Vector3d dp = (p - target_p);
  double dr2 = dp.head(2).squaredNorm();

  double pen = lower - dr2;
  if (pen > 0) {
    double pen2 = pen * pen;
    gradp.head(2) -= 6 * pen2 * dp.head(2) * rhoNoKilling_;
    costp += pen2 * pen * rhoNoKilling_;
  }

  return;
}

void TrajOpt::grad_cost_visibility(const Eigen::Vector3d& pos,
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
                                            double& cost)
{
    double des_u, des_v, des_ut, des_vt;
    des_u = des_image_p(0);
    des_v = des_image_p(1);
    des_ut = des_image_v(0);
    des_vt = des_image_v(1);
    // pc_center = Rcb_fixed * Rcb(theta) * (Rbw * (center - p) - pbc)
    const Eigen::Quaterniond qcb_fixed(cam2body_R_.transpose());
    // const Eigen::Matrix3d Rbc(cam2body_R_);
    const Eigen::Vector3d pbc = cam2body_p_;
    Eigen::Quaterniond qbw, qcb, q_theta; 
    //q.inverse();
    qbw.w() = q(0); qbw.x() = -q(1); qbw.y() = -q(2); qbw.z() = -q(3); 
    q_theta = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_theta_inv = q_theta.inverse();
    qcb = qcb_fixed * q_theta_inv;
    Eigen::Matrix3d Rbc = q_theta.toRotationMatrix() * cam2body_R_;
    //1. position on the image
    Eigen::Vector3d pc_center;
    double pena_u, pena_v, u, v;
    pc_center = qcb * (qbw * (center - pos) - pbc); 
    u = pc_center.x() / pc_center.z() * fx_ + cx_;
    v = pc_center.y() / pc_center.z() * fy_ + cy_;
    pena_u = u - des_u;
    pena_v = v - des_v;
    double pen = pena_u * pena_u + pena_v * pena_v;
    double grad_tmp;
    cost += rhoVisibilityPos_ * sigmoidF(pen, - cx_ * cx_ , cx_ * cx_, grad_tmp);  // low: 10 pixel, up : cx
    // gradient propagation
    double grad_cost_u, grad_cost_v; 
    grad_cost_u = rhoVisibilityPos_ * 2 * pena_u * grad_tmp;
    grad_cost_v = rhoVisibilityPos_ * 2 * pena_v * grad_tmp;
    double grad_u_x, grad_u_z, grad_v_y, grad_v_z;
    Eigen::Vector3d grad_cost_pc;
    Eigen::Vector4d grad_cost_q, grad_cost_qtheta_inverse;
    Eigen::Matrix3d grad_pc_pos;
    Eigen::MatrixXd grad_pb_qbw, grad_pc_qbw;
    Eigen::MatrixXd grad_pb_qtheta, grad_pc_qtheta_inv;
    grad_u_x = fx_ / pc_center.z();
    grad_u_z =  -pc_center.x() / pc_center.z() / pc_center.z() * fx_;
    grad_v_y = fy_ / pc_center.z();
    grad_v_z = -pc_center.y() / pc_center.z() / pc_center.z() * fy_;
    grad_cost_pc(0) = grad_cost_u * grad_u_x;
    grad_cost_pc(1) = grad_cost_v * grad_v_y;
    grad_cost_pc(2) = grad_cost_u * grad_u_z + grad_cost_v * grad_v_z;

    grad_pc_pos = -(qcb * qbw).toRotationMatrix().transpose();
    gradPos += grad_pc_pos * grad_cost_pc;

    grad_pb_qbw.resize(4, 3);
    grad_pc_qbw.resize(4, 3);
    getJacobian(center - pos, qbw, grad_pb_qbw);
    grad_pc_qbw = grad_pb_qbw * Rbc;
    // inverse q
    grad_cost_q = - grad_pc_qbw * grad_cost_pc;
    grad_cost_q(0) *= -1;
    gradQuat += grad_cost_q;

    grad_pb_qtheta.resize(4, 3);
    grad_pc_qtheta_inv.resize(4, 3);
    getJacobian(qbw * (center - pos) - pbc, q_theta_inv, grad_pb_qtheta);
    grad_pc_qtheta_inv = grad_pb_qtheta * cam2body_R_;
    grad_cost_qtheta_inverse = grad_pc_qtheta_inv * grad_cost_pc;
    Eigen::Vector4d grad_qinverse_theta;  //actually transpose
    grad_qinverse_theta << -sin(theta/2)/2.0, 0.0, 0.0, -cos(theta/2)/2.0;
    gradTheta += grad_qinverse_theta.dot(grad_cost_qtheta_inverse);

    // 2. velocity on the image
    // vc_center = Rcb_fixed * [ -S_dTheta * R^-1(theta) * (Rbw * (center - p) - pbc) 
    //             + R^-1(theta) * (S(omg)^T * Rbw * (center - p) + Rbw * (v_center - v)) ] 

    double ut, vt;
    Eigen::Vector3d vc_center_1, vc_center_2, vc_center;
    Eigen::Matrix3d S_omg, S_dTheta;
    S_omg << 0.0, -omg(2), omg(1),
                    omg(2), 0.0, -omg(0),
                    -omg(1), omg(0), 0.0;
    S_dTheta << 0.0, -dTheta, 0.0,
                    dTheta, 0.0, 0.0,
                    0.0, 0.0, 0.0;
    vc_center_1 = -S_dTheta * q_theta_inv * (qbw * (center - pos) - pbc);
    vc_center_2 = q_theta_inv * (-S_omg * qbw * (center - pos) + qbw * (v_center - vel));
    vc_center = qcb_fixed * (vc_center_1 + vc_center_2);

    double z2 = pc_center(2) * pc_center(2);
    ut = vc_center(0) * pc_center(2) - pc_center(0) * vc_center(2);
    ut *= fx_ / z2;
    vt = vc_center(1) * pc_center(2) - pc_center(1) * vc_center(2);
    vt *= fy_ / z2;
    double pena_ut, pena_vt;
    pena_ut = ut - des_ut;
    pena_vt = vt - des_vt;
    pen = pena_ut * pena_ut + pena_vt * pena_vt;
    cost += rhoVisibilityVel_ * sigmoidF(pen, 400, 150*150, grad_tmp);  // low: 20 pixel/s  up: 150 pixel/s
    double grad_costv_ut, grad_costv_vt;
    grad_costv_ut = rhoVisibilityVel_ * 2 * pena_ut * grad_tmp;
    grad_costv_vt = rhoVisibilityVel_ * 2 * pena_vt * grad_tmp;

    // gradient propagation
    Eigen::Vector3d grad_costv_pc, grad_costv_vc, grad_ut_pc, grad_vt_pc, grad_ut_vc, grad_vt_vc;
    grad_ut_pc(0) = -vc_center(2) / pc_center(2) / pc_center(2);
    grad_ut_pc(1) = 0.0;
    grad_ut_pc(2) = (-vc_center(0) + 2 * pc_center(0) * vc_center(2) / pc_center(2)) / z2;
    grad_ut_pc *= fx_;
    grad_vt_pc(0) = 0.0;
    grad_vt_pc(1) = -vc_center(2) / pc_center(2) / pc_center(2);
    grad_vt_pc(2) = (-vc_center(1) + 2 * pc_center(1) * vc_center(2) / pc_center(2)) / z2;
    grad_vt_pc *= fy_;
    grad_ut_vc(0) = 1.0 / pc_center(2);
    grad_ut_vc(1) = 0.0;
    grad_ut_vc(2) = -pc_center(0) / z2;
    grad_ut_vc *= fx_;
    grad_vt_vc(0) = 0.0;
    grad_vt_vc(1) = 1.0 / pc_center(2);
    grad_vt_vc(2) = -pc_center(1) / z2;
    grad_vt_vc *= fy_;
    grad_costv_pc = grad_costv_ut * grad_ut_pc + grad_costv_vt * grad_vt_pc;
    grad_costv_vc = grad_costv_ut * grad_ut_vc + grad_costv_vt * grad_vt_vc;
    
    Eigen::Vector3d pb_center = qbw * (center - pos);
    Eigen::Vector4d grad_costv_qbw, grad_costv_q;
    Eigen::Matrix3d grad_vc_pos, grad_vc_vel, grad_vb_omg, grad_vc_omg;
    Eigen::MatrixXd grad_vc_qbw, grad_vb_qbw;

    grad_vc_pos = ( qcb_fixed * S_dTheta * q_theta_inv * qbw + qcb * S_omg * qbw).transpose();
    grad_vc_vel = -(qcb * qbw).toRotationMatrix().transpose();
    // vc_center = Rcb* (- omg x Rbw * (center - p) + Rbw * (v_center - v)) 
    grad_vb_omg.row(0) = Eigen::Vector3d(1,0,0).cross(pb_center).transpose();
    grad_vb_omg.row(1) = Eigen::Vector3d(0,1,0).cross(pb_center).transpose();
    grad_vb_omg.row(2) = Eigen::Vector3d(0,0,1).cross(pb_center).transpose();
    grad_vc_omg = -grad_vb_omg * Rbc;
    grad_vc_qbw.resize(4, 3);
    grad_vb_qbw.resize(4, 3);
    getJacobian(v_center - vel, qbw, grad_vb_qbw);
    grad_vc_qbw = grad_pb_qbw * S_omg * Rbc
                            + grad_vb_qbw * Rbc
                            + grad_pb_qbw * (qcb_fixed * (-S_dTheta) * q_theta_inv).transpose();
    
    Eigen::Vector3d pb_theta_center = q_theta_inv * (pb_center - pbc);
    Eigen::Vector3d grad_vb_dTheta = Eigen::Vector3d(0,0,1).cross(pb_theta_center);
    Eigen::Vector3d grad_vc_dTheta = -grad_vb_dTheta.transpose() * cam2body_R_;
    Eigen::MatrixXd grad_vb_qtheta, grad_vc_qtheta;
    grad_vb_qtheta.resize(4, 3);
    grad_vc_qtheta.resize(4, 3);
    getJacobian(-S_omg * qbw * (center - pos) + qbw * (v_center - vel), 
                            q_theta_inv, grad_vb_qtheta);
    grad_vc_qtheta = grad_vb_qtheta * cam2body_R_
                                    + grad_pb_qtheta * S_dTheta * cam2body_R_;

    gradPos += grad_pc_pos * grad_costv_pc + grad_vc_pos * grad_costv_vc;
    gradVel +=  grad_vc_vel * grad_costv_vc;
    grad_costv_qbw = grad_vc_qbw * grad_costv_vc + grad_pc_qbw * grad_costv_pc;
    grad_costv_q = -grad_costv_qbw;
    grad_costv_q(0) *= -1;
    gradQuat += grad_costv_q;
    gradOmg += grad_vc_omg * grad_costv_vc; 
    gradTheta += grad_qinverse_theta.dot(grad_vc_qtheta * grad_costv_vc + grad_pc_qtheta_inv * grad_costv_pc);
    gradThetaDot += grad_vc_dTheta.dot(grad_costv_vc);

}



/*
* function getJacobian
* for a quaternion rotation: r = q*p*q^(-1)
* returns dr_dq
*/
void TrajOpt::getJacobian(const Eigen::Vector3d& p,
                                    const Eigen::Quaterniond& q,
                                    Eigen::MatrixXd& Jacobian)
{
    Jacobian.resize(4,3);
    Jacobian.row(0) << p(0)*q.w() + p(2)*q.y() - p(1)*q.z(), p(1)*q.w() + p(0)*q.z() - p(2)*q.x(), p(1)*q.x() - p(0)*q.y() + p(2)*q.w();
    Jacobian.row(1) << p(0)*q.x() + p(1)*q.y() + p(2)*q.z(), p(0)*q.y() - p(1)*q.x() - p(2)*q.w(), p(1)*q.w() + p(0)*q.z() - p(2)*q.x();
    Jacobian.row(2) << p(1)*q.x() - p(0)*q.y() + p(2)*q.w(), p(0)*q.x() + p(1)*q.y() + p(2)*q.z(), p(1)*q.z() - p(0)*q.w() - p(2)*q.y();
    Jacobian.row(3) << p(2)*q.x() - p(0)*q.z() - p(1)*q.w(), p(0)*q.w() - p(1)*q.z() + p(2)*q.y(), p(0)*q.x() + p(1)*q.y() + p(2)*q.z();
    Jacobian = 2*Jacobian;
}

  void TrajOpt::softmax(const Eigen::VectorXd& x,
                            double &ret,
                            Eigen::VectorXd& grad)
  {
    const double alpha = 10;
    Eigen::VectorXd exp_alpha_x;
    exp_alpha_x.resize(x.size());
    grad.resize(x.size());
    for(int i = 0; i < x.size(); i++)
        exp_alpha_x(i) = exp(alpha * x(i));
    ret = exp_alpha_x.dot(x) / exp_alpha_x.sum();
    for(int i = 0; i < x.size(); i++)
    {
        grad(i) = 1 + alpha * (x(i) - ret);
        grad(i) *= exp(alpha * x(i));
    }
    grad /= exp_alpha_x.sum();
  }

}  // namespace traj_opt
