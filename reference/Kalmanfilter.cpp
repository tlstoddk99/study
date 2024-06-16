#include "KalmanFilter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

const float PI2 = 2 * M_PI;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  // 상태 벡터, 오차 공분산 행렬, 상태 전이 행렬, 측정 행렬, 측정 잡음 행렬, 프로세스 잡음 행렬을 초기화합니다.
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // 상태 예측
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // 칼만 필터 업데이트 방정식을 사용하여 상태를 업데이트합니다.

  VectorXd z_pred = H_ * x_;

  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  // 새로운 추정값
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}

VectorXd RadarCartesianToPolar(const VectorXd &x_state){
  /*
   * 레이더 측정치를 카르테시안 좌표(x, y, vx, vy)에서 극좌표(rho, phi, rho_dot)로 변환합니다.
  */
  float px, py, vx, vy;
  px = x_state[0];
  py = x_state[1];
  vx = x_state[2];
  vy = x_state[3];

  float rho, phi, rho_dot;
  rho = sqrt(px*px + py*py);
  phi = atan2(py, px);  // -pi와 pi 사이의 값을 반환합니다.

  // rho가 매우 작으면, 0으로 나누는 것을 방지하기 위해 rho를 0.000001로 설정합니다.
  if(rho < 0.000001)
    rho = 0.000001;

  rho_dot = (px * vx + py * vy) / rho;

  VectorXd z_pred = VectorXd(3);
  z_pred << rho, phi, rho_dot;

  return z_pred;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * 확장 칼만 필터 업데이트 방정식을 사용하여 상태를 업데이트합니다.
  */

  // 레이더 측정치를 카르테시안 좌표(x, y, vx, vy)에서 극좌표(rho, phi, rho_dot)로 변환합니다.
  VectorXd z_pred = RadarCartesianToPolar(x_);
  VectorXd y = z - z_pred;

  // 각도를 -pi에서 pi 사이로 정규화합니다.
  while(y(1) > M_PI){
    y(1) -= PI2;
  }

  while(y(1) < -M_PI){
    y(1) += PI2;
  }

  // KalmanFilter::Update() 함수와 동일한 과정을 수행합니다.
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  // 새로운 추정값
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
