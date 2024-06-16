#ifndef KALMANFILTER_H
#define KALMANFILTER_H
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "matplotlibcpp.h"

class KalmanFilter {
public:

  // 상태 벡터
  Eigen::VectorXd x_;

  // 상태 공분산 행렬
  Eigen::MatrixXd P_;

  // 상태 전이 행렬
  Eigen::MatrixXd F_;

  // 프로세스 공분산 행렬
  Eigen::MatrixXd Q_;

  // 측정 행렬
  Eigen::MatrixXd H_;

  // 측정 공분산 행렬
  Eigen::MatrixXd R_;

  /**
   * 생성자
   */
  KalmanFilter();

  /**
   * 소멸자
   */
  virtual ~KalmanFilter();

  /**
   * 칼만 필터 초기화
   * @param x_in 초기 상태
   * @param P_in 초기 상태 공분산
   * @param F_in 전이 행렬
   * @param H_in 측정 행렬
   * @param R_in 측정 공분산 행렬
   * @param Q_in 프로세스 공분산 행렬
   */
  void Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in, Eigen::MatrixXd &F_in,
      Eigen::MatrixXd &H_in, Eigen::MatrixXd &R_in, Eigen::MatrixXd &Q_in);

  /**
   * 예측
   * 프로세스 모델을 사용하여 상태와 상태 공분산을 예측
   * @param delta_T k와 k+1 사이의 시간 (초)
   */
  void Predict();

  /**
   * 표준 칼만 필터 방정식을 사용하여 상태 업데이트
   * @param z k+1에서의 측정값
   */
  void Update(const Eigen::VectorXd &z);

  /**
   * 확장 칼만 필터 방정식을 사용하여 상태 업데이트
   * @param z k+1에서의 측정값
   */
  void UpdateEKF(const Eigen::VectorXd &z);

};

#endif /* KALMAN_FILTER_H_ */
