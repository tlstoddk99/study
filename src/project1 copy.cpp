#include <stdio.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "matplotlibcpp.h"

using namespace std;
namespace plt = matplotlibcpp;

const double L = 0.4; // Length between front and rear axle
const double V = 1.4;
const double maxSteering = M_PI / 8; // Maximum steering angle
const double maxAcceleration = 4.0;  // Maximum acceleration
const double maxDeceleration = 2.0;  // Maximum deceleration

Eigen::Matrix4d A;
Eigen::Matrix<double, 4, 2> B;

void update_state(Eigen::Vector4d &X, double delta, double a, double dt)
{
    // state(0) += state(3) * cos(state(2)) * dt;
    // state(1) += state(3) * sin(state(2)) * dt;
    // state(2) += state(3) / L * tan(delta) * dt;
    // state(3) += a * dt;

    Eigen::Vector4d X_new;
    Eigen::Vector2d input;

    A << 0, 0, 0, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

    B << 0, 0,
        0, 0,
        V / L, 0,
        0, 1;

    if (delta > maxSteering)
        delta = maxSteering;
    else if (delta < -maxSteering)
        delta = -maxSteering;


    if (a > maxAcceleration)
        a = maxAcceleration;
    else if (a < -maxDeceleration)
        a = -maxDeceleration;


    input << delta, a;

    X_new = (Eigen::Matrix4d::Identity() + A * dt) * X + B * input * dt;

    X = X_new;
}

void compute_optimal_input(vector<Eigen::Vector4d> x_ref, Eigen::Vector4d x_init,
                           vector<Eigen::Vector2d> &U, double dt)
{
    Eigen::Matrix4d H;
    Eigen::Matrix4d Q;
    Eigen::Matrix2d R;

    Eigen::Matrix4d A;
    Eigen::Vector4d X_now;
    Eigen::Matrix<double, 4, 2> B;
    Eigen::Vector2d U_now;

    vector<Eigen::Matrix4d> A_d;
    vector<Eigen::Matrix<double, 4, 2>> B_d;

    vector<Eigen::Matrix4d> K;
    vector<Eigen::Vector4d> S;
    Eigen::Matrix4d K_now;
    Eigen::Vector4d S_now;

    double delta, accel;
    // vector<Eigen::Vector2d> U;

    H << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    Q << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 0;

    R << 1, 0,
        0, 1;

    A << 0, 0, 0, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

    B << 0, 0,
        0, 0,
        V / L, 0,
        0, 1;

    reverse(x_ref.begin(), x_ref.end());
    X_now = x_ref.front();

    K_now = H;
    S_now = -H * X_now;
    cout << "xref: " << x_ref.size() << endl;
    // Compute the optimal input Gain Backward
    for (int i = 0; i < x_ref.size() - 2; i++)
    {
        K.push_back(K_now);
        S.push_back(S_now);

        // K_now = -((-K_now * A_d[i] - A_d[i].transpose() * K_now - Q + K_now * B_d[i] * R.inverse() * B_d[i].transpose() * K_now)) * dt + K_now;
        // S_now = -(-(A_d[i].transpose() - K_now * B_d[i] * (R.inverse()) * B_d[i].transpose()) * S_now + Q * x_ref[i]) * dt + S_now;
        K_now = -((-K_now * A - A.transpose() * K_now - Q + K_now * B * R.inverse() * B.transpose() * K_now)) * dt + K_now;
        S_now = -(-(A.transpose() - K_now * B * (R.inverse()) * B.transpose()) * S_now + Q * x_ref[i]) * dt + S_now;
    }
    cout << "K: " << K.size() << endl;
    K.push_back(K_now);
    S.push_back(S_now);

    reverse(K.begin(), K.end());
    reverse(S.begin(), S.end());

    X_now = x_init;

    for (int i = 0; i < x_ref.size() - 1; i++)
    {
        U_now = -R.inverse() * B.transpose() * K[i] * X_now - R.inverse() * B.transpose() * S[i];

        if (U_now(0) > maxSteering)
            U_now(0) = maxSteering;
        else if (U_now(0) < -maxSteering)
            U_now(0) = -maxSteering;

        if (U_now(1) > maxAcceleration)
            U_now(1) = maxAcceleration;
        else if (U_now(1) < -maxDeceleration)
            U_now(1) = -maxDeceleration;

        U.push_back(U_now);
        delta = U_now(0);
        accel = U_now(1);

        update_state(X_now, delta, accel, dt);
    }
}
void getCoordinates(const std::string &filename, std::vector<double> &x_ref, std::vector<double> &y_ref)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        throw std::runtime_error("Failed to open file.");
    }

    std::string line;

    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        // Extract x coordinate
        if (std::getline(ss, value, ' '))
        {
            x_ref.push_back(std::stod(value));
        }

        // Extract y coordinate
        if (std::getline(ss, value, ' '))
        {
            y_ref.push_back(std::stod(value));
        }

        // Ignore the rest of the line
    }

    file.close();
}
void getConstraintLane(const std::vector<double> &x_ref, const std::vector<double> &y_ref,
                       std::vector<double> &x_left, std::vector<double> &y_left,
                       std::vector<double> &x_right, std::vector<double> &y_right)
{
    // Define the width of the lane
    double lane_width = 3;

    // Define the normal vector
    double normal_x, normal_y;

    // Define the tangent vector
    double tangent_x, tangent_y;

    // Define the distance between the centerline and the left/right lane
    double distance = lane_width / 2;

    // Iterate over the centerline
    for (size_t i = 0; i < x_ref.size(); i++)
    {
        // Compute the tangent vector
        if (i == 0)
        {
            tangent_x = x_ref[i + 1] - x_ref[i];
            tangent_y = y_ref[i + 1] - y_ref[i];
        }
        else if (i == x_ref.size() - 1)
        {
            tangent_x = x_ref[i] - x_ref[i - 1];
            tangent_y = y_ref[i] - y_ref[i - 1];
        }
        else
        {
            tangent_x = x_ref[i + 1] - x_ref[i - 1];
            tangent_y = y_ref[i + 1] - y_ref[i - 1];
        }

        // Normalize the tangent vector
        double norm = std::sqrt(std::pow(tangent_x, 2) + std::pow(tangent_y, 2));
        tangent_x /= norm;
        tangent_y /= norm;

        // Compute the normal vector
        normal_x = -tangent_y;
        normal_y = tangent_x;

        // Compute the left lane
        x_left.push_back(x_ref[i] + distance * normal_x);
        y_left.push_back(y_ref[i] + distance * normal_y);

        // Compute the right lane
        x_right.push_back(x_ref[i] - distance * normal_x);
        y_right.push_back(y_ref[i] - distance * normal_y);
    }
}
void getReferenceState(const std::vector<double> &x_ref, const std::vector<double> &y_ref,
                       std::vector<Eigen::Vector4d> &state_ref, double dt)
{
    // Iterate over the centerline
    for (size_t i = 0; i < x_ref.size(); i++)
    {
        // Define the state
        Eigen::Vector4d state;

        // Define the x and y coordinates
        state(0) = x_ref[i];
        state(1) = y_ref[i];

        // Define the yaw angle
        if (i == 0)
        {
            state(2) = std::atan2(y_ref[i + 1] - y_ref[i], x_ref[i + 1] - x_ref[i]);
            state(3) = 0;
        }
        else if (i == x_ref.size() - 1)
        {
            state(2) = std::atan2(y_ref[i] - y_ref[i - 1], x_ref[i] - x_ref[i - 1]);
            state(3) = (sqrt(pow(x_ref[i] - x_ref[i - 1], 2) + pow(y_ref[i] - y_ref[i - 1], 2)) / dt);
        }
        else
        {
            state(2) = std::atan2(y_ref[i + 1] - y_ref[i - 1], x_ref[i + 1] - x_ref[i - 1]);
            state(3) = (sqrt(pow(x_ref[i + 1] - x_ref[i], 2) + pow(y_ref[i + 1] - y_ref[i], 2)) / dt);
        }

        // Define the velocity

        // Store the state
        state_ref.push_back(state);
    }
}
int main()
{
    double dt = 0.01;
    Eigen::Vector4d x_init;
    vector<double> x, y, u, v;
    ifstream file;
    string SaoPaulo_centerline = "/home/a/mpc_homework/f1tenth_racetracks/SaoPaulo/SaoPaulo_centerline.csv";
    vector<double> x_ref, y_ref;
    vector<double> x_left, y_left;
    vector<double> x_right, y_right;

    vector<Eigen::Vector2d> U;

    vector<Eigen::Vector4d> state_ref;

    getCoordinates(SaoPaulo_centerline, x_ref, y_ref);
    getConstraintLane(x_ref, y_ref, x_left, y_left, x_right, y_right);
    getReferenceState(x_ref, y_ref, state_ref, dt);
    compute_optimal_input(state_ref, x_init, U, dt);

    x_init << state_ref[0](0), state_ref[0](1), state_ref[0](2), state_ref[0](3);
    for (int i = 0; i < x_ref.size()-1; i++)
    {
        x.push_back(x_init(0));
        y.push_back(x_init(1));
        u.push_back(cos(x_init(2)));
        v.push_back(sin(x_init(2)));

        plt::quiver(x, y, u, v);
        plt::plot(x_ref, y_ref, "gray");
        plt::plot(x_left, y_left, "orange");
        plt::plot(x_right, y_right, "orange");

        plt::xlim(-60, 120);
        plt::ylim(-30, 60);

        plt::pause(0.01);
        plt::clf();

        x.clear();
        y.clear();
        u.clear();
        v.clear();

        update_state(x_init, U[i](0), U[i](1), dt);
        cout << "delta: " << U[i](0) << "\taccel: " << U[i](1) << "\n";
        cout<<"x: "<<x_init(0)<<"\ty: "<<x_init(1)<<"\ttheta: "<<x_init(2)<<"\tv: "<<x_init(3)<<endl;
        cout<<"x_ref: "<<x_ref[i]<<"\ty_ref: "<<y_ref[i]<<"\ttheta_ref: "<<state_ref[i](2)<<"\tv_ref: "<<state_ref[i](3)<<endl;
        cout<<"-----------------------------------"<<"\n\n\n";
    }

    return 0;
}