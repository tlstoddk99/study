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

const double L = 0.4;                // Length between front and rear axle
const double maxSteering = M_PI / 8; // Maximum steering angle
const double maxAcceleration = 4.0;  // Maximum acceleration
const double maxDeceleration = 2.0;  // Maximum deceleration

class VehicleModel
{
private:
public:
    Eigen::Vector4d state;
    // Eigen::Matrix4d A;
    // Eigen::Matrix<double, 4, 2> B;
    // initialize the state of the vehicle
    VehicleModel(double initX = 0.0, double initY = 0.0, double initTheta = 0.0, double initV = 0.0)
        : state(initX, initY, initTheta, initV) {}

    // Update the state of the vehicle
    void update(Eigen::Vector4d &state_u, double steer, double a, double dt)
    {
        // // Limit the steering angle
        // steer = max(-maxSteering, min(steer, maxSteering));

        // // Limit the acceleration
        // a = max(-maxDeceleration, min(a, maxAcceleration));

        state_u(0) += state_u(3) * cos(state_u(2)) * dt; // x
        state_u(1) += state_u(3) * sin(state_u(2)) * dt; // y
        state_u(2) += state_u(3) / L * tan(steer) * dt;  // theta
        state_u(2) = fmod(state_u(2), 2.0 * M_PI);       // Normalize theta
        state_u(3) += a * dt;                            // v
    }

    void downdate(Eigen::Vector4d &state_d, double steer, double a, double dt)
    {
        // // Limit the steering angle
        // steer = max(-maxSteering, min(steer, maxSteering));

        // // Limit the acceleration
        // a = max(-maxDeceleration, min(a, maxAcceleration));

        state_d(0) -= state_d(3) * cos(state_d(2)) * dt; // x
        state_d(1) -= state_d(3) * sin(state_d(2)) * dt; // y
        state_d(2) -= state_d(3) / L * tan(steer) * dt;  // theta
        state_d(2) = fmod(state_d(2), 2.0 * M_PI);       // Normalize theta
        state_d(3) -= a * dt;                            // v
    }

    void printState()
    {
        cout << "\tx: " << state(0) << "\ty: " << state(1) << "\tyaw: " << state(2) << "\tv: " << state(3) << "\n\n\n";
    }

    //
};

void compute_optimal_input(vector<Eigen::Vector4d> x_ref, VehicleModel &calcVehicle,
                           vector<double> &delta, vector<double> &a, double dt)
{
    // Define the cost matrices
    Eigen::Matrix4d Q;
    Eigen::Matrix<double, 2, 2> R;

    // Using Discrete Algebraic Riccati Equation to solve for the optimal gain
    Eigen::Matrix4d A;
    Eigen::Matrix<double, 4, 2> B;

    Eigen::Matrix4d H;

    Eigen::Vector4d x_ref_f = x_ref.back();
    // Eigen::Vector4d x_ref_0 = x_ref.front();

    vector<Eigen::Matrix4d> K;
    vector<Eigen::Vector4d> S;
    vector<Eigen::Vector2d> U;

    // init the system
    A << 0, 0, -x_ref_f(3) * sin(x_ref_f(2)), cos(x_ref_f(2)),
        0, 0, x_ref_f(3) * cos(x_ref_f(2)), sin(x_ref_f(2)),
        0, 0, 0, 1.0 / L * tan(0),
        0, 0, 0, 0;
    B << 0, 0,
        0, 0,
        0, x_ref_f(3) / (L * pow(cos(0), 2)),
        0, 1;

    Q << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    R << 1, 0,
        0, 1;

    H << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

    // P.push_back(Eigen::Matrix4d::Zero());
    K.push_back(H);
    S.push_back(-H * x_ref_f);

    Eigen::Vector4d curr_x = x_ref_f;
    double L = 0.4;

    // Compute the P to final state
    for (int i = 0; i < x_ref.size() - 2; i++)
    {
        K.push_back(-((-K[i] * A-A.transpose()*K[i]-Q+K[i]*B*R.inverse()*B.transpose()*K[i]))*dt+K[i]);
        S.push_back(-(-(A.transpose()-K[i]*B*(R.inverse())*B.transpose())*S[i]+Q*x_ref[x_ref.size()-2-i])*dt+S[i]);
        U.push_back(-R.inverse()*B.transpose()*K[i]*curr_x-R.inverse()*B.transpose()*S[i]);
        
    //     // // constraint the input
    //     // U[i](0) = max(-maxSteering, min(U[i](0), maxSteering));
    //     // U[i](1) = max(-maxDeceleration, min(U[i](1), maxAcceleration));

        // update the backward state
        calcVehicle.downdate(curr_x, U[i](0), U[i](1), dt);
        //     myVehicle.printState();
        // update A,B
        // A << 0, 0, -curr_x(3) * sin(curr_x(2)), cos(curr_x(2)),
        //     0, 0, curr_x(3) * cos(curr_x(2)), sin(curr_x(2)),
        //     0, 0, 0, 1.0 / L * tan(U[i](0)),
        //     0, 0, 0, 0;

        // B << 0, 0,
        //     0, 0,
        //     1/(L*cos(U[i](0))*cos(U[i](0))), 0,
        //     0, 1;

        A<<1,0,0,0,
           0,1,0,0,
           0,0,1,0,
           0,0,0,1;
        B<<1,0,
            1,0,
            0,1,
            0,1;
    }

    // reverse the input
    reverse(U.begin(), U.end());

    delta.clear();
    a.clear();

    for (int i = 0; i < U.size(); i++)
    {
        delta.push_back(U[i](0));
        a.push_back(U[i](1));
    }
}
void readCoordinates(const std::string &filename, std::vector<double> &x_ref, std::vector<double> &y_ref)
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
                       std::vector<Eigen::Vector4d> &state_ref,double dt)
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
            state(3) = 0;
        }
        else
        {
            state(2) = std::atan2(y_ref[i + 1] - y_ref[i - 1], x_ref[i + 1] - x_ref[i - 1]);
            state(3)=(sqrt(pow(x_ref[i+1]-x_ref[i],2)+pow(y_ref[i+1]-y_ref[i],2))/dt);
        }

        // Define the velocity
        

        // Store the state
        state_ref.push_back(state);
    }
}
int main()
{

    VehicleModel simVehicle(0, 0, 0, 0);
    VehicleModel calcVehicle(0, 0, 0, 0);
    double steerAngle = 0;
    double acceleration = 0;
    double dt = 0.01;

    vector<double> x, y, u, v;
    ifstream file;
    string SaoPaulo_centerline = "/home/a/mpc_homework/f1tenth_racetracks/SaoPaulo/SaoPaulo_centerline.csv";
    vector<double> x_ref, y_ref;
    vector<double> x_left, y_left;
    vector<double> x_right, y_right;

    vector<double> delta;
    vector<double> accel;

    vector<Eigen::Vector4d> state_ref;

    readCoordinates(SaoPaulo_centerline, x_ref, y_ref);
    getConstraintLane(x_ref, y_ref, x_left, y_left, x_right, y_right);
    getReferenceState(x_ref, y_ref, state_ref,dt);

    const int simTime_sec = x_ref.size() * dt;

    compute_optimal_input(state_ref, calcVehicle, delta, accel, dt);
    

    for (int i = 0; i < int(simTime_sec / dt); i++)
    {
        simVehicle.update(simVehicle.state, delta[i], accel[i], dt);
        // simVehicle.printState();
        cout << "delta: " << delta[i] << "\taccel: " << accel[i] << "\n";
        
        x.push_back(simVehicle.state(0));
        y.push_back(simVehicle.state(1));
        u.push_back(cos(simVehicle.state(2)));
        v.push_back(sin(simVehicle.state(2)));

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
    }

    return 0;
}