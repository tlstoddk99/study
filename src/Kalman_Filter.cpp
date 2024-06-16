#include <eigen3/Eigen/Dense>
#include <math.h>
#include <random>
#include <iostream>
#include <vector>
#include "matplotlibcpp.h"

#define STATE_D 2
#define CONTROL_D 1
#define MEASUREMENT_D 1

using namespace std;
using namespace Eigen;
namespace plt = matplotlibcpp;

void init_Matrix(MatrixXd &A, MatrixXd &B, MatrixXd &G, MatrixXd &H, MatrixXd &Q, MatrixXd &R, MatrixXd &P, MatrixXd &K)
{
    A << 1, 1,
        0, 1;
    B << 0.5, 1;
    G << 1, 0,
        0, 1;
    H << 1, 0;
    Q << 0.5, 0,
        0, 0.5;
    R << 10;
    P << 0, 0,
        0, 0;
    K << 0, 0;
}

void init_Vector(VectorXd &x, VectorXd &x_hat, VectorXd &u, VectorXd &z, VectorXd &v, VectorXd &w, double dt)
{
    x << 100, 0;
    x_hat << 60, 0;
    u << -9.81 * dt * dt;
    z << 0;
    v << 0;
    w << 0, 0;
}

void gen_noise(VectorXd &v_or_w, MatrixXd &Q_or_R)
{
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> d(0, 1);
    for (int i = 0; i < v_or_w.size(); i++)
    {
        v_or_w(i) = d(gen) * sqrt(Q_or_R(i, i));
    }
}

void predict(MatrixXd &A, MatrixXd &B, MatrixXd &G, MatrixXd &Q,
             VectorXd &x_hat, MatrixXd &P, VectorXd &u, VectorXd &w, double dt)
{
    gen_noise(w, Q);
    x_hat = (A * x_hat + B * u + G * w);
    P = A * P * A.transpose() + G * Q * G.transpose();
}

void update(MatrixXd &H, MatrixXd &R, MatrixXd &P,
            VectorXd &z, VectorXd &x_hat, MatrixXd &K)
{
    K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
    x_hat = x_hat + K * (z - H * x_hat);
    P = (MatrixXd::Identity(STATE_D, STATE_D) - K * H) * P;
}

void true_update(VectorXd &x, double dt, MatrixXd A, MatrixXd B, VectorXd u)
{
    const double g = 9.81;
    double y = x(0);
    double v = x(1);
    double y_new = y + v * dt - 0.5 * g * dt * dt;
    double v_new = v - g * dt;

    x(0) = y_new;
    x(1) = v_new;
}

void measure_update(VectorXd &z, VectorXd &x, MatrixXd &H, VectorXd &v, MatrixXd &R)
{
    gen_noise(v, R);
    z = H * x + v;
}

int main()
{
    // Define the matrices
    MatrixXd A(STATE_D, STATE_D);
    MatrixXd B(STATE_D, CONTROL_D);
    MatrixXd G(STATE_D, STATE_D);
    MatrixXd H(MEASUREMENT_D, STATE_D);
    MatrixXd Q(STATE_D, STATE_D);
    MatrixXd R(MEASUREMENT_D, MEASUREMENT_D);
    MatrixXd P(STATE_D, STATE_D);
    MatrixXd K(STATE_D, MEASUREMENT_D);

    // Define the vectors
    VectorXd x(STATE_D);
    VectorXd x_hat(STATE_D);
    VectorXd u(CONTROL_D);
    VectorXd z(MEASUREMENT_D);

    // Define the noise vectors
    VectorXd w(STATE_D);
    VectorXd v(MEASUREMENT_D);

    // Define the time steps
    int T = 4;
    double dt = 0.4;
    int index = int(T / dt);

    // Define the vectors to store the data
    vector<double> x_data;
    vector<double> x_hat_data;
    vector<double> z_data;
    vector<double> t_data;

    // Initialize the system
    init_Matrix(A, B, G, H, Q, R, P, K);
    init_Vector(x, x_hat, u, z, v, w, dt);

    // Simulate the system
    for (int t = 0; t < index; t++)
    {
        // Update the true state
        true_update(x, dt, A, B, u);
        x_data.push_back(x(0));

        // Measure the state
        measure_update(z, x, H, v, R);
        z_data.push_back(z(0));

        // Predict the state
        predict(A, B, G, Q, x_hat, P, u, w, dt);

        // Update the state
        update(H, R, P, z, x_hat, K);

        // Store the data
        x_hat_data.push_back(x_hat(0));

        t_data.push_back(t * dt);

        cout << t * dt << "sec-----------------\n\n";
        cout << "x_hat\n"
             << x_hat(0) << "\n\n";
        cout << "P \n"
             << P << "\n\n";
        cout << "K \n"
             << K << "\n";
        cout << "------------------------\n\n\n";
    }

    // Plot the data
    plt::figure();

    plt::title("Kalman Filter");
    plt::xlabel("Time");
    plt::ylabel("Position");
    plt::named_plot("True Position", t_data, x_data);
    plt::named_plot("Measured Position", t_data, z_data);
    plt::named_plot("Estimated Position", t_data, x_hat_data);
    plt::show();

    return 0;
}
