#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

#define PI 3.14159
#define G  9.8  //Gravity

#define I 0.5	//Inertia, Nm
#define M 1.0	//Mass, Kg
#define R 0.1	//Radius, m
#define B 0.1	//Friction coef.
#define dt 0.01 // sec
#define Tf PI  // final time

time_t start, end;

double get_atheta(double torque, double theta, double vtheta)
{
	return (torque-M*G*R*cos(theta)-B*vtheta)/I;
}

double get_desired_theta(double t)
{
	return PI/2-PI/4*cos(t);
}

int main()
{
	double torque = 0;  // 토크 입력
	double theta=PI/4;  // 초기 위치
	double vtheta=0.0;  // 초기 속도
	double atheta = get_atheta(torque, theta, vtheta); // 각가속도

	double Kp, Kd, Ki;
	// Kp = 25.0;
	// Kd = 10.0;
	// Ki = 0.001;
	Kp = 35.0;
	Kd = 10.0;
	Ki = 0.0;

	double E=0.0;
	double E_dot=0.0;
	double E_prev = 0.0;
	double E_int = 0.0;

	double disired_theta = 0.0;
	double disired_theta_dot = 0.0;

	ofstream fout;
	fout.open("dynamics.txt", ofstream::out);

	double t = 0;
	do
	{
		disired_theta = get_desired_theta(t);
		disired_theta_dot = PI/4*sin(t);

		

		double theta_t = theta;
		double vtheta_t = vtheta;
		double atheta_t = atheta;


		
		E= get_desired_theta(t) - theta;
		E_dot = (E - E_prev)/dt;
		E_int = E_int + E*dt;

		torque = Kp*E + Kd*E_dot + Ki*E_int;

		t = t+dt;

		vtheta = vtheta_t + atheta_t * dt;
		theta = theta_t + vtheta_t * dt + 0.5 * atheta_t * dt*dt;
		atheta = get_atheta(torque, theta, vtheta);


		cout << "t=" << t << " theta=" << theta << " vtheta=" << vtheta << " atheta=" << atheta << endl;
		fout << t << '\t' << theta << '\t' <<disired_theta << '\t' 
							<< vtheta << '\t' << disired_theta_dot << '\t'
							<<torque<<endl;


		E_prev = E;
		
	}while(t <= Tf);
	fout.close();
	return 0;
}

