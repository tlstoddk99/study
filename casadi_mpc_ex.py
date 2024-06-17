import casadi as ca
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 100  # 제어 구간의 수
center_w = "/home/a/mpc_homework/map/centerline_waypoints.csv"
df_1 = pd.read_csv(center_w, sep=",", header=None)
x_c = df_1[0].values
y_c = df_1[1].values

left_w = "/home/a/mpc_homework/map/left_waypoints.csv"
df_2 = pd.read_csv(left_w, sep=",", header=None)
x_l = df_2[0].values
y_l = df_2[1].values

right_w = "/home/a/mpc_homework/map/right_waypoints.csv"
df_3 = pd.read_csv(right_w, sep=",", header=None)
x_r = df_3[0].values
y_r = df_3[1].values

spline = "/home/a/mpc_homework/map/center_spline_derivatives.csv"
df_4 = pd.read_csv(spline, sep=",", header=None)
s_x = df_4[0].values
s_y = df_4[1].values

width = "/home/a/mpc_homework/map/track_widths.csv"
df_5 = pd.read_csv(width, sep=",", header=None)
w_l = df_5[0].values
w_r = df_5[1].values

opti = ca.Opti()  # 최적화 문제 설정

# ---- 차량 매개변수 ----
m = 0.35  # 차량 질량 (kg)
Iz = 0.047  # 차량 관성 모멘트 (kg*m^2)
lf = 0.15  # 앞 바퀴와 무게 중심 사이 거리 (m)
lr = 0.17  # 뒷 바퀴와 무게 중심 사이 거리 (m)
Bf, Cf, Df = 1.5, 1.5, 35.0  # Pacejka 타이어 모델 매개변수
Br, Cr, Dr = 1.5, 1.5, 35.0  # Pacejka 타이어 모델 매개변수
Cm1, Cm2, Cd, Croll = 1.0, 0.05, 0.3, 0.05  # 종방향 힘 매개변수

# ---- 결정 변수 ---------
X = opti.variable(7, N + 1)  # 상태 변수 경로
p_x = X[0, :]  # x 위치
p_y = X[1, :]  # y 위치
yaw = X[2, :]  # yaw (방위각)
v_x = X[3, :]  # x 방향 속도
v_y = X[4, :]  # y 방향 속도
omega = X[5, :]  # yaw rate
theta = X[6, :]  # spline length

U = opti.variable(3, N)  # 제어 변수 경로 (스로틀, 스티어링,v_theta)
T = opti.variable()  # 최종 시간

# ---- 목적 함수 ---------
q_c = 1
q_l = 1

opti.minimize(T)  # 최소 시간을 목표로 경주


# ---- 동적 제약 조건 ----
def vehicle_dynamics(x, u):
    alpha_f = -ca.atan2((lf * x[5] + x[4]), x[3]) + u[1]
    alpha_r = ca.atan2((lr * x[5] - x[4]), x[3])

    Ff_y = Df * ca.sin(Cf * ca.arctan(Bf * alpha_f))
    Fr_y = Dr * ca.sin(Cr * ca.arctan(Br * alpha_r))

    Fr_x = (Cm1 - Cm2 * x[3]) * u[0] - Cr - Cd * x[3] ** 2

    dxdt = ca.vertcat(
        x[3] * ca.cos(x[2]) - x[4] * ca.sin(x[2]),  # dx/dt = vx * cos(yaw) - vy * sin(yaw)
        x[3] * ca.sin(x[2]) + x[4] * ca.cos(x[2]),  # dy/dt = vx * sin(yaw) + vy * cos(yaw)
        x[5],  # dyaw/dt = omega
        (Fr_x - Ff_y * ca.sin(u[1]) + m * x[4] * x[5]) / m,  # dvx/dt
        (Fr_y + Ff_y * ca.cos(u[1])) / m,  # dvy/dt
        (Ff_y * lf * ca.cos(u[1]) - Fr_y * lr) / Iz,  # domega/dt
        u[2],  # dtheta/dt
    )
    return dxdt


dt = T / N  # 제어 구간의 길이
for k in range(N):  # 제어 구간을 따라 반복
    # Runge-Kutta 4 차수 적분
    k1 = vehicle_dynamics(X[:, k], U[:, k])
    k2 = vehicle_dynamics(X[:, k] + dt / 2 * k1, U[:, k])
    k3 = vehicle_dynamics(X[:, k] + dt / 2 * k2, U[:, k])
    k4 = vehicle_dynamics(X[:, k] + dt * k3, U[:, k])
    x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X[:, k + 1] == x_next)  # 연속성 제약 조건


# ---- 경로 제약 조건 -----------
def closest_idx(x, y, x_c, y_c):
    # Convert x_c and y_c to CasADi DM (Dense Matrix)
    x_c = ca.DM(x_c)
    y_c = ca.DM(y_c)
    
    # Calculate Euclidean distances using CasADi functions
    distances = ca.sqrt((x - x_c)**2 + (y - y_c)**2)
    
    # Initialize minimum distance and index
    min_dist = distances[0]
    min_idx = 0
    
    # Iterate to find the minimum distance and its index
    for i in range(1, distances.size()[0]):
        min_dist = ca.if_else(distances[i] < min_dist, distances[i], min_dist)
        min_idx = ca.if_else(distances[i] < min_dist, i, min_idx)
    
    return min_dist, min_idx

# Assume N, p_x, p_y, x_c, y_c, and opti are defined earlier
for k in range(N + 1):
    distance, idx = closest_idx(p_x[k], p_y[k], x_c, y_c)
    
    # Define distance as a variable within opti
    dist_var = opti.variable()
    opti.subject_to(dist_var == distance)
    
    # Set the constraint using the variable
    opti.subject_to(opti.bounded(0, dist_var, 0.7))



# ---- 경계 조건 --------
opti.subject_to(p_x[0] == 0)  # 시작 x 위치
opti.subject_to(p_y[0] == 0)  # 시작 y 위치
opti.subject_to(yaw[0] == 1)  # 시작 yaw
opti.subject_to(v_x[0] == 0)  # 시작 x 속도
opti.subject_to(v_y[0] == 0)  # 시작 y 속도
opti.subject_to(omega[0] == 0)  # 시작 yaw rate

opti.subject_to(p_x[-1] == x_c[-1])  # 목표 x 위치
opti.subject_to(p_y[-1] == y_c[-1])  # 목표 y 위치

opti.subject_to(opti.bounded(0, U[0, :], 5))  # 스로틀 제어 변수 제한
opti.subject_to(opti.bounded(-0.4, U[1, :], 0.4))  # 스티어링 제어 변수 제한
# ---- 기타 제약 조건  ----------
opti.subject_to(T >= 0)  # 시간은 양수여야 함
# opti.subject_to(p_x >= 0)  # x 위치는 양수여야 함

# ---- 초기 값 설정 ----
opti.set_initial(p_x, 0)
opti.set_initial(p_y, 0)
opti.set_initial(yaw, 0)
opti.set_initial(v_x, 0.1)
opti.set_initial(v_y, 0)
opti.set_initial(omega, 0)
opti.set_initial(U, 0)
opti.set_initial(T, 1)

# ---- NLP 문제 풀기 ----
opti.solver("ipopt")  # 수치적 솔버 설정

try:
    sol = opti.solve()  # 실제 문제 풀기
except RuntimeError as e:
    print(f"Solver failed: {e}")
    print("Returning initial values for debugging purposes.")
    sol = opti.debug

# ---- 후처리 ----
plt.figure()
# plt.plot(sol.value(p_x), sol.value(p_y), label="trajectory")  # 위치 플롯
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.plot(x_c, y_c, "g", label="centerline")
plt.plot(x_l, y_l, "r", label="left boundary")
plt.plot(x_r, y_r, "r", label="right boundary")
plt.show()

plt.subplot(2, 1, 1)
plt.plot(sol.value(v_x), label="speed vx")  # 속도 플롯
plt.plot(sol.value(v_y), label="speed vy")  # 속도 플롯
plt.xlabel("Time step")
plt.ylabel("Speed")
plt.legend()

plt.subplot(2, 1, 2)
plt.step(range(N), sol.value(U[0, :]), "k", label="throttle")  # 스로틀 플롯
plt.step(range(N), sol.value(U[1, :]), "r", label="steering")  # 스티어링 플롯
plt.xlabel("Time step")
plt.ylabel("Control input")
plt.legend()
plt.show()
