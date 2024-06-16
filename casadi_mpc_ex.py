from casadi import *
import numpy as np
import matplotlib.pyplot as plt

N = 100  # 제어 구간의 수

opti = Opti()  # 최적화 문제 설정

# ---- 차량 매개변수 ----
m = 0.35  # 차량 질량 (kg)
Iz = 0.47  # 차량 관성 모멘트 (kg*m^2)
lf = 0.15  # 앞 바퀴와 무게 중심 사이 거리 (m)
lr = 0.17  # 뒷 바퀴와 무게 중심 사이 거리 (m)
Bf, Cf, Df = 1.5, 1.5, 35.0  # Pacejka 타이어 모델 매개변수
Br, Cr, Dr = 1.5, 1.5, 35.0  # Pacejka 타이어 모델 매개변수
Cm1, Cm2, Cd, Croll = 1.0, 0.05, 0.3, 0.05  # 종방향 힘 매개변수

# ---- 결정 변수 ---------
X = opti.variable(6, N+1)  # 상태 변수 경로
p_x = X[0, :]  # x 위치
p_y = X[1, :]  # y 위치
yaw = X[2, :]  # yaw (방위각)
v_x = X[3, :]  # x 방향 속도
v_y = X[4, :]  # y 방향 속도
omega = X[5, :]  # yaw rate

U = opti.variable(2, N)  # 제어 변수 경로 (스로틀, 스티어링)
T = opti.variable()  # 최종 시간

# ---- 목적 함수 ---------
opti.minimize(T)  # 최소 시간을 목표로 경주

# ---- 동적 제약 조건 ----
def vehicle_dynamics(x, u):
    alpha_f = -atan2((x[4] + lf * x[5]), x[3]) + u[1]
    alpha_r = -atan2((x[4] - lr * x[5]), x[3])
    
    Ff_y = Df * sin(Cf * arctan(Bf * alpha_f))
    Fr_y = Dr * sin(Cr * arctan(Br * alpha_r))
    
    Fx = (Cm1 - Cm2 * x[3]) * u[0] - Cd * x[3]**2 - Croll
    
    dxdt = vertcat(
        x[3] * cos(x[2]) - x[4] * sin(x[2]),  # dx/dt = vx * cos(yaw) - vy * sin(yaw)
        x[3] * sin(x[2]) + x[4] * cos(x[2]),  # dy/dt = vx * sin(yaw) + vy * cos(yaw)
        x[5],  # dyaw/dt = omega
        Fx / m - Ff_y * sin(u[1]) / m + x[4] * x[5],  # dvx/dt
        (Fr_y + Ff_y * cos(u[1])) / m - x[3] * x[5],  # dvy/dt
        (Ff_y * lf * cos(u[1]) - Fr_y * lr) / Iz  # domega/dt
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
track_width = 4.0  # 트랙 폭 (단순화된 모델)
opti.subject_to(p_y <= track_width / 2)  # 트랙 상단 경계
opti.subject_to(p_y >= -track_width / 2)  # 트랙 하단 경계
opti.subject_to(opti.bounded(0, U[0, :], 1))  # 스로틀 제어 변수 제한
opti.subject_to(opti.bounded(-0.4, U[1, :], 0.4))  # 스티어링 제어 변수 제한

# ---- 경계 조건 --------
opti.subject_to(p_x[0] == 0)  # 시작 x 위치
opti.subject_to(p_y[0] == 0)  # 시작 y 위치
opti.subject_to(yaw[0] == 0)  # 시작 yaw
opti.subject_to(v_x[0] == 0)  # 시작 x 속도
opti.subject_to(v_y[0] == 0)  # 시작 y 속도
opti.subject_to(omega[0] == 0)  # 시작 yaw rate

opti.subject_to(p_x[-1] == 3)  # 목표 x 위치
opti.subject_to(p_y[-1] == 0)  # 목표 y 위치

# ---- 기타 제약 조건  ----------
opti.subject_to(T >= 0)  # 시간은 양수여야 함

# ---- 초기 값 설정 ----
opti.set_initial(p_x, np.linspace(0, 3, N+1))
opti.set_initial(p_y, np.linspace(0, 0, N+1))
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
plt.subplot(3, 1, 1)
plt.plot(sol.value(p_x), sol.value(p_y), label="trajectory")  # 위치 플롯
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(sol.value(v_x), label="speed vx")  # 속도 플롯
plt.plot(sol.value(v_y), label="speed vy")  # 속도 플롯
plt.xlabel('Time step')
plt.ylabel('Speed')
plt.legend()

plt.subplot(3, 1, 3)
plt.step(range(N), sol.value(U[0, :]), 'k', label="throttle")  # 스로틀 플롯
plt.step(range(N), sol.value(U[1, :]), 'r', label="steering")  # 스티어링 플롯
plt.xlabel('Time step')
plt.ylabel('Control input')
plt.legend()

plt.show()
