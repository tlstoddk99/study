from casadi import *

N = 100  # 제어 구간의 수

opti = Opti()  # 최적화 문제 설정

# ---- 결정 변수 ---------
X = opti.variable(4, N+1)  # 상태 변수 경로
p_x = X[0, :]  # x 위치
p_y = X[1, :]  # y 위치
yaw = X[2, :]  # yaw (방위각)
v = X[3, :]    # 속도
U = opti.variable(2, N)    # 제어 변수 경로 (스로틀, 스티어링)
T = opti.variable()        # 최종 시간

# ---- 목적 함수 ---------
opti.minimize(T)  # 최소 시간을 목표로 경주

# ---- 동적 제약 조건 --------
f = lambda x, u: vertcat(
    x[3] * cos(x[2]),    # dx/dt = v * cos(yaw)
    x[3] * sin(x[2]),    # dy/dt = v * sin(yaw)
    u[1],                # dyaw/dt = steering
    u[0] - x[3]          # dv/dt = throttle - v (감속 포함)
)

dt = T / N  # 제어 구간의 길이
for k in range(N):  # 제어 구간을 따라 반복
    # Runge-Kutta 4 차수 적분
    k1 = f(X[:, k], U[:, k])
    k2 = f(X[:, k] + dt / 2 * k1, U[:, k])
    k3 = f(X[:, k] + dt / 2 * k2, U[:, k])
    k4 = f(X[:, k] + dt * k3, U[:, k])
    x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X[:, k+1] == x_next)  # 연속성 제약 조건

# ---- 경로 제약 조건 -----------
track_width = 1.0  # 트랙 폭 (단순화된 모델)
opti.subject_to(p_y <= track_width / 2)  # 트랙 상단 경계
opti.subject_to(p_y >= -track_width / 2)  # 트랙 하단 경계
opti.subject_to(opti.bounded(0, U[0, :], 1))  # 스로틀 제어 변수 제한
opti.subject_to(opti.bounded(-1, U[1, :], 1))  # 스티어링 제어 변수 제한

# ---- 경계 조건 --------
opti.subject_to(p_x[0] == 0)   # 시작 x 위치 0
opti.subject_to(p_y[0] == 0)   # 시작 y 위치 0
opti.subject_to(yaw[0] == 0)   # 시작 yaw 0
opti.subject_to(v[0] == 0)     # 정지 상태에서 시작
opti.subject_to(p_x[-1] == 3)  # 목표 x 위치 3
opti.subject_to(p_y[-1] == 0)  # 목표 y 위치 0

# ---- 기타 제약 조건  ----------
opti.subject_to(T >= 0)  # 시간은 양수여야 함

# ---- 초기 값 설정 ----
opti.set_initial(v, 1)
opti.set_initial(T, 1)

# ---- NLP 문제 풀기 ----
opti.solver("ipopt")  # 수치적 솔버 설정
sol = opti.solve()    # 실제 문제 풀기

# ---- 후처리 ----
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(sol.value(p_x), sol.value(p_y), label="trajectory")  # 위치 플롯
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(sol.value(v), label="speed")  # 속도 플롯
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
