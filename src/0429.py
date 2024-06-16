import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# 차량 파라미터 정의
m = 1500  # 차량 질량 (kg)
lf = 1.2  # 전륜까지의 거리 (m)
lr = 1.2  # 후륜까지의 거리 (m)
Iz = 2250  # 관성 모멘트 (kg*m^2)
Csf = 80000  # 전륜 코너링 강성 (N/rad)
Csr = 80000  # 후륜 코너링 강성 (N/rad)
a = 1.2  # 전륜과 무게중심의 거리 (m)
b = 1.2  # 후륜과 무게중심의 거리 (m)
h = 0.5  # 무게중심의 높이 (m)
g = 9.81  # 중력 가속도 (m/s^2)
v = 15  # 차량 속도 (m/s)
mu = 0.7  # 마찰 계수

# 시뮬레이션 설정
T = 10.0  # 최종 시간 (s)
N = 100  # 시간 단계 수
dt = T/N  # 시간 간격 (s)

# 최적화 문제 설정
opti = ca.Opti()

# 변수 초기화
x = opti.variable(N+1)
y = opti.variable(N+1)
psi = opti.variable(N+1)
omega = opti.variable(N+1)
beta = opti.variable(N+1)
delta = opti.variable(N+1) # 여기서 delta의 차원이 N+1로 바뀌었습니다.

# 초기 조건
opti.subject_to(x[0] == 0)
opti.subject_to(y[0] == 0)
opti.subject_to(psi[0] == 0)
opti.subject_to(omega[0] == 0)
opti.subject_to(beta[0] == 0)
# delta[0]의 초기 조건은 제약 조건으로 설정되지 않습니다.

# 동적 제약 조건과 목적 함수 정의
for k in range(N):
    # 슬립 각도 계산
    alpha_f = delta[k] - beta[k] - lf * omega[k] / v
    alpha_r = -beta[k] + lr * omega[k] / v

    # 측면 힘 계산
    Fyf = 2 * mu * Csf * (m * g * lr - m * a * h * ca.power(omega[k], 2)) * alpha_f / (lf + lr)
    Fyr = 2 * mu * Csr * (m * g * lf + m * a * h * ca.power(omega[k], 2)) * alpha_r / (lf + lr)

    # 상태 업데이트
    opti.subject_to(x[k+1] == x[k] + dt * (v * ca.cos(psi[k] + beta[k])))
    opti.subject_to(y[k+1] == y[k] + dt * (v * ca.sin(psi[k] + beta[k])))
    opti.subject_to(psi[k+1] == psi[k] + dt * omega[k])
    opti.subject_to(omega[k+1] == omega[k] + dt * (lf * Fyf - lr * Fyr) / Iz)
    opti.subject_to(beta[k+1] == beta[k] + dt * ((Fyf + Fyr) / (m * v) - omega[k]))

# 목적 함수 설정
A = 10.0  # SIN 함수의 진폭
B = 1 * np.pi / T  # SIN 함수의 주기
J = ca.sum1((y - A * ca.sin(B * x))**2) * dt
opti.minimize(J)

# 조향 각의 제한 설정
delta_max = np.deg2rad(25)  # 최대 조향 각도 (라디안)
opti.subject_to(opti.bounded(-delta_max, delta, delta_max))

# 최적화 솔버 설정
opti.solver('ipopt')

# 문제 풀이
sol = opti.solve()

# 결과 추출
x_opt = sol.value(x)
y_opt = sol.value(y)
psi_opt = sol.value(psi)
beta_opt = sol.value(beta)
delta_opt = sol.value(delta)

# 결과 플롯
plt.figure(figsize=(10, 5))
plt.plot(x_opt, y_opt, 'r-', label='Optimal Vehicle Path')
plt.plot(x_opt, A * np.sin(B * x_opt), 'b--', label='Desired SIN Path')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vehicle Path Optimization with Nonlinear Model')
plt.show()
