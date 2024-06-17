import casadi as ca
import pandas as pd
import matplotlib.pyplot as plt

N = 200  # 제어 구간의 수

# 데이터 로드
center_w = "/home/a/mpc_homework/map/centerline_waypoints.csv"
df_1 = pd.read_csv(center_w, sep=",", header=None)
x_c = df_1[0].values
y_c = df_1[1].values
x_c = ca.DM(x_c)
y_c = ca.DM(y_c)

left_w = "/home/a/mpc_homework/map/left_waypoints.csv"
df_2 = pd.read_csv(left_w, sep=",", header=None)
x_l = df_2[0].values
y_l = df_2[1].values
x_l = ca.DM(x_l)
y_l = ca.DM(y_l)

right_w = "/home/a/mpc_homework/map/right_waypoints.csv"
df_3 = pd.read_csv(right_w, sep=",", header=None)
x_r = df_3[0].values
y_r = df_3[1].values
x_r = ca.DM(x_r)
y_r = ca.DM(y_r)

spline = "/home/a/mpc_homework/map/center_spline_derivatives.csv"
df_4 = pd.read_csv(spline, sep=",", header=None)
s_x = df_4[0].values
s_y = df_4[1].values
s_x = ca.DM(s_x)
s_y = ca.DM(s_y)

width = "/home/a/mpc_homework/map/track_widths.csv"
df_5 = pd.read_csv(width, sep=",", header=None)
w_l = df_5[0].values
w_r = df_5[1].values
w_l = ca.DM(w_l)
w_r = ca.DM(w_r)

opti = ca.Opti()

# 차량 매개변수
m, Iz, lf, lr = 0.35, 0.047, 0.15, 0.17
Bf, Cf, Df, Br, Cr, Dr = 1.5, 1.5, 35.0, 1.5, 1.5, 35.0
Cm1, Cm2, Cd, Croll = 1.0, 0.05, 0.3, 0.05
params = (m, Iz, lf, lr, Bf, Cf, Df, Br, Cr, Dr, Cm1, Cm2, Cd, Croll)

# 초기 상태 및 제어 입력
x0 = ca.MX([x_c[0], y_c[0], 0, 0.1, 0, 0, 0])
x_ref = ca.MX([x_c[300], y_c[300], 0, 0, 0, 0, 0])
u_ref = ca.MX([0, 0, 0])
x_min = ca.MX([-ca.inf] * 7)
x_max = ca.MX([ca.inf] * 7)
u_min = ca.MX([-3, -0.4, -ca.inf])
u_max = ca.MX([5, 0.4, ca.inf])

# 가중치 행렬
Q = ca.diag([1, 1, 1, 1, 1, 1, 1])
R = ca.diag([1, 1, 1])

# 결정 변수
X = opti.variable(7, N + 1)
U = opti.variable(3, N)

# 선형화 함수 정의
def linearize_vehicle_dynamics(x, u, params):
    m, Iz, lf, lr, Bf, Cf, Df, Br, Cr, Dr, Cm1, Cm2, Cd, Croll = params
    
    alpha_f = -ca.atan2((lf * x[5] + x[4]), x[3]) + u[1]
    alpha_r = ca.atan2((lr * x[5] - x[4]), x[3])
    
    Ff_y = Df * ca.sin(Cf * ca.arctan(Bf * alpha_f))
    Fr_y = Dr * ca.sin(Cr * ca.arctan(Br * alpha_r))
    
    Fr_x = (Cm1 - Cm2 * x[3]) * u[0] - Cr - Cd * x[3]**2
    
    dxdt = ca.vertcat(
        x[3] * ca.cos(x[2]) - x[4] * ca.sin(x[2]),
        x[3] * ca.sin(x[2]) + x[4] * ca.cos(x[2]),
        x[5],
        (Fr_x - Ff_y * ca.sin(u[1]) + m * x[4] * x[5]) / m,
        (Fr_y + Ff_y * ca.cos(u[1])) / m,
        (Ff_y * lf * ca.cos(u[1]) - Fr_y * lr) / Iz,
        u[2]
    )
    
    A = ca.jacobian(dxdt, x)
    B = ca.jacobian(dxdt, u)
    
    return ca.Function('A', [x, u], [A]), ca.Function('B', [x, u], [B])

# 선형화
A_func, B_func = linearize_vehicle_dynamics(x0, u_ref, params)
A = A_func(x0, u_ref)
B = B_func(x0, u_ref)

# QP 목적 함수 정의
def define_qp_objective(opti, X, U, x_ref, u_ref, Q, R):
    obj = 0
    for k in range(N):
        obj += ca.mtimes([(X[:, k] - x_ref).T, Q, (X[:, k] - x_ref)]) \
               + ca.mtimes([(U[:, k] - u_ref).T, R, (U[:, k] - u_ref)])
    opti.minimize(obj)

# 목적 함수 설정
define_qp_objective(opti, X, U, x_ref, u_ref, Q, R)

# QP 제약 조건 정의
def define_qp_constraints(opti, X, U, A, B, x0, x_min, x_max, u_min, u_max):
    # 초기 상태 제약 조건
    opti.subject_to(X[:, 0] == x0)
    
    # 동적 제약 조건
    for k in range(N):
        x_next = ca.mtimes(A, X[:, k]) + ca.mtimes(B, U[:, k])
        opti.subject_to(X[:, k + 1] == x_next)
    
    # 상태 및 제어 입력의 경계 제약 조건
    for k in range(N + 1):
        opti.subject_to(opti.bounded(x_min, X[:, k], x_max))
    for k in range(N):
        opti.subject_to(opti.bounded(u_min, U[:, k], u_max))

# 제약 조건 설정
define_qp_constraints(opti, X, U, A, B, x0, x_min, x_max, u_min, u_max)

# 솔버 설정 및 문제 해결
opti.solver('qpoases')
sol = opti.solve()

# 결과 시각화
plt.figure()
plt.plot(sol.value(X[0, :]), sol.value(X[1, :]), "b", label="trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.plot(x_c, y_c, "g--", label="centerline")
plt.plot(x_l, y_l, "r", label="left boundary")
plt.plot(x_r, y_r, "r", label="right boundary")
plt.show()

plt.subplot(2, 1, 1)
plt.plot(sol.value(X[2, :]), label="yaw")
plt.plot(sol.value(X[3, :]), label="v_x")
plt.plot(sol.value(X[4, :]), label="v_y")
plt.xlabel("Time step")
plt.ylabel("State")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sol.value(U[0, :]), "k", label="throttle")
plt.plot(sol.value(U[1, :]), "r", label="steering")
plt.xlabel("Time step")
plt.ylabel("Control input")
plt.legend()
plt.show()
