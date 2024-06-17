import casadi as ca
import matplotlib.pyplot as plt
import pandas as pd

# ---- 데이터 로드 함수 ----
def load_data():
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

    return N, x_c, y_c, x_l, y_l, x_r, y_r, s_x, s_y, w_l, w_r

# ---- 차량 동역학 함수 ----
def vehicle_dynamics(x, u, params):
    m, Iz, lf, lr, Bf, Cf, Df, Br, Cr, Dr, Cm1, Cm2, Cd, Croll = params

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

# ---- 경로 제약 조건 함수 ----
def closest_idx(x, y, x_c, y_c):
    x_c = ca.DM(x_c)
    y_c = ca.DM(y_c)
    
    distances = ca.sqrt((x - x_c)**2 + (y - y_c)**2)
    
    min_dist = distances[0]
    min_idx = 0
    
    for i in range(1, distances.size()[0]):
        min_dist = ca.if_else(distances[i] < min_dist, distances[i], min_dist)
        min_idx = ca.if_else(distances[i] < min_dist, i, min_idx)
    
    return min_dist, min_idx

# ---- 초기 값 설정 함수 ----
def set_initial_values(opti, X, U, T):
    opti.set_initial(X[0, :], 0)  # p_x
    opti.set_initial(X[1, :], 0)  # p_y
    opti.set_initial(X[2, :], 0)  # yaw
    opti.set_initial(X[3, :], 0.1)  # v_x
    opti.set_initial(X[4, :], 0)  # v_y
    opti.set_initial(X[5, :], 0)  # omega
    opti.set_initial(X[6, :], 0)  # theta
    opti.set_initial(U, 0)
    opti.set_initial(T, 1)

# ---- 후처리 및 결과 시각화 함수 ----
def plot_results(sol, p_x, p_y, U, x_c, y_c, x_l, y_l, x_r, y_r, N):
    plt.figure()
    plt.plot(sol.value(p_x), sol.value(p_y), "b",label="trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.plot(x_c, y_c, "g", label="centerline")
    plt.plot(x_l, y_l, "r", label="left boundary")
    plt.plot(x_r, y_r, "r", label="right boundary")
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(sol.value(U[0, :]), label="throttle")
    plt.plot(sol.value(U[1, :]), label="steering")
    plt.xlabel("Time step")
    plt.ylabel("Control input")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(sol.value(U[0, :]), "k", label="throttle")
    plt.plot(sol.value(U[1, :]), "r", label="steering")
    plt.xlabel("Time step")
    plt.ylabel("Control input")
    plt.legend()
    plt.show()

# ---- 메인 함수 ----
def main():
    N, x_c, y_c, x_l, y_l, x_r, y_r, s_x, s_y, w_l, w_r = load_data()

    opti = ca.Opti()  # 최적화 문제 설정

    # 차량 매개변수
    m, Iz, lf, lr = 0.35, 0.047, 0.15, 0.17
    Bf, Cf, Df, Br, Cr, Dr = 1.5, 1.5, 35.0, 1.5, 1.5, 35.0
    Cm1, Cm2, Cd, Croll = 1.0, 0.05, 0.3, 0.05
    params = (m, Iz, lf, lr, Bf, Cf, Df, Br, Cr, Dr, Cm1, Cm2, Cd, Croll)

    # 결정 변수
    X = opti.variable(7, N + 1)
    p_x, p_y, yaw, v_x, v_y, omega, theta = X[0, :], X[1, :], X[2, :], X[3, :], X[4, :], X[5, :], X[6, :]
    U = opti.variable(3, N)
    T = opti.variable()

    # 목적 함수
    opti.minimize(T)  # 최소 시간을 목표로 경주

    # 동적 제약 조건
    dt = T / N
    for k in range(N):
        k1 = vehicle_dynamics(X[:, k], U[:, k], params)
        k2 = vehicle_dynamics(X[:, k] + dt / 2 * k1, U[:, k], params)
        k3 = vehicle_dynamics(X[:, k] + dt / 2 * k2, U[:, k], params)
        k4 = vehicle_dynamics(X[:, k] + dt * k3, U[:, k], params)
        x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        opti.subject_to(X[:, k + 1] == x_next)

    # 경로 제약 조건
    for k in range(N + 1):
        distance, idx = closest_idx(p_x[k], p_y[k], x_c, y_c)
        opti.subject_to(opti.bounded(0, distance, 0.7))

    # 경계 조건
    opti.subject_to(p_x[0] == 0)
    opti.subject_to(p_y[0] == 0)
    opti.subject_to(yaw[0] == 1)
    opti.subject_to(v_x[0] == 0)
    opti.subject_to(v_y[0] == 0)
    opti.subject_to(omega[0] == 0)
    opti.subject_to(p_x[-1] == x_c[-1])
    opti.subject_to(p_y[-1] == y_c[-1])
    opti.subject_to(opti.bounded(-5, U[0, :], 5))
    opti.subject_to(opti.bounded(-0.4, U[1, :], 0.4))
    opti.subject_to(T >= 0)

    # 초기 값 설정
    set_initial_values(opti, X, U, T)

    # NLP 문제 풀기
    opti.solver("ipopt")

    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"Solver failed: {e}")
        print("Returning initial values for debugging purposes.")
        sol = opti.debug

    # 후처리
    plot_results(sol, p_x, p_y, U, x_c, y_c, x_l, y_l, x_r, y_r, N)

if __name__ == "__main__":
    main()
