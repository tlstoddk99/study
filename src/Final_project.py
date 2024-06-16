import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import casadi as ca


def get_dx(x, u,dt):
    # Parameters (constants)
    m = 0.35  # mass of the car
    Iz = 0.47  # yaw moment of inertia
    lf = 0.15  # distance to front axle
    lr = 0.17  # distance to rear axle
    Bf, Cf, Df = 1.5, 1.5, 35.0  # Pacejka tire model parameters
    Br, Cr, Dr = 1.5, 1.5, 35.0  # Pacejka tire model parameters
    Cm1, Cm2, Cd, Croll = 1.0, 0.05, 0.3, 0.05  # Longitudinal force parameters

    # State variables
    px, py, yaw, vx, vy, omega, tau, delta, theta = x

    # Control inputs (derivatives of tau, delta, and theta)
    dtau, ddelta, dtheta = u

    # Tire slip angles
    alpha_f = -np.arctan((vy + lf * omega) / vx) + delta
    alpha_r = -np.arctan((vy - lr * omega) / vx)

    # Tire forces
    Ff = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
    Fr = Dr * np.sin(Cr * np.arctan(Br * alpha_r))

    # Longitudinal force
    Fx = (Cm1 - Cm2 * vx) * tau - Cd * vx**2 - Croll

    # System dynamics (derivatives)
    f_ = np.array(
        [
            vx * np.cos(yaw) - vy * np.sin(yaw),  # px_dot
            vx * np.sin(yaw) + vy * np.cos(yaw),  # py_dot
            omega,  # yaw_dot
            (((Fx - Ff * np.sin(delta)) / m + vy * omega))*dt,  # vx_dot
            ((Fr + Ff * np.cos(delta)) / m - vx * omega)*dt,  # vy_dot
            ((Ff * lf * np.cos(delta) - Fr * lr) / Iz)*dt,  # omega_dot
            dtau,  # tau_dot
            ddelta,  # delta_dot
            dtheta,  # theta_dot
        ]
    )

    return f_


def read_center_coordinates():
    file_path = "/home/a/mpc_homework/src/center_path.csv"
    df = pd.read_csv(file_path, sep=",", header=None)
    x_c = df[0].values
    y_c = df[1].values
    center_path = np.array([x_c, y_c])
    return center_path


def get_arc_length(x, y, track_coords):

    track_x = track_coords[0]
    track_y = track_coords[1]

    # Calculate distances between consecutive track points
    distances = np.sqrt(np.diff(track_x) ** 2 + np.diff(track_y) ** 2)

    # Calculate cumulative distance (arc length)
    arc_lengths = np.concatenate(([0], np.cumsum(distances)))

    # Calculate distances from the given point to each track point
    distances_to_point = np.sqrt((track_x - x) ** 2 + (track_y - y) ** 2)

    # Find the index of the closest track point
    closest_point_index = np.argmin(distances_to_point)

    # Return the arc length corresponding to the closest track point
    return arc_lengths[closest_point_index]


def get_track_error(x, center_path, theta_hat):
    # x에서 변수 추출
    px, py, yaw, vx, vy, omega, tau, delta, theta = x

    # 경로 중심선들의 좌표
    track_x = center_path[0]
    track_y = center_path[1]

    # 경로 중심선의 도함수 (유한 차분이나 스플라인 보간법으로 계산된다고 가정)
    dpx_dtheta = np.gradient(track_x)
    dpy_dtheta = np.gradient(track_y)

    # Find the index of the closest point on the centerline
    distances = np.sqrt((track_x - px) ** 2 + (track_y - py) ** 2)
    theta_index = np.argmin(distances)

    # 중심선에서 가장 가까운 점의 좌표
    p_x_hat = track_x[theta_index]
    p_y_hat = track_y[theta_index]
    dpx_dtheta_hat = dpx_dtheta[theta_index]
    dpy_dtheta_hat = dpy_dtheta[theta_index]

    # 컨투어링 에러(eC)와 랙 에러(eL) 계산
    e_C = dpy_dtheta_hat * (px - p_x_hat) - dpx_dtheta_hat * (py - p_y_hat)
    e_L = dpx_dtheta_hat * (px - p_x_hat) + dpy_dtheta_hat * (py - p_y_hat)

    return e_C, e_L


def get_track_constraints(x, center_path):
    px, py, yaw, vx, vy, omega, tau, delta, theta = x

    # Track center line derivatives with respect to theta
    track_x = center_path[0]
    track_y = center_path[1]

    track_width = 0.7  # Track width
    # slack = 0.1  # Slack variable

    index = np.argmin((track_x - px) ** 2 + (track_y - py) ** 2)

    px_hat = track_x[index]
    py_hat = track_y[index]

    con = (px - px_hat) ** 2 + (py - py_hat) ** 2 - (track_width / 2) ** 2

    return con


def get_L(x, u, center_path, v_d=5.0):
    px, py, yaw, vx, vy, omega, tau, delta, theta = x
    dtau, ddelta, dtheta = u

    e_c, e_l = get_track_error(x, center_path, theta)

    q_c = 1.0
    q_l = 1.0
    q_dtau = 1.0
    q_ddelta = 1.0
    q_dtheta = 1.0

    L = (
        q_c * e_c**2
        + q_l * e_l**2
        + q_dtau * dtau**2
        + q_ddelta * ddelta**2
        + q_dtheta * (dtheta - v_d) ** 2
    )

    return L


def get_u(x,center_path, near_index):
    
    px, py, yaw, vx, vy, omega, tau, delta, theta = x
    
    near_x, near_y, near_index = get_near_center_and_index(x, center_path)
    
    Ld = 0.5  # Look-ahead distance
    Lf = 0.3  # Wheelbase
    

    # Transform the lookahead point to the vehicle's coordinate frame
    dx = near_x - px
    dy = near_y - py

    # Compute the angle to the lookahead point
    alpha = np.arctan2(dy, dx) - yaw

    # Compute the steering angle delta using the Pure Pursuit control law
    delta = np.arctan2(2 * Ld * np.sin(alpha), Ld)

    # Compute the rate of change of the steering angle
    d_delta = (delta - x[7]) / 0.1  # Assuming a time step of 0.1 seconds
    
    u = np.array([0.0, d_delta, 0.0])
        
        
    
    
    return u


def get_near_center_and_index(x, center_path):
    px, py, yaw, vx, vy, omega, tau, delta, theta = x

    track_x = center_path[0]
    track_y = center_path[1]

    distances = np.sqrt((track_x - px) ** 2 + (track_y - py) ** 2)
    index = np.argmin(distances)

    return track_x[index], track_y[index], index


if __name__ == "__main__":
    # Example usage
    init_x = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1])  # Initial state
    init_u = np.array([0, 0.0, 0.0])  # Initial control input
    t_final = 30.0  # Final time
    delta_t = 0.005  # Time step
    iters = int(t_final / delta_t)  # Number of iterations

    time_history = np.arange(0.0, t_final, delta_t)  # Time history
    center_path = read_center_coordinates()
    x = init_x
    u = init_u

    x_history = []
    u_history = []

    for i in range(iters):
        prev_x = x
        x = prev_x + get_dx(x,u,delta_t) * delta_t

        # nomalize yaw
        x[2] = (x[2] + np.pi) % (2 * np.pi) - np.pi
        if abs(x[3])>10:
            x[3]=0.1
        x[8] = get_arc_length(x[0], x[1], center_path)

        dtheta = get_arc_length(x[0], x[1], center_path) - get_arc_length(
            prev_x[0], prev_x[1], center_path
        )

        u=get_u(x,center_path,0)
        
        # u_sol = get_u(x, center_path, horizon=10, dt=delta_t)
        # u = u_sol[:, 0]
        # u = np.array([5 * np.sin(i * delta_t), -np.sin(i * delta_t * 4), dtheta])

        x_history.append(x)
        u_history.append(u)

    x_history = np.array(x_history)
    u_history = np.array(u_history)

    plt.figure()
    plt.plot(x_history[:, 0], x_history[:, 1], label="history")
    plt.plot(center_path[0], center_path[1], label="center path")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.legend()
    plt.title("Trajectory in XY Plane")

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time_history, x_history[:, 2], label="yaw")
    plt.plot(time_history, x_history[:, 3], label="vx")
    plt.plot(time_history, x_history[:, 4], label="vy")
    plt.plot(time_history, x_history[:, 5], label="omega")
    plt.legend()
    plt.ylabel("States")
    plt.title("Yaw, vx, vy, and omega over Time")

    plt.subplot(3, 1, 2)
    plt.plot(time_history, x_history[:, 6], label="tau")
    plt.plot(time_history, x_history[:, 7], label="delta")
    plt.plot(time_history, x_history[:, 8], label="theta")
    plt.legend()
    plt.ylabel("Control Inputs")
    plt.title("Tau, Delta, and Theta over Time")

    plt.subplot(3, 1, 3)
    plt.plot(time_history, u_history[:, 0], label="dtau")
    plt.plot(time_history, u_history[:, 1], label="ddelta")
    plt.plot(time_history, u_history[:, 2], label="dtheta")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Control Inputs Derivatives")
    plt.title("Derivatives of Control Inputs over Time")
    plt.tight_layout()
    plt.show()
