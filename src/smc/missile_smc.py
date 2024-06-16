import numpy as np
import matplotlib.pyplot as plt

# Constants from the paper
m = 150  # Mass (kg)
V = 1100  # Velocity (m/s) at Mach 3.7 at sea level (approx.)
I_y = 150  # Moment of inertia about the y-axis (kg m^2)
I_z = 150  # Moment of inertia about the z-axis (kg m^2)
C_alpha = -0.1  # Aerodynamic coefficient for alpha
C_beta = -0.1  # Aerodynamic coefficient for beta
C_m_alpha = -0.1  # Moment coefficient for alpha
C_m_delta_e = -0.1  # Moment coefficient for elevator deflection
C_m_beta = -0.1  # Moment coefficient for beta
C_m_delta_r = -0.1  # Moment coefficient for rudder deflection
L = 1.0  # Reference length (m)
lT = 1.1  # Location of the reaction jet (m)
q_bar = 500  # Dynamic pressure (Pa)
S = 0.5  # Reference area (m^2)
tau_a = 0.1  # Time constant for tail fin
tau_f = 0.1  # Time constant for reaction jet


def update_true_state(state, control_input, dt):
    # State variables of the missile
    # alpha: Angle of Attack
    # beta: Sideslip Angle
    # q: Pitch Rate
    # r: Yaw Rate
    # delta_e: Elevator Deflection Angle
    # delta_r: Rudder Deflection Angle
    # Ty: Lateral Force
    # Tz: Vertical Force
    # a_y: Lateral acceleration
    # a_z: Vertical acceleration
    alpha, beta, q, r, delta_e, delta_r, Ty, Tz, a_y, a_z = state

    # Control input variables
    # delta_ec: Commanded Elevator Deflection
    # delta_rc: Commanded Rudder Deflection
    # Tyc: Commanded Lateral Force
    # Tzc: Commanded Vertical Force
    delta_ec, delta_rc, Tyc, Tzc = control_input

    state = np.array(state).reshape(10, 1)
    control_input = np.array(control_input).reshape(4, 1)

    # Calculate aerodynamic forces and moments
    Y = q_bar * S * (C_beta * beta + C_m_delta_r * delta_r)  # Side force
    Z = q_bar * S * (C_alpha * alpha + C_m_delta_e * delta_e)  # Lift force
    Ma = q_bar * S * L * (C_m_alpha * alpha + C_m_delta_e * delta_e)  # Pitching moment
    Na = q_bar * S * L * (C_m_beta * beta + C_m_delta_r * delta_r)  # Yawing moment

    # Calculate state derivatives
    f = np.array(
        [
            q
            - r * np.sin(alpha) * np.tan(beta)
            + (Z + Tz) * np.cos(alpha) / (m * V * np.cos(beta)),  # Alpha derivative
            -r * np.cos(alpha)
            + (Y + Ty) * np.cos(beta)
            + (Z + Tz) * np.sin(alpha) * np.sin(beta) / (m * V),  # Beta derivative
            (Ma - Tz * lT) / I_y,  # Pitch rate derivative (q_dot)
            (Na + Ty * lT) / I_z,  # Yaw rate derivative (r_dot)
            -delta_e / tau_a,  # Elevator deflection rate
            -delta_r / tau_a,  # Rudder deflection rate
            -Ty / tau_f,  # Control force Ty rate
            -Tz / tau_f,  # Control force Tz rate
            (q_bar * S * C_beta / m)
            * (
                -r * np.cos(alpha)
                + (Y * np.cos(beta) + Z * np.sin(alpha) * np.sin(beta)) / (m * V)
            ),
            (q_bar * S * C_alpha / m)
            * (
                q
                - r * np.sin(alpha) * np.tan(beta)
                + Z * np.cos(alpha) / (m * V * np.cos(beta))
            ),
        ]
    ).reshape(-1, 1)

    # Control input matrix
    g = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1 / tau_a, 0, 0, 0],
            [0, 1 / tau_a, 0, 0],
            [0, 0, 1 / tau_f, 0],
            [0, 0, 0, 1 / tau_f],
            [q_bar * S * C_m_delta_r / (m * tau_a), 0, 1 / (m * tau_f), 0],
            [0, q_bar * S * C_m_delta_e / (m * tau_a), 0, 1 / (m * tau_f)],
        ]
    ).reshape(10, 4)

    # Control input vector
    u = np.array([delta_ec, delta_rc, Tyc, Tzc]).reshape(-1, 1)

    # Compute next state using Euler method
    next_state = state + (f + g @ u) * dt

    return next_state


def plot_results(time_steps, state_history, trajectory_history,u_history):

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    # plt.plot(time_steps, state_history[:, 0], label="alpha: Angle of Attack (rad)")

    # plt.plot(time_steps, state_history[:, 1], label="beta: Sideslip Angle (rad)")

    plt.plot(time_steps, state_history[:, 2], label="q: Pitch Rate (rad/s)")
    plt.plot(
        time_steps,
        trajectory_history[:, 0, 1],
        label="q_des: Desired Pitch Rate (rad/s)",
    )

    # plt.plot(time_steps, state_history[:, 3], label="r: Yaw Rate (rad/s)")
    # plt.plot(
    #     time_steps,
    #     trajectory_history[:, 0],
    #     label="q_des: Desired Angle of Attack (rad)",
    # )
    # plt.plot(
    #     time_steps,
    #     trajectory_history[:, 1],
    #     label="r_des: Desired Sideslip Angle (rad)",
    # )
    plt.xlabel("Time (s)")
    # plt.ylabel("body axis angles (rad)")
    plt.legend()
    # plt.title("State history (body axis angles)")

    plt.subplot(2, 1, 2)
    plt.plot(time_steps, u_history[:, 0], label="delta_e: Elevator Deflection Angle (rad)")
    plt.plot(time_steps, u_history[:, 1], label="delta_r: Rudder Deflection Angle (rad)")
    plt.plot(time_steps, u_history[:, 2], label="Ty: Lateral Force (N)")
    plt.plot(time_steps, u_history[:, 3], label="Tz: Vertical Force (N)")
    plt.xlabel("Time (s)")
    plt.ylabel("Control inputs")
    
    # plt.plot(
    #     time_steps,
    #     state_history[:, 4],
    #     label="delta_e: Elevator Deflection Angle (rad)",
    # )
    # plt.plot(
    #     time_steps, state_history[:, 5], label="delta_r: Rudder Deflection Angle (rad)"
    # )
    # plt.plot(time_steps, state_history[:, 6], label="Ty: Lateral Force (N)")
    # plt.plot(time_steps, state_history[:, 7], label="Tz: Vertical Force (N)")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Input states")
    # plt.legend()
    # plt.title("State history (control inputs)")

    plt.tight_layout()
    plt.show()


def simulate_missile(initial_state, u, dt, t_final):
    time_iter = int(t_final / dt)  # Number of iterations
    time_steps = np.arange(0, t_final, dt)  # Time steps for x-axis

    # Initialize list to record state history
    x_history = []
    x_d_history = []
    u_history = []

    # Set the initial state
    x = initial_state

    # Run simulation
    for i in range(time_iter):
        x = update_true_state(x, u, dt)
        x_d = get_x_d(i * dt, dt, time_iter * dt)
        u = smc_controller(x, x_d, u)

        x_history.append(x)
        x_d_history.append(x_d)
        u_history.append(u)

    # Convert state history to array for plotting
    x_history = np.array(x_history)
    x_d_history = np.array(x_d_history)
    u_history = np.array(u_history)

    return time_steps, x_history, x_d_history, u_history


def get_x_d(t, dt, t_f):
    """
    Generate the trajectory of parabolic missile

    Inputs:
    t: Time step
    dt: Time step size
    t_f: Final time

    Outputs:
    x_d: Desired state of the missile

    a_z_d: Desired vertical acceleration
    q_d: Desired pitch rate
    q_dot_d: Desired pitch rate derivative
    a_y_d: Desired lateral acceleration
    r_d: Desired yaw rate
    r_dot_d: Desired yaw rate derivative

    """
    # Define time-varying attack angle (alpha) and slip angle (beta) as functions of time
    a_z_d = 10 * np.sin(2 * np.pi * t / t_f)
    q_d = -np.cos(2 * np.pi * t / t_f)
    q_dot_d = 2 * np.pi / t_f * np.sin(2 * np.pi * t / t_f)
    a_y_d = 10 * np.sin(2 * np.pi * t / t_f)
    r_d = np.cos(2 * np.pi * t / t_f)
    r_dot_d = -2 * np.pi / t_f * np.sin(2 * np.pi * t / t_f)

    # Calculate the second derivative of the desired trajectory
    a_z_d_dot = 10 * 2 * np.pi / t_f * np.cos(2 * np.pi * t / t_f)
    q_d_dot = -2 * np.pi / t_f * np.sin(2 * np.pi * t / t_f)
    q_ddot_d = -2 * np.pi / t_f * np.cos(2 * np.pi * t / t_f)
    a_y_d_dot = 10 * 2 * np.pi / t_f * np.cos(2 * np.pi * t / t_f)
    r_d_dot = 2 * np.pi / t_f * np.sin(2 * np.pi * t / t_f)
    r_ddot_d = 2 * np.pi / t_f * np.cos(2 * np.pi * t / t_f)

    # Calculate the third derivative of the desired trajectory
    a_z_d_ddot = -10 * 2 * np.pi / t_f * np.sin(2 * np.pi * t / t_f)
    q_d_ddot = -2 * np.pi / t_f * np.cos(2 * np.pi * t / t_f)
    q_dddot_d = 2 * np.pi / t_f * np.sin(2 * np.pi * t / t_f)
    a_y_d_ddot = -10 * 2 * np.pi / t_f * np.sin(2 * np.pi * t / t_f)
    r_d_ddot = -2 * np.pi / t_f * np.cos(2 * np.pi * t / t_f)
    r_dddot_d = -2 * np.pi / t_f * np.sin(2 * np.pi * t / t_f)

    x_d = np.array(
        [
            [a_z_d, q_d, q_dot_d, a_y_d, r_d, r_dot_d],
            [a_z_d_dot, q_d_dot, q_ddot_d, a_y_d_dot, r_d_dot, r_ddot_d],
            [a_z_d_ddot, q_d_ddot, q_dddot_d, a_y_d_ddot, r_d_ddot, r_dddot_d],
        ]
    )

    return x_d


def smc_controller(x, x_d, u):
    """
    Sliding Mode Controller

    Inputs:

    state: Current state of the missile
    0 alpha: Angle of Attack
    1 beta: Sideslip Angle
    2 q: Pitch Rate
    3 r: Yaw Rate
    4 delta_e: Elevator Deflection Angle
    5 delta_r: Rudder Deflection Angle
    6 Ty: Lateral Force
    7 Tz: Vertical Force
    8 a_y: Lateral acceleration
    9 a_z: Vertical acceleration

    x_d: Desired state of the missile
    0 a_z_d: Desired vertical acceleration
    1 q_d: Desired pitch rate
    2 q_dot_d: Desired pitch rate derivative
    3 a_y_d: Desired lateral acceleration
    4 r_d: Desired yaw rate
    5 r_dot_d: Desired yaw rate derivative

    u: Control inputs
    0 delta_ec: Commanded Elevator Deflection
    1 delta_rc: Commanded Rudder Deflection
    2 Tyc: Commanded Lateral Force
    3 Tzc: Commanded Vertical Force

    Outputs:
    control_input: Control inputs for the missile

    """

    # error state of the missile

    e1 = x[9] - x_d[0][0]  # e1=a_z-a_z_d
    e2 = x[2] - x_d[0][1]  # e2=q-q_d
    e3 = x[3] - x_d[0][2]  # e3=q_dot-q_dot_d
    e4 = x[8] - x_d[0][3]  # e4=a_y-a_y_d
    e5 = x[4] - x_d[0][4]  # e5=r-r_d
    e6 = x[5] - x_d[0][5]  # e6=r_dot-r_dot_d

    c1 = -0.1
    c2 = -4.1
    c3 = -24.3
    c4 = -0.11
    c5 = -0.02
    c6 = 0.0073
    c7 = 5.99e-6

    c1_prime = c1
    c2_prime = c2
    c3_prime = -c3
    c4_prime = c4
    c5_prime = c5
    c6_prime = c6

    alpha, beta, q, r, delta_e, delta_r, T_y, T_z, a_y, a_z = x
    a_z_d, q_d, q_dot_d, a_y_d, r_d, r_dot_d = x_d[0]
    a_z_d_dot, q_d_dot, q_ddot_d, a_y_d_dot, r_d_dot, r_ddot_d = x_d[1]

    # Psi1 calculation
    term1_psi1 = (
        q
        - r * np.sin(alpha) * np.tan(beta)
        + (c4 * alpha + c5 * delta_e + c7 * T_z) * np.cos(alpha) / np.cos(beta)
    )
    Psi1 = c4 * V * term1_psi1 - c5 * V / tau_a * delta_e - c7 * V / tau_f * T_z - a_z_d

    # Psi2 calculation
    term1_psi2 = c1 * q + c2 * alpha + c3 * delta_e - c6 * T_z
    term2_psi2 = (
        q
        - r * np.sin(alpha) * np.tan(beta)
        + (c4 * alpha + c5 * delta_e + c7 * T_z) * np.cos(alpha) / np.cos(beta)
    )
    Psi2 = (
        c1 * term1_psi2
        + c2 * term2_psi2
        - c3 / tau_a * delta_e
        + c6 / tau_f * T_z
        - q_dot_d
    )

    # Psi3 calculation
    term1_psi3 = (
        -r * np.cos(alpha)
        + (c4 * beta + c5 * delta_r + c7 * T_y) * np.cos(beta)
        + (c4 * alpha + c5 * delta_e + c7 * T_z) * np.sin(alpha) * np.sin(beta)
    )
    Psi3 = c4 * V * term1_psi3 - c5 * V / tau_a * delta_r - c7 * V / tau_f * T_y - a_y_d

    # Psi4 calculation
    term1_psi4 = c1_prime * r + c2_prime * beta + c3_prime * delta_r + c6_prime * T_y
    term2_psi4 = (
        -r * np.cos(alpha)
        + (c4_prime * beta + c5_prime * delta_r + c7 * T_y) * np.cos(beta)
        + (c4 * alpha + c5 * delta_e + c7 * T_z) * np.sin(alpha) * np.sin(beta)
    )
    Psi4 = (
        c1_prime * term1_psi4
        + c2_prime * term2_psi4
        - c3_prime / tau_a * delta_r
        + c6_prime / tau_f * T_y
        - r_dot_d
    )

    f = np.array([Psi1, e3, Psi2, Psi3, e6, Psi4]).reshape(-1, 1)

    g = np.array(
        [
            [c5 * V / tau_a, c7 * V / tau_f, 0, 0],
            [0, 0, 0, 0],
            [c3 / tau_a, -c6 / tau_f, 0, 0],
            [0, 0, c5_prime * V / tau_a, c7 * V / tau_f],
            [0, 0, 0, 0],
            [0, 0, c3_prime / tau_a, c6_prime / tau_f],
        ]
    ).reshape(6, 4)

    # E=f+g@u

    lambda_ = 20
    eta = 0.1

    S_s = e3 + lambda_ * e2
    gu = -Psi2 - lambda_ * e3 + q_ddot_d - eta * np.sign(S_s)
    u1 = gu / (c3 / tau_a)
    u2 = gu / (-c6 / tau_f)
    u1 = u1.item()
    u2 = u2.item()
    u = np.array([[0], [0], [u1], [u2]]).reshape(4, 1)
    return u


# Example usage
init_state = [0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0]  # Initial state
init_input = [0.0, -0.0, 0.0, 0.0]  # Control inputs (elevator, rudder, Ty, Tz)
init_input = np.array(init_input).reshape(4, 1)
dt = 0.001  # Time step (seconds)
t_final = 1  # Final time (seconds)
time_iter = int(t_final / dt)  # Number of iterations
time_steps = np.arange(0, t_final, dt)  # Time steps for x-axis

# Run simulation
time_steps, x_history, x_d_history,u_history = simulate_missile(
    init_state, init_input, dt, t_final
)

plot_results(time_steps, x_history, x_d_history,u_history)
