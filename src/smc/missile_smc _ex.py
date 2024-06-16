import numpy as np
import matplotlib.pyplot as plt

lambda_ = 20
eta = 0.1

# Define the constants
Iy = 150  # kg*m^2
Iz = 150  # kg*m^2
m = 150  # kg
lT = 1.1  # m
tau_d = 0.005
tau_f = 0.001

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


def psi1(q, r, alpha, beta, V, delta_e, T_z, a_zc):
    term1 = (
        q
        - r * np.sin(alpha) * np.tan(beta)
        + (c4 * alpha + c5 * delta_e + c7 * T_z) * np.cos(alpha) / np.cos(beta)
    )
    return c4 * V * term1 - c5 * V / tau_a * delta_e - c7 * V / tau_f * T_z - a_zc


def psi2(q, r, alpha, beta, delta_e, T_z, q_c_dot):
    term1 = c1 * q + c2 * alpha + c3 * delta_e - c6 * T_z
    term2 = (
        q
        - r * np.sin(alpha) * np.tan(beta)
        + (c4 * alpha + c5 * delta_e + c7 * T_z) * np.cos(alpha) / np.cos(beta)
    )
    return c1 * term1 + c2 * term2 - c3 / tau_a * delta_e + c6 / tau_f * T_z - q_c_dot


def psi3(r, alpha, beta, V, delta_r, T_y, a_yc):
    term1 = (
        -r * np.cos(alpha)
        + (c4 * beta + c5 * delta_r + c7 * T_y) * np.cos(beta)
        + (c4 * alpha + c5 * delta_e + c7 * T_z) * np.sin(alpha) * np.sin(beta)
    )
    return c4 * V * term1 - c5 * V / tau_a * delta_r - c7 * V / tau_f * T_y - a_yc


def psi4(r, alpha, beta, delta_r, T_y, r_c_dot):
    term1 = c1_prime * r + c2_prime * beta + c3_prime * delta_r + c6_prime * T_y
    term2 = (
        -r * np.cos(alpha)
        + (c4_prime * beta + c5_prime * delta_r + c7 * T_y) * np.cos(beta)
        + (c4 * alpha + c5 * delta_e + c7 * T_z) * np.sin(alpha) * np.sin(beta)
    )
    return (
        c1_prime * term1
        + c2_prime * term2
        - c3_prime / tau_a * delta_r
        + c6_prime / tau_f * T_y
        - r_c_dot
    )


def f_matrix(V):
    return np.array([[]])


def gamma_matrix(V):
    return np.array(
        [
            [c5 * V / tau_d, c7 * V / tau_f, 0, 0],
            [c3 / tau_d, -c6 / tau_f, 0, 0],
            [0, 0, c5_prime * V / tau_d, c7 * V / tau_f],
            [0, 0, c3_prime / tau_d, c6_prime / tau_f],
        ]
    )


x = np.array([[0], [0]])
delt = 1 / 100
t = np.arange(0, 10 + delt, delt)
Hz = 2
amp = 2


xd = amp * np.sin(t / Hz)
xd_dot = amp * np.cos(t / Hz) / Hz
xd_ddot = -amp * np.sin(t / Hz) / (Hz * Hz)

s = np.zeros(len(t))
sat_ = np.zeros(len(t))
u = np.zeros(len(t))

for i in range(len(t)):
    x_tilde = x[0, i] - xd[i]
    x_tildedot = x[1, i] - xd_dot[i]
    s[i] = x_tildedot + lambda_ * x_tilde
    F = 0.5 * x[1, i] ** 2 * abs(np.cos(3 * x[0, i]))

    if s[i] > 1:
        sat_[i] = 1
    elif s[i] < -1:
        sat_[i] = -1
    else:
        sat_[i] = s[i] / 0.1

    f_hat = -(1.5 * x[1, i] ** 2 * np.cos(3 * x[0, i]))
    u[i] = -f_hat + xd_ddot[i] - 20 * x_tildedot - (F + eta) * np.sign(s[i])

    at = abs(np.sin(t[i])) + 1
    f = np.array([[x[1, i]], [-at * x[1, i] ** 2 * np.cos(3 * x[0, i]) + u[i]]])
    x = np.hstack((x, x[:, i : i + 1] + delt * f))

plt.figure(1)
plt.plot(t, x[0, :-1] - xd, label="Original")
plt.title("Tracking Error")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.legend()

plt.figure(2)
plt.plot(t, u, label="Original")
plt.title("Control Input")
plt.xlabel("Time (s)")
plt.ylabel("u")
plt.legend()
plt.show()
