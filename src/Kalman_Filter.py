import numpy as np
import matplotlib.pyplot as plt

STATE_D = 2
CONTROL_D = 1
MEASUREMENT_D = 1

def init_matrix():
    A = np.array([[1, 1],
                  [0, 1]])
    B = np.array([[0.5],
                  [1]])
    G = np.array([[1, 0],
                  [0, 1]])
    H = np.array([[1, 0]])
    Q = np.array([[0.5, 0],
                  [0, 0.5]])
    R = np.array([[1]])
    P = np.array([[0, 0],
                  [0, 0]])
    K = np.array([[0],
                  [0]])
    return A, B, G, H, Q, R, P, K

def init_vector(dt):
    x = np.array([10, 30])
    x_hat = np.array([60, 0])
    u = np.array([0])
    z = np.array([0])
    v = np.array([0])
    w = np.array([0, 0])
    return x, x_hat, u, z, v, w

def gen_noise(v_or_w, Q_or_R):
    return np.random.normal(0, 1, size=v_or_w.shape) * np.sqrt(np.diag(Q_or_R))

def predict(A, B, G, Q, x_hat, P, u, w, dt):
    w = gen_noise(w, Q)
    x_hat = A @ x_hat + B @ u + G @ w
    P = A @ P @ A.T + G @ Q @ G.T
    return x_hat, P

def update(H, R, P, z, x_hat, K):
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_hat = x_hat + K @ (z - H @ x_hat)
    P = (np.eye(STATE_D) - K @ H) @ P
    return x_hat, P, K

def true_update(x, dt, A, B, u):
    g = 9.81
    y = x[0]
    v = x[1]
    y_new = y + v * dt - 0.5 * g * dt * dt
    v_new = v - g * dt
    x[0] = y_new
    x[1] = v_new
    return x

def measure_update(z, x, H, v, R):
    v = gen_noise(v, R)
    z = H @ x + v
    return z

# Define the matrices
A, B, G, H, Q, R, P, K = init_matrix()

# Define the vectors
x, x_hat, u, z, v, w = init_vector(0.4)

# Define the time steps
T = 5
dt = 0.2
index = int(T / dt)

# Define the vectors to store the data
x_data = []
x_hat_data = []
z_data = []
t_data = []

# Simulate the system
for t in range(index):
    # Update the true state
    x = true_update(x, dt, A, B, u)
    x_data.append(x[0])

    # Measure the state
    z = measure_update(z, x, H, v, R)
    z_data.append(z[0])

    # Predict the state
    x_hat, P = predict(A, B, G, Q, x_hat, P, u, w, dt)

    # Update the state
    x_hat, P, K = update(H, R, P, z, x_hat, K)

    # Store the data
    x_hat_data.append(x_hat[0])
    t_data.append(t * dt)

    print(f"{t * dt} sec-----------------")
    print(f"x_hat\n{x_hat[0]}\n")
    print(f"P \n{P}\n")
    print(f"K \n{K}\n")
    print("------------------------\n\n")

# Plot the data
plt.figure()
plt.title("Kalman Filter")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.plot(t_data, x_data, label="True Position")
plt.plot(t_data, z_data, label="Measured Position")
plt.plot(t_data, x_hat_data, label="Estimated Position")
plt.legend()
plt.show()
