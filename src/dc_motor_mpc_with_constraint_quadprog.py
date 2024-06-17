import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# 예측 모델 생성
def get_prediction_matrices(A_list, B_list, C_list, Np, Nc):
    F = np.zeros((Np, A_list[0].shape[0]))
    Phi = np.zeros((Np, Nc * B_list[0].shape[1]))

    for i in range(Np):
        C_prod_A = C_list[i]
        for j in range(1, i + 1):
            C_prod_A = C_prod_A @ A_list[i - j]
        F[i, :] = C_prod_A

        for j in range(Nc):
            if i >= j:
                C_prod_A = C_list[i]
                for k in range(j):
                    C_prod_A = C_prod_A @ A_list[i - k - 1]
                Phi[i, j * B_list[0].shape[1] : (j + 1) * B_list[0].shape[1]] = (
                    C_prod_A @ B_list[i - j]
                ).reshape(1, -1)

    return F, Phi

def solve_qp(E, F, M, gamma):
    P = matrix(E)
    q = matrix(F)
    G = matrix(M)
    h = matrix(gamma)
    sol = solvers.qp(P, q, G, h)
    x = np.array(sol["x"]).flatten()
    return x

def get_time_varying_matrices(k):
    A = np.array([[0.9048 + 0.0001*k, 0], [0.0952 - 0.0001*k, 1]])
    B = np.array([[0.0952], [0.0048 + 0.0001*k]])
    C = np.array([[0, 1]])
    return A, B, C

# 초기 상태
x0 = np.array([[0], [0]])
u0 = 0
real_x = x0
x_hat = x0
z = x0
temp_real_x = x0
u_k = u0

Np = 100
Nc = 10
R = 1 * np.eye(Nc)
referenceSignal = 2 * np.ones((Np, 1))

x_list = []
y_list = []
u_list = []
delta_u_list = []
delta_U = np.array([[0]])

for k in range(100):
    A_list = []
    B_list = []
    C_list = []
    for i in range(Np):
        A, B, C = get_time_varying_matrices(k + i)
        A_list.append(A)
        B_list.append(B)
        C_list.append(C)
    
    A_Aug = np.block([[A_list[0], np.zeros((2, 1))], [C_list[0] @ A_list[0], np.eye(1)]])
    B_Aug = np.vstack([B_list[0], C_list[0] @ B_list[0]])
    C_Aug = np.hstack([C_list[0], np.ones((1, 1))])
    
    w = np.array([[0.01*(np.random.normal(0, 1))], [0.01*(np.random.normal(0, 1))]])
    temp_real_x = real_x
    temp_x_hat = x_hat
    temp_x_hat.reshape(-1, 1)
    u_k = u_k + delta_U[0]

    real_x = A_list[0] @ real_x + B_list[0] * u_k + w

    delta_x = real_x - temp_real_x
    x_Aug = np.vstack([delta_x, C_list[0] @ real_x])

    F, Phi = get_prediction_matrices(A_list, B_list, C_list, Np, Nc)
    E = Phi.T @ Phi + R
    f = -Phi.T @ (referenceSignal - F @ x_Aug)

    U_min, U_max = -0.3, 0.5
    delta_U_min, delta_U_max = -0.1, 0.2

    M = np.zeros((4, Nc))
    M[0, 0] = 1
    M[1, 0] = -1
    M[2, 0] = 1
    M[3, 0] = -1

    gamma = np.array([[U_max - u_k[0]], [-U_min + u_k[0]], [delta_U_max], [-delta_U_min]])
    delta_U = solve_qp(E, f, M, gamma)

    x_list.append(real_x)
    y_list.append(C_Aug @ x_Aug)
    u_list.append(u_k)
    delta_u_list.append(delta_U[0])

x_val_1 = [x[0].item() for x in x_list]
x_val_2 = [x[1].item() for x in x_list]
u_val = [u.item() for u in u_list]
y_val = [y.item() for y in y_list]
delta_u_val = [delta_u.item() for delta_u in delta_u_list]

plt.plot(x_val_1, label="x1")
plt.plot(x_val_2, label="x2")
plt.plot(y_val, label="y")
plt.legend()
plt.show()

plt.plot(u_val, label="u")
plt.plot(delta_u_val, label="delta_u")
plt.legend()
plt.show()
