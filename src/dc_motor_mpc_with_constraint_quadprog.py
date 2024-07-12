import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# 예측 모델 생성
def get_prediction_matrices(A, B, C, Np, Nc):
    F = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(1, Np + 1)])
    Phi = np.zeros((Np, Nc * B.shape[1]))
    for i in range(Np):
        for j in range(Nc):
            if i >= j:
                Phi[i, j * B.shape[1] : (j + 1) * B.shape[1]] = (
                    C @ np.linalg.matrix_power(A, i - j) @ B
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
    # 시간에 따라 변하는 시스템 행렬 정의
    A = np.array([[0.9048 + 0.001*k, 0], [0.0952 - 0.001*k, 1]])
    B = np.array([[0.0952], [0.0048 + 0.01*k]])
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
# referenceSignal = 2 * np.ones((Np, 1))
Time = np.linspace(0, 2 * np.pi, Np)
referenceSignal= (0.1*np.sin(10*Time)+1.5).reshape(-1, 1)


x_list = []
y_list = []
u_list = []
delta_u_list = []
delta_U = np.array([[0]])

for k in range(100):
    A, B, C = get_time_varying_matrices(k)
    A_Aug = np.block([[A, np.zeros((2, 1))], [C @ A, np.eye(1)]])
    B_Aug = np.vstack([B, C @ B])
    C_Aug = np.hstack([C, np.ones((1, 1))])
    
    w = np.array([[0.02*(np.random.normal(0, 1))], [0.04*(np.random.normal(0, 1))]])
    temp_real_x = real_x
    temp_x_hat = x_hat
    temp_x_hat.reshape(-1, 1)
    u_k = u_k + delta_U[0]

    real_x = A @ real_x + B * u_k + w

    delta_x = real_x - temp_real_x
    x_Aug = np.vstack([delta_x, C @ real_x])

    F, Phi = get_prediction_matrices(A_Aug, B_Aug, C_Aug, Np, Nc)
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

# plt.plot(x_val_1, label="x1")
plt.plot(x_val_2, label="x2")
plt.plot(y_val, label="y")
plt.plot(referenceSignal, label="reference")
plt.legend()
plt.show()

plt.plot(u_val, "r",label="u")
plt.plot(U_max*np.ones(100),"g--",label="u_max")
plt.plot(U_min*np.ones(100),"b--",label="u_min")
plt.legend()
plt.show()

plt.plot(delta_u_val,"r" ,label="delta_u")
plt.plot(delta_U_max*np.ones(100),"g--",label="delta_u_max")
plt.plot(delta_U_min*np.ones(100),"b--",label="delta_u_min")
plt.legend()
plt.show()
