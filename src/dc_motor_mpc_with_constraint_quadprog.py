import numpy as np
import matplotlib.pyplot as plt
import quadprog
from cvxopt import matrix, solvers


# 예측 모델 생성
def get_prediction_matrices(A, B, C, Np, Nc):
    # 예측 행렬 F 계산
    F = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(1, Np + 1)])

    # 제어 행렬 Phi 계산
    Phi = np.zeros((Np, Nc * B.shape[1]))  # 각 열이 입력 벡터 B의 개별 열에 해당
    for i in range(Np):
        for j in range(Nc):
            if i >= j:
                Phi[i, j * B.shape[1] : (j + 1) * B.shape[1]] = (
                    C @ np.linalg.matrix_power(A, i - j) @ B
                ).reshape(1, -1)

    return F, Phi


def solve_qp(E, F, M, gamma):
    print("E: ", E)
    print("F: ", F)
    print("M: ", M)
    print("gamma: ", gamma)

    # Convert numpy arrays to cvxopt matrices
    P = matrix(E)
    q = matrix(F)
    G = matrix(M)
    h = matrix(gamma)

    # Solve the QP problem
    sol = solvers.qp(P, q, G, h)

    # Extract the solution
    x = np.array(sol["x"]).flatten()

    return x


def init_kalman_filter_matrix():
    k_G = np.array([[1, 0], [0, 1]])
    k_H = np.array([[1, 0]])
    k_Q = np.array([[0.5, 0], [0, 0.5]])
    k_R = np.array([[0.1]])
    k_P = np.array([[0, 0], [0, 0]])
    k_K = np.array([[0], [0]])
    
    k_G = k_G.reshape(-1, 2)
    k_H = k_H.reshape(-1, 2)
    k_Q = k_Q.reshape(-1, 2)
    k_R = k_R.reshape(-1, 1)
    k_P = k_P.reshape(-1, 2)
    k_K = k_K.reshape(-1, 2)
    
    return k_G, k_H, k_Q, k_R, k_P, k_K


def init_kalman_filter_vector():
    k_x = np.array([0, 0])
    k_x_hat = np.array([0, 0])
    k_u = np.array([0,0])
    k_z = np.array([0])
    k_v = np.array([0])
    k_w = np.array([0, 0])
    
    k_x = k_x.reshape(-1, 1)
    k_x_hat = k_x_hat.reshape(-1, 1)
    k_u = k_u.reshape(-1, 1)
    k_z = k_z.reshape(-1, 1)
    k_v = k_v.reshape(-1, 1)
    k_w = k_w.reshape(-1, 1)
    
    return k_x, k_x_hat, k_u, k_z, k_v, k_w


def gen_kalman_filter_noise(Q_or_R):
    noise = np.random.normal(0, 1, (2, 1)) * np.sqrt(np.diag(Q_or_R)).reshape(2, 1)
    return noise

def kalman_predict(A, B, G, Q, x_hat, P, u, w, dt):
    w = np.array([[0], [0]]).reshape(-1, 1) 
    x_hat = A @ x_hat + B @ u + G @ w
    P = A @ P @ A.T + G @ Q @ G.T
    x_hat = x_hat.reshape(-1, 1)
    
    return x_hat, P


def kalman_update(H, R, P, z, x_hat, K):
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_hat = x_hat + K @ (z - (H @ x_hat)).reshape(-1, 1)
    
    P = (np.eye(2) - K @ H) @ P
    return x_hat, P, K


def measure_update(z, x, H, v, R):
    v = np.array(np.random.normal(0, 1) ).reshape(-1, 1)
    z = H @ x + v
    return z


# 초기 상태
A = np.array([[0.9048, 0], [0.0952, 1]])
B = np.array([[0.0952], [0.0048]])
C = np.array([[0, 1]])

A = A.reshape(-1, 2)
B = B.reshape(-1, 2)
C = C.reshape(-1, 2)

# 예측 지평과 제어 지평
Np = 30
Nc = 10

# 제어 가중치 행렬
R = 1 * np.eye(Nc)

# 참조 신호
referenceSignal = 2 * np.ones((Np, 1))

x0 = np.array([[0], [0]])
u0 = 0
real_x = x0
x_hat = x0
z = x0
temp_real_x = x0
u_k = u0

G_k, H_k, Q_k, R_k, P_k, K_k = init_kalman_filter_matrix()
k_x, x_hat, u_k, z, k_v, k_w = init_kalman_filter_vector()

x_Aug = np.vstack([x0, C @ x0])
A_Aug = np.block([[A, np.zeros((2, 1))], [C @ A, np.eye(1)]])
B_Aug = np.vstack([B, C @ B])
C_Aug = np.hstack([C, np.ones((1, 1))])


x_list = []
y_list = []
u_list = []
delta_u_list = []
delta_U = np.array([[0]])


for k in range(100):

    temp_real_x = real_x
    temp_x_hat = x_hat
    temp_x_hat.reshape(-1, 1)
    u_k = u_k + delta_U[0]

    z = measure_update(z, real_x, H_k, k_v, R_k)
    x_hat, P_k = kalman_predict(A, B, G_k, Q_k, x_hat, P_k, u_k, k_w, 1)
    x_hat, P_k, K_k = kalman_update(H_k, R_k, P_k, z, x_hat, K_k)

    real_x = A @ real_x + B * u_k

    delta_x = x_hat - temp_x_hat
    x_Aug = np.vstack([delta_x, C @ x_hat])

    # delta_x = real_x - temp_real_x
    # x_Aug = np.vstack([delta_x, C @ real_x])

    # 예측 행렬 계산
    F, Phi = get_prediction_matrices(A_Aug, B_Aug, C_Aug, Np, Nc)

    # QP 문제로 변환 (제약조건 추가)
    E = Phi.T @ Phi + R
    f = -Phi.T @ (referenceSignal - F @ x_Aug)

    # 0<=u<=0.6 , -0.2<=delta_u<=0.2 np.vstack(u, delta_u) 제약조건 설정
    U_min, U_max = -0.3, 0.5
    delta_U_min, delta_U_max = -0.1, 0.2

    # 제약조건
    M = np.vstack((np.ones(Nc), -np.ones(Nc), np.ones(Nc), -np.ones(Nc)))
    M = np.zeros((4, Nc))
    M[0, 0] = 1
    M[1, 0] = -1
    M[2, 0] = 1
    M[3, 0] = -1

    gamma = np.array(
        [[U_max - u_k[0]], [-U_min + u_k[0]], [delta_U_max], [-delta_U_min]]
    )

    delta_U = solve_qp(E, f, M, gamma)

    # 기록 저장
    x_list.append(real_x)
    y_list.append(C_Aug @ x_Aug)
    u_list.append(u_k)
    delta_u_list.append(delta_U[0])


# 결과 플로팅
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
