import numpy as np
import matplotlib.pyplot as plt
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


# 초기 상태
A = np.array([[0.9048, 0], [0.0952, 1]])
B = np.array([[0.0952], [0.0048]])
C = np.array([[0, 1]])


# 예측 지평과 제어 지평
Np = 100
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
    w = np.array([[0.01*(np.random.normal(0, 1))], [0.01*(np.random.normal(0, 1))]])
    temp_real_x = real_x
    temp_x_hat = x_hat
    temp_x_hat.reshape(-1, 1)
    u_k = u_k + delta_U[0]

    real_x = A @ real_x + B * u_k + w

    delta_x = real_x - temp_real_x
    x_Aug = np.vstack([delta_x, C @ real_x])

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
