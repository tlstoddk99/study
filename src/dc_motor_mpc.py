import numpy as np
import matplotlib.pyplot as plt

# 시스템 매개변수 정의
A = np.array([[0.9048, 0], [0.0952, 1]])
B = np.array([[0.0952], [0.0048]])
C = np.array([[0, 1]])

# 예측 지평과 제어 지평
Np = 10
Nc = 1

# 제어 가중치 행렬
R = 0.1 * np.eye(Nc)

# 참조 신호
r = 2 * np.ones((Np, 1))


# 초기 상태
x0 = np.array([[0], [0]])
u0 = 0


# 예측 모델 생성
def get_prediction_matrices(A, B, C, Np, Nc):
    F = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(1, Np + 1)])
    Phi = np.zeros((Np, Nc))
    for i in range(Np):
        for j in range(Nc):
            if i >= j:
                Phi[i, j] = (C @ np.linalg.matrix_power(A, i - j) @ B).item()
    return F, Phi


# 시뮬레이션 수행
x = x0
temp_x = x0
u_k = u0
x_Aug = np.vstack([x0, C @ x0])
A_Aug = np.block([[A, np.zeros((2, 1))], [C @ A, np.eye(1)]])
B_Aug = np.vstack([B, C @ B])
C_Aug = np.hstack([C, np.ones((1, 1))])
x_list = []
y_list = []
u_list = []
delta_U=np.array([[0]])

for k in range(100):
    
    temp_x = x
    u_k = u_k + delta_U[0]
    x = A @ x + B * u_k
    delta_x = x - temp_x
    x_Aug = np.vstack([delta_x, C @ x])

    # 예측 행렬 계산
    F, Phi = get_prediction_matrices(A_Aug, B_Aug, C_Aug, Np, Nc)
    
    
    # 제어 입력 계산
    delta_U = np.linalg.inv(Phi.T @ Phi + R) @ Phi.T @ (r - F @ x_Aug)

    # 기록 저장
    x_list.append(x_Aug)
    y_list.append(C_Aug @ x_Aug)
    u_list.append(u_k)

print("x_list: ", x_list)
print("y: ", y_list)
print("u_seq: ", u_list)

# 결과 플로팅
x_val_1 = [x[0].item() for x in x_list]
x_val_2 = [x[1].item() for x in x_list]
u_val = [u.item() for u in u_list]
y_val = [y.item() for y in y_list]


plt.plot(x_val_1, label="x1")
plt.legend()
# plt.show()
plt.plot(x_val_2, label="x2")
plt.legend()
plt.show()
plt.plot(u_val, label="u")
plt.legend()
plt.show()
plt.plot(y_val, label="y")
plt.legend()
plt.show()