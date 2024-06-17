import numpy as np
import matplotlib.pyplot as plt

# System matrices
A = np.array([[0.9048, 0], [0.0952, 1]])
B = np.array([[0.0952], [0.0048]])
C = np.array([[0, 1]])

# 예측 지평과 제어 지평
Np = 60
Nc = 5

# 제어 가중치 행렬
R = 1 * np.eye(Nc)

# 참조 신호
referenceSignal = 2 * np.ones((Np, 1))


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


def hildreth_qp(E, F, M, gamma, max_iter=1000, tol=1e-6):

    # Compute H and K matrices
    E_inv = np.linalg.pinv(E)
    H = M @ E_inv @ M.T
    K = (gamma + M @ E_inv @ F)

    # Initialize
    H_row=np.shape(H)[0]
    H_col=np.shape(H)[1]
    n=gamma.shape[0]
    
    lamb = np.zeros(n).reshape(-1, 1)
    w = np.zeros(n).reshape(-1, 1)

    # Iterate to update lambda
    for m in range(max_iter):
        lamb_old = lamb.copy()
        
        for i in range(n):
            sum1=np.zeros((1,1))
            sum2=np.zeros((1,1))
            
            for j in range(i):
                sum1+=H[i,j]*lamb[j]
            for j in range(i+1,n):
                sum2+=H[i,j]*lamb_old[j]
            w[i]=(-sum1-sum2-K[i])/H[i,i]
            lamb[i]=max(w[i],0)
            
        # Check for convergence
        if np.linalg.norm(lamb - lamb_old) < tol:
            break

    # Compute the solution x
    x = -E_inv @ (F + M.T @ lamb)

    return x


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
delta_u_list=[]
delta_U = np.array([[0]])


for k in range(100):

    temp_x = x
    u_k = u_k + delta_U[0]
    x = A @ x + B * u_k
    delta_x = x - temp_x
    x_Aug = np.vstack([delta_x, C @ x])

    # 예측 행렬 계산
    F, Phi = get_prediction_matrices(A_Aug, B_Aug, C_Aug, Np, Nc)

    # QP로 풀어낼 문제 (제약조건 없을 때)
    # delta_U = np.linalg.inv(Phi.T @ Phi + R) @ Phi.T @ (r - F @ x_Aug)

    # Hildreth's QP 문제로 변환 (제약조건 추가)
    E = Phi.T @ Phi + R
    f = -Phi.T @ (referenceSignal - F @ x_Aug)

    # 0<=u<=0.6 , -0.2<=delta_u<=0.2 np.vstack(u, delta_u) 제약조건 설정
    U_min, U_max = -0.3, 0.5
    delta_U_min, delta_U_max = -0.1, 0.2

    # 제약조건
    # M = np.vstack((np.ones(Nc), -np.ones(Nc), np.ones(Nc), -np.ones(Nc)))
    M=np.zeros((4,Nc))
    M[0,0]=1
    M[1,0]=-1
    M[2,0]=1
    M[3,0]=-1
    
    gamma = np.array([[U_max-u_k[0]], [-U_min+u_k[0]], [delta_U_max], [-delta_U_min]])
    delta_U = hildreth_qp(E, f, M, gamma)
    # 기록 저장
    x_list.append(x)
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
