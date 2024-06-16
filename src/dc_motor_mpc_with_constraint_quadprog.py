import numpy as np
import matplotlib.pyplot as plt
import quadprog

# 시스템 파라미터
dt = 0.1  # 시간 간격
T = 20  # 예측 지평선
N = int(T / dt)  # 시간 스텝 수
Q = np.diag([1.0, 1.0, 0.1, 0.1])  # 상태 가중치
R = np.diag([0.1, 0.1])  # 제어 입력 가중치

# 차량 모델 (간단한 kinematic model)
def vehicle_model(x, u, dt):
    x_next = np.zeros_like(x)
    x_next[0] = x[0] + dt * x[2] * np.cos(x[3])
    x_next[1] = x[1] + dt * x[2] * np.sin(x[3])
    x_next[2] = x[2] + dt * u[0]
    x_next[3] = x[3] + dt * u[1]
    return x_next

# 목표 궤적 생성
ref_traj = np.zeros((N, 4))
for t in range(N):
    ref_traj[t, 0] = 10.0 * np.sin(0.1 * t)
    ref_traj[t, 1] = 10.0 * np.cos(0.1 * t)
    ref_traj[t, 2] = 1.0
    ref_traj[t, 3] = 0.1 * t

# MPC 비용 함수 행렬 생성
def create_cost_matrices(Q, R, N):
    Q_bar = np.kron(np.eye(N), Q)
    R_bar = np.kron(np.eye(N), R)
    return Q_bar, R_bar

Q_bar, R_bar = create_cost_matrices(Q, R, N)

# 구속 조건 행렬 생성
def create_constraint_matrices(N, x0):
    G = np.zeros((4 * N, 4 * N))
    E = np.zeros((4 * N, 4))
    for i in range(N):
        if i == 0:
            G[4*i:4*i+4, 4*i:4*i+4] = np.eye(4)
            E[4*i:4*i+4, :] = np.eye(4)
        else:
            G[4*i:4*i+4, 4*(i-1):4*(i-1)+4] = -np.eye(4)
            G[4*i:4*i+4, 4*i:4*i+4] = np.eye(4)
    E = E[4:, :]
    return G, E @ x0

G, h = create_constraint_matrices(N, np.zeros(4))

# QP 문제 설정
def solve_mpc(x0, ref_traj, Q_bar, R_bar, G, h):
    H = 2 * np.block([
        [Q_bar, np.zeros((4 * N, 2 * N))],
        [np.zeros((2 * N, 4 * N)), R_bar]
    ])
    f = np.zeros(4 * N + 2 * N)
    for i in range(N):
        f[4*i:4*i+4] = -2 * Q_bar[4*i:4*i+4, 4*i:4*i+4] @ ref_traj[i]
    G_new = np.block([
        [G, np.zeros((G.shape[0], 2 * N))],
        [np.zeros((2 * N, G.shape[1])), np.eye(2 * N)]
    ])
    h_new = np.hstack([h, np.zeros(2 * N)])
    H = np.vstack([np.hstack([H, G_new.T]), np.hstack([G_new, np.zeros((G_new.shape[0], G_new.shape[0]))])])
    f = np.hstack([f, h_new])
    bounds = np.hstack([np.zeros(4 * N + 2 * N), np.zeros(G_new.shape[0])])
    
    solution = quadprog.solve_qp(H, f, np.zeros((0, 0)), np.zeros(0), bounds, bounds)
    return solution[0][:4*N].reshape(N, 4)

# 초기 상태
x0 = np.array([0.0, 0.0, 1.0, 0.0])

# MPC로 궤적 추적
state_traj = [x0]
for t in range(N):
    u_opt = solve_mpc(state_traj[-1], ref_traj[t:], Q_bar, R_bar, G, h)
    x_next = vehicle_model(state_traj[-1], u_opt[0], dt)
    state_traj.append(x_next)

state_traj = np.array(state_traj)

# 결과 시각화
plt.figure()
plt.plot(ref_traj[:, 0], ref_traj[:, 1], 'r--', label='Reference Trajectory')
plt.plot(state_traj[:, 0], state_traj[:, 1], 'b-', label='MPC Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
