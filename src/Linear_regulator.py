import numpy as np
import matplotlib.pyplot as plt

# 시스템 파라미터
A = np.array([[0.9974, 0.0539], [-0.1078, 1.1591]])
B = np.array([[0.0013], [0.0539]])
R = R=np.array([[0.05]])
Q = np.array([[0.25, 0.00], [0.00, 0.05]])
H=np.array([[0.0, 0.0], [0.0, 0.0]])
P=H
N = 200

# 변수 초기화
F=[]
U=[]
X=[]

x_k = np.array([[2.0], [1.0]])

#뒤에서 부터 계산
for i in range(N, 0, -1):
    F_k = (-np.linalg.inv(R + B.T @ P @ B)) @ (B.T @ P @ A)
    P_k = (A + B * F_k).T @ P @ (A + B * F_k) + F_k.T @ R @ F_k + Q

    F.append(F_k)

    P = P_k 
 
# 시간 순서로 변환
F.reverse()

for j in range(N):
    u_star = F[j] @ x_k
    x_star = A @ x_k + B @ u_star

    U.append(u_star)
    X.append(x_star)

    x_k = x_star


# F 데이터 추출
f1 = [f[0, 0] for f in F]
f2 = [f[0, 1] for f in F]

# F 그래프
plt.plot(range(N), f1, label='f1', color='b')
plt.plot(range(N), f2, label='f2', color='r')
plt.title('Control Matrix Elements over Time')
plt.xlabel('Time')
plt.ylabel('Control Matrix Elements')
plt.legend()
plt.xticks(range(0, N+10, 10))
plt.yticks(range(-7,3,1))
plt.grid()
plt.show()

# X 및 U 데이터 추출
u = [u[0, 0] for u in U]
x1 = [x[0, 0] for x in X]
x2 = [x[1, 0] for x in X]

# X 및 U 그래프
plt.plot(range(N), x1, label='x1', color='g')
plt.plot(range(N), x2, label='x2', color='m')
plt.plot(range(N), u, label='u', color='c')
plt.title('State Variables and Control Input over Time')
plt.xlabel('Time')
plt.ylabel('Variables')
plt.legend()
plt.xticks(range(0, N+10, 10))
plt.yticks(range(-10,9,1))
plt.grid()
plt.show()

