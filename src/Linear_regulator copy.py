import numpy as np
import matplotlib.pyplot as plt

# System parameters
# 시스템 파라미터 설정
A = np.array([[0.9974, 0.0539],
              [-0.1078, 1.1591]])  # State transition matrix
B = np.array([[0.0013],
              [0.0539]])          # Control input matrix
R = np.array([[0.01]])             # Weight matrix for control input
Q = np.array([[0.25, 0.00],
              [0.00, 0.05]])      # Weight matrix for state error
H = np.array([[0.0, 0.0],
              [0.0, 0.0]])        # Initial state error covariance matrix
P = H                             # Initial state error covariance matrix
N = 200                           # Time steps

# Initialize variables
# 변수 초기화
control_matrices = []  # 제어 행렬 저장 리스트
control_inputs = []    # 제어 입력 저장 리스트
state_variables = []   # 상태 변수 저장 리스트

initial_state = np.array([[2.0], [1.0]])  # Initial state

# Calculate control matrices backwards
# 뒤에서부터 계산하여 제어 행렬 계산
for i in range(N, 0, -1):
    # Calculate control matrix
    # 제어 행렬 계산
    control_matrix = (-np.linalg.inv(R + B.T @ P @ B)) @ (B.T @ P @ A)
    # Update state error covariance matrix
    # 상태 공분산 행렬 업데이트
    P = (A + B @ control_matrix).T @ P @ (A + B @ control_matrix) + control_matrix.T @ R @ control_matrix + Q

    control_matrices.append(control_matrix)

# Reverse control matrices
# 제어 행렬을 역순으로 저장
control_matrices.reverse()

# Calculate state and control input forwards
# 계산된 제어 행렬을 사용하여 상태 변수와 제어 입력 계산
for j in range(N):
    control_input = control_matrices[j] @ initial_state           # Calculate control input
    next_state = A @ initial_state + B @ control_input  # Calculate state variable

    control_inputs.append(control_input)
    state_variables.append(next_state)

    initial_state = next_state

# Extract control matrix data
# 제어 행렬 데이터 추출
control_matrix_1 = [control_matrix[0, 0] for control_matrix in control_matrices]
control_matrix_2 = [control_matrix[0, 1] for control_matrix in control_matrices]

# Plot control matrices
# 제어 행렬 그래프
plt.plot(range(N), control_matrix_1, label="Control Matrix 1", color="b")
plt.plot(range(N), control_matrix_2, label="Control Matrix 2", color="r")
plt.title("Control Matrix Elements over Time")
plt.xlabel("Time")
plt.ylabel("Control Matrix Elements")
plt.legend()
plt.grid()
plt.show()

# Extract state and control input data
# 상태 변수와 제어 입력 데이터 추출
control_inputs_data = [control_input[0, 0] for control_input in control_inputs]
state_variable_1 = [next_state[0, 0] for next_state in state_variables]
state_variable_2 = [next_state[1, 0] for next_state in state_variables]

# Plot state variables and control inputs
# 상태 변수와 제어 입력 그래프
plt.plot(range(N), state_variable_1, label="State Variable 1", color="g")
plt.plot(range(N), state_variable_2, label="State Variable 2", color="m")
plt.plot(range(N), control_inputs_data, label="Control Input", color="c")
plt.title("State Variables and Control Input over Time")
plt.xlabel("Time")
plt.ylabel("Variables")
plt.legend()
plt.grid()
plt.show()
