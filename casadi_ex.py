from casadi import *

N = 100 # 제어 구간의 수

opti = Opti() # 최적화 문제 설정

# ---- 결정 변수 ---------
X = opti.variable(2,N+1) # 상태 변수 경로
p_x   = X[0,:]           # 위치
speed = X[1,:]           # 속도
U = opti.variable(1,N)   # 제어 변수 경로 (스로틀)
T = opti.variable()      # 최종 시간

# ---- 목적 함수 ---------
opti.minimize(T) # 최소 시간을 목표로 경주

# ---- 동적 제약 조건 --------
f = lambda x,u: vertcat(x[1],u-x[1]) # dx/dt = f(x,u) 

dt = T/N # 제어 구간의 길이
for k in range(N): # 제어 구간을 따라 반복
   # Runge-Kutta 4 차수 적분
   k1 = f(X[:,k],         U[:,k])
   k2 = f(X[:,k]+dt/2*k1, U[:,k])
   k3 = f(X[:,k]+dt/2*k2, U[:,k])
   k4 = f(X[:,k]+dt*k3,   U[:,k])
   x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
   opti.subject_to(X[:,k+1]==x_next) # 연속성 제약 조건

# ---- 경로 제약 조건 -----------
limit = lambda pos: 1-sin(2*pi*pos)/2
opti.subject_to(speed<=limit(p_x))   # 트랙 속도 제한
opti.subject_to(opti.bounded(0,U,1)) # 제어 변수 제한

# ---- 경계 조건 --------
opti.subject_to(p_x[0]==0)   # 시작 위치 0
opti.subject_to(speed[0]==0) # 정지 상태에서 시작
opti.subject_to(p_x[-1]==3)  # 목표 위치 1

# ---- 기타 제약 조건  ----------
opti.subject_to(T>=0) # 시간은 양수여야 함

# ---- 초기 값 설정 ----
opti.set_initial(speed, 1)
opti.set_initial(T, 1)

# ---- NLP 문제 풀기 ----
opti.solver("ipopt") # 수치적 솔버 설정
sol = opti.solve()   # 실제 문제 풀기

# ---- 후처리 ----
import matplotlib.pyplot as plt
plt.figure()
plt.plot(sol.value(speed),label="speed")       # 속도 플롯
plt.plot(sol.value(p_x),label="pos")           # 위치 플롯
plt.plot(limit(sol.value(p_x)),'r--',label="speed limit") # 속도 제한 플롯
plt.step(range(N),sol.value(U),'k',label="throttle")      # 스로틀 플롯
plt.legend(loc="upper left")


plt.show()

