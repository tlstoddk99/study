import numpy as np
import matplotlib.pyplot as plt

lambda_ = 20
eta = 0.1

x = np.array([[0], [0]])
delt = 1/100
t = np.arange(0, 10+delt, delt)
Hz = 2
amp=2


xd = amp * np.sin(t / Hz)
xd_dot = amp * np.cos(t / Hz) / Hz
xd_ddot = -amp * np.sin(t / Hz) / (Hz * Hz)

s = np.zeros(len(t))
sat_ = np.zeros(len(t))
u = np.zeros(len(t))

for i in range(len(t)):
    x_tilde = x[0, i] - xd[i]
    x_tildedot = x[1, i] - xd_dot[i]
    s[i] = x_tildedot + lambda_ * x_tilde
    F = 0.5 * x[1, i] ** 2 * abs(np.cos(3 * x[0, i]))

    # if s[i] > 1:
    #     sat_[i] = 1
    # elif s[i] < -1:
    #     sat_[i] = -1
    # else:
    #     sat_[i] = 0#s[i] / 0.1

    f_hat = -(1.5 * x[1, i] ** 2 * np.cos(3 * x[0, i]))
    u[i] = -f_hat + xd_ddot[i] - 20 * x_tildedot - (F + eta) * np.sign(s[i])

    at = abs(np.sin(t[i])) + 1
    f = np.array([[x[1, i]], [-at * x[1, i] ** 2 * np.cos(3 * x[0, i]) + u[i]]])
    x = np.hstack((x, x[:, i:i+1] + delt * f))

plt.figure(1)
plt.plot(t, x[0, :-1] - xd, label='Original')
plt.title('Tracking Error')
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.legend()

plt.figure(2)
plt.plot(t, u, label='Original')
plt.title('Control Input')
plt.xlabel('Time (s)')
plt.ylabel('u')
plt.legend()


plt.figure(3)
plt.plot(t, x[0, :-1], label='Original')
plt.plot(t, xd, label='Desired')
plt.title('Position')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend()
plt.show()