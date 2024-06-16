import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path = '/home/a/Downloads/mpc_homework/dynamics.txt'
data = pd.read_csv(file_path, sep='\t', header=None)

# Plot 1
plt.subplot(1, 3, 1)
plt.plot(data[0], data[1])
plt.plot(data[0], data[2])
plt.legend(['theta', 'disired theta'])
plt.title('Time Theta-Disired Theta Plot')
plt.xlabel('Time (s)')

# Plot 2
plt.subplot(1, 3, 2)
plt.plot(data[0], data[3])
plt.plot(data[0], data[4])
plt.legend(['theta_dot', 'disired theta_dot'])
plt.title('Time Theta_dot-Disired Theta_dot Plot')
plt.xlabel('Time (s)')

# # Plot 3
plt.subplot(1, 3, 3)
plt.plot(data[0], data[5])
plt.legend(['Torque'])
plt.title('Time Torque Plot')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()
