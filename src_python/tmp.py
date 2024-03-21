import matplotlib.pyplot as plt
import numpy as np

N = 100
x = np.arange(N)
mean_1 = 25 + np.random.normal(0.1, 1, N).cumsum()
std_1 = 3 + np.random.normal(0, .08, N).cumsum()

mean_2 = 15 + np.random.normal(0.2, 1, N).cumsum()
std_2 = 4 + np.random.normal(0, .1, N).cumsum()

plt.plot(x, mean_1, 'b-', label='mean_1')
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
plt.plot(x, mean_2, 'r--', label='mean_2')
plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)

plt.legend(title='title')
plt.show()