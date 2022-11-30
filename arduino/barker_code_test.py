import numpy as np
import matplotlib.pyplot as plt

scale = 3
n = 100
some_signal = scale*np.random.rand(n)
index = np.random.randint(0, n-20)

barker_code = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
some_signal[index : index+len(barker_code)] += barker_code

auto = np.correlate(some_signal, barker_code, mode='full')

print(index)
print(np.argmax(auto))

plt.plot(some_signal)
plt.show()
plt.plot(auto)
plt.show()