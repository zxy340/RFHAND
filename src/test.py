import numpy as np
import cv2
import matplotlib.pyplot as plt

a1 = 10
a = a1 * np.exp(complex(0, 1) * 5)
array = [1, np.exp(complex(0, 1)), np.exp(complex(0, 1) * 2), np.exp(complex(0, 1) * 3), np.exp(complex(0, 1) * 4), np.exp(complex(0, 1) * 5), np.exp(complex(0, 1) * 6), np.exp(complex(0, 1) * 7)]
b = []
for i in range(len(array)):
    b.append(array[i] * a)
result = np.fft.fft(b)
plt.plot(result)
plt.show()