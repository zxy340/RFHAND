import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import torch.nn as nn

a = "123456"
b = a
print(id(a))
print(id(b))

b = b + "789"
print(a[0])
print(b[0])
print(id(a[0]))
print(id(b[0]))