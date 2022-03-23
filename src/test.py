import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import torch.nn as nn

# pool of square window of size=3, stride=2
m = nn.MaxPool2d(3, stride=2)
# pool of non-square window
m = nn.MaxPool2d((3, 2), stride=(2, 1))
input = autograd.Variable(torch.randn(20, 16, 50, 32))
output = m(input)