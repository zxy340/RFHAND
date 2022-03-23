import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import numpy as np
import os

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Sequential(
            nn.Linear(32768, 1024, bias=False),
            nn.ReLU(),
            nn.Sigmoid())

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 128, bias=False),
            nn.ReLU(),
            nn.Sigmoid())

        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(128, 2, bias=False),
            nn.ReLU(),
            nn.Sigmoid())

    def forward(self, x):
        nn.Dropout(p=0.3),
        # print('the size of the input is {}'.format(np.shape(x)))
        out1 = self.layer1(x)
        # print('the size of the out1 is {}'.format(np.shape(out1)))
        out2 = self.layer2(out1)
        # print('the size of the out2 is {}'.format(np.shape(out2)))
        out3 = self.layer3(out2)
        # print('the size of the out3 is {}'.format(np.shape(out3)))
        out = out3.reshape(len(out3), -1)
        out = self.fc1(out)
        # print('the size of the fc1 is {}'.format(np.shape(out)))
        out = self.fc2(out)
        # print('the size of the fc2 is {}'.format(np.shape(out)))
        out = self.fc3(out)
        # print('the size of the fc3 is {}'.format(np.shape(out)))
        return out

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(128*6, 2, bias=False),
            nn.ReLU(),
            nn.Sigmoid())

    def forward(self, x):
        nn.Dropout(p=0.3),
        x = x.reshape(len(x), -1)
        out = self.fc(x)
        return out