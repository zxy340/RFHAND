import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(16),  # 对这16个结果进行规范处理，
            nn.ReLU(),	 # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=2, bias=False),	 # 14+2*2-5+1=14  该次卷积后output_size = 14*14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))	 # 14/2=7 池化后为7*7

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2, bias=False),	 # 14+2*2-5+1=14  该次卷积后output_size = 14*14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))	 # 14/2=7 池化后为7*7

        self.fc1 = nn.Sequential(
            nn.Linear(21760, 1024, bias=False),
            nn.ReLU(),
            nn.Sigmoid())

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 128, bias=False),
            nn.ReLU(),
            nn.Sigmoid())

        self.fc3 = nn.Sequential(
            nn.Linear(128, 2, bias=False),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Sigmoid())

    def forward(self, x):
        # nn.Dropout(p=0.3),
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = out3.reshape(len(out3), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        # return out
        return out