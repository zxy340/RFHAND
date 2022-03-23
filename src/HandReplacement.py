import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from data import GetLoader, data_loader
from CNN import CNN, Linear
import torch
import os
import matplotlib.pyplot as plt
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"  #（代表仅使用第0，1号GPU）
# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# .............load RGB and mmWave data plus the peaks information ................
# if os.path.exists('../data/x_train.npy') & os.path.exists('../data/y_train.npy') \
#      & os.path.exists('../data/x_test.npy') & os.path.exists('../data/y_test.npy'):
#     x_train = np.load('../data/x_train.npy')
#     y_train = np.load('../data/y_train.npy')
#     x_test = np.load('../data/x_test.npy')
#     y_test = np.load('../data/y_test.npy')
# else:
current_data_index = 0  # indicate the current loaded action index
num_per_act = 1  # indicate the data folder number of one action
top_size = 128  # indicate how many points are selected from one frame
CloudPoint_size = 6  # x, y, z, V, energy, R
x_train, y_train, x_test, y_test = data_loader(current_data_index, num_per_act, top_size, CloudPoint_size)
np.save('../data/x_train.npy', x_train)
np.save('../data/y_train.npy', y_train)
np.save('../data/x_test.npy', x_test)
np.save('../data/y_test.npy', y_test)
print('The shape of x_train, y_train, x_test, y_test are {}, {}, {}, {}'.format(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)))
# ..................................................................................

# .............load RGB and mmWave data plus the peaks information ................
# image = x_train[0].squeeze()
# image = abs(image)
# plt.imshow(image)
# plt.show()
# ..................................................................................

# .............basic information of training and testing set.......................
training_data_count = len(x_train)  # number of training series
testing_data_count = len(x_test)  # number of testing series
# ..................................................................................

# ..................................................................................
# use GetLoader to load the data and return Dataset object, which contains data and labels
torch_data = GetLoader(x_train, y_train)
train_data = DataLoader(torch_data, batch_size=16, shuffle=True, drop_last=False)
torch_data = GetLoader(x_test, y_test)
test_data = DataLoader(torch_data, batch_size=16, shuffle=True, drop_last=False)
# ....................................................................................

# .............Hyper Parameters and initial model parameters..........................
epochs = 10
lr = 0.001  # learning rate
# initial model
model = CNN().to(device)
# model = Linear().to(device)
for name, param in model.named_parameters():
    nn.init.normal_(param)
# loss and optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)
# optimizer = torch.optim.SGD(model.parameters(), lr)
# .....................................................................................

# ...........................train and store the model.................................
# train the model
for epoch in range(1, epochs + 1):
    # if epoch % 3 == 0:
    #     lr = lr / 10
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    for i, (data, labels) in enumerate(train_data):
        data = data.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(data.float())
        loss = criterion(outputs.float(), labels.float())
        loss = loss + 0.01 * (torch.norm(outputs[:, 0]) + torch.norm(outputs[:, 1]))
        optimizer.zero_grad()

        # backward and optimize
        loss.backward()
        optimizer.step()

        # if i % 30 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch, epochs, i + 1, training_data_count / 16, loss.item()))

# ........................................................................................

# ............................test the trained model.............................
# Test the model
model.eval()
with torch.no_grad():
    score = 0
    for data, label in test_data:
        data = data.to(device)
        label = label.to(device)
        outputs = model(data.float())
        score = score + torch.sqrt(((label - outputs) ** 2).sum())
        print(outputs)

        x1 = outputs.cpu().numpy()[:, 0]
        y1 = outputs.cpu().numpy()[:, 1]
        x2 = label.cpu().numpy()[:, 0]
        y2 = label.cpu().numpy()[:, 1]
        plt.plot(x1, y1, 'g^', x2, y2, 'bs')
        plt.show()
        plt.pause(5)

    print('The average error on the {} test mmWave data: {}'.format(testing_data_count, score / testing_data_count))
# ..................................................................................................