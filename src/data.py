import numpy as np
import torch
from os.path import getsize
from complex_process import data_process
import os

RGBWidth = 1920
RGBHeight = 1080

# .............load RGB and mmWave data plus the peaks information ................
def data_loader(current_data_index, num_per_act):
    data = np.empty((1, 64, 256))
    label = np.empty((1, 2))
    for index in range(1, num_per_act + 1):
        # ...............get the mmWave data and RGB data................
        if os.path.exists('../data/' + str(current_data_index) + '/' + str(index) + '/mmWave/mmWave.npy') & \
                os.path.exists('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/peaks.npy'):
            mmWave_syn_data = data_process(current_data_index, index)
            mmWave_syn_data = abs(mmWave_syn_data)
            mmWave_syn_data = np.reshape(mmWave_syn_data, (len(mmWave_syn_data) * 255, 64, 256))
            peaks = np.load('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/peaks.npy')
            peaks = peaks[:, 0, :].squeeze()
            peaks = peaks[2:]
            peaks = np.repeat(peaks, 255, axis=0)
            data = np.concatenate([data, mmWave_syn_data], axis=0)
            label = np.concatenate([label, peaks], axis=0)
        print('The syn mmWave data with the shape of {} has been successfully loaded from ../data/{}/{}/mmWave/mmWave.npy'.format(
            np.shape(data), current_data_index, index))
        print('The syn peaks data with the shape of {} has been successfully loaded from ../data/{}/{}/RGB/peaks.npy'.format(
            np.shape(label), current_data_index, index))
    data = data[1:]
    data = data[:, np.newaxis, :, :]
    label = label[1:]
    seq = np.arange(0, len(data), 1)
    np.random.shuffle(seq)
    x_train = data[seq[:int(len(data)/5*4)]]
    y_train = label[seq[:int(len(data)/5*4)]]
    x_test = data[seq[int(len(data)/5*4):]]
    y_test = label[seq[int(len(data)/5*4):]]
    return x_train, y_train, x_test, y_test
# ..................................................................................

class GetLoader(torch.utils.data.Dataset):
    # initial function, get the data and label
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index is divided based on the batchsize, finally return the data and corresponding labels
    def __getitem__(self, index):
        data = abs(self.data[index])
        labels = self.label[index]
        labels[0] = labels[0] / RGBWidth
        labels[1] = labels[1] / RGBHeight
        return data, labels
    # for DataLoader better dividing the data, we use this function to return the length of the data
    def __len__(self):
        return len(self.data)