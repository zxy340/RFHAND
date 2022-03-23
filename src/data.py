import numpy as np
import torch
import sys
import cv2
from os.path import getsize
from complex_process import data_process
import os

RGBWidth = 256
RGBHeight = 256
chirp_num = 128  # Chirp Total
sample_num = 256  # Sample Length
numTxAntennas = 3  # the number of the Tx Antenna
numRxAntennas = 4  # the number of the Rx Antenna
padding_num = 64  # the length of the padding dimension

def scatter2image(scatter_data):
    image_data = np.zeros((len(scatter_data), 1, RGBHeight, RGBWidth), dtype=np.uint8)
    x_min = scatter_data[:, :, 0].min()
    x_max = scatter_data[:, :, 0].max()
    z_min = scatter_data[:, :, 1].min()
    z_max = scatter_data[:, :, 1].max()
    energy_max = scatter_data[:, :, 2].max()
    for frame in range(len(scatter_data)):
        for point in range(len(scatter_data[0])):
            x = int(float(scatter_data[frame, point, 0] - x_min) / (x_max - x_min) * (RGBWidth - 1))
            z = int(float(scatter_data[frame, point, 1] - z_min) / (z_max - z_min) * (RGBHeight - 1))
            # x_lower = max(x - 5, 0)
            # x_higher = min(x + 5, 1920)
            # z_lower = max(z - 5, 0)
            # z_higher = min(z + 5, 1080)
            energy = np.uint8(scatter_data[frame, point, 2] / energy_max * 255)
            image_data[frame, 0, z, x] = energy
            # image_data[frame, 0, z_lower:z_higher, x_lower:x_higher] = np.ones((z_higher-z_lower, x_higher-x_lower)) * energy
    return image_data

# .............load RGB and mmWave data plus the peaks information ................
def data_loader(current_data_index, num_per_act, top_size, CloudPoint_size):
    data = np.empty((1, top_size, CloudPoint_size))
    label = np.empty((1, 2))
    for index in range(1, num_per_act + 1):
        # ...............get the mmWave data and RGB data................
        if os.path.exists('../data/' + str(current_data_index) + '/' + str(index) + '/mmWave/mmWave.npy') & \
                os.path.exists('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/peaks.npy'):
            mmWave_syn_data = data_process(current_data_index, index, top_size, CloudPoint_size)
            if len(mmWave_syn_data) == 0:
                continue
            peaks = np.load('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/peaks.npy')
            peaks = peaks[:, 0, :].squeeze()   # get the x-y coordinate of the first hand keypoint
            peaks[:, 0] = peaks[:, 0] / 1920 * RGBWidth
            peaks[:, 1] = peaks[:, 1] / 1080 * RGBHeight
            squ = []
            for frame in range(len(peaks)):
                if (peaks[frame, 0] > 0) | (peaks[frame, 1] > 0):
                    squ.append(frame)
            data = np.concatenate([data, mmWave_syn_data[squ]], axis=0)
            label = np.concatenate([label, peaks[squ]], axis=0)
        print('The syn mmWave data with the shape of {} has been successfully loaded from ../data/{}/{}/mmWave/mmWave.npy'.format(
            np.shape(data), current_data_index, index))
        print('The syn peaks data with the shape of {} has been successfully loaded from ../data/{}/{}/RGB/peaks.npy'.format(
            np.shape(label), current_data_index, index))
    # data = data[1:]
    data = data[1:, :, [0, 2, 4]]
    data = scatter2image(data)
    # cv2.imshow('image', np.transpose(data[0], (1, 2, 0)))
    # cv2.waitKey(0)
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
        data = self.data[index]
        labels = self.label[index]
        labels[0] = labels[0] / RGBWidth
        labels[1] = labels[1] / RGBHeight
        return data, labels
    # for DataLoader better dividing the data, we use this function to return the length of the data
    def __len__(self):
        return len(self.data)