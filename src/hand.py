import cv2
import json
import numpy as np
import math
import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import torch
import os
from skimage.measure import label

from model import handpose_model
import util

class Hand(object):
    def __init__(self, model_path):
        self.model = handpose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        scale_search = [0.5, 1.0, 1.5, 2.0]
        # scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 22))
        # paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float()
            if torch.cuda.is_available():
                data = data.cuda()
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                output = self.model(data).cpu().numpy()
                # output = self.model(data).numpy()q

            # extract outputs, resize, and remove padding
            heatmap = np.transpose(np.squeeze(output), (1, 2, 0))  # output 1 is heatmaps
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            heatmap_avg += heatmap / len(multiplier)

        all_peaks = []
        for part in range(21):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)
            # 全部小于阈值
            if np.sum(binary) == 0:
                all_peaks.append([0, 0])
                continue
            label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
            max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0

            y, x = util.npmax(map_ori)
            all_peaks.append([x, y])
        return np.array(all_peaks)

if __name__ == "__main__":
    hand_estimation = Hand('../model/hand_pose_model.pth')

    current_data_index = 0  # indicate the current loaded action index
    num_per_act = 1  # indicate the data folder number of one action

    for index in range(1, num_per_act + 1):
        # ...............get the mmWave data and RGB data................
        if os.path.exists('../data/' + str(current_data_index) + '/' + str(index) + '/mmWave/mmWave.npy') & \
                os.path.exists('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/RGB.npy') & \
                os.path.exists('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/Depth.npy'):
            RGB_syn_data = np.load('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/RGB.npy')
            print('The syn RGB data has been successfully loaded from ../data/' + str(current_data_index) + '/' + str(index) + '/RGB/RGB.npy')
            print('The shape of the RGB data is {}'.format(np.shape(RGB_syn_data)))
            peaks_all = np.empty((len(RGB_syn_data), 21, 2))
            if not os.path.exists('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/peaks.npy'):
                for i in range(len(RGB_syn_data)):
                    print('Current processed image index is {}, the total number of images to be processed is {}'.format(i, len(RGB_syn_data)))
                    oriImg = RGB_syn_data[i, :, :, :3]  # B,G,R order
                    peaks = hand_estimation(oriImg)
                    peaks_all[i] = peaks
                np.save('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/peaks.npy', peaks_all)
            peaks_all = np.load('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/peaks.npy')
            for i in range(len(RGB_syn_data)):
                canvas = util.draw_handpose(RGB_syn_data[i, :, :, :3], peaks_all[i], True)
                # name = "../images/Depth" + str(i) + ".jpg"
                # cv2.imwrite(name, canvas)
                cv2.imshow('', canvas)
                cv2.waitKey(0)