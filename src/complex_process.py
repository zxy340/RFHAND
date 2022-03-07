import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def data_process(current_data_index, index):
    index = index + 8
    if os.path.exists('../data/' + str(current_data_index) + '/' + str(index) + '/mmWave/mmWave.npy') & \
            os.path.exists('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/RGB.npy'):
        mmWave_data = np.load('../data/' + str(current_data_index) + '/' + str(index) + '/mmWave/mmWave.npy')
        print(
            'The syn mmWave data with the shape of {} has been successfully loaded from ../data/{}/{}/mmWave/mmWave.npy'.format(
                np.shape(mmWave_data), current_data_index, index)
        )
        RGB_data = np.load('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/RGB.npy')
        RGB_data = RGB_data[:, :, :, [2, 1, 0, 3]]
        print(
            'The syn RGB data with the shape of {} has been successfully loaded from ../data/{}/{}/RGB/RGB.npy'.format(
                np.shape(RGB_data), current_data_index, index)
        )
        Bandwidth = 3657.8e6  # Sweep Bandwidth
        sample_num = 256  # Sample Length
        chirp_num = 255  # Chirp Total
        frame_num = len(mmWave_data)  # Frame Total
        channel_num = 4  # Channel Total
        LightVelocity = 3e8  # Speed of Light
        FreqStart = 77e9  # Start Frequency
        numTxAntennas = 1  # the number of the Tx Antenna
        numRxAntennas = 4  # the number of the Rx Antenna
        padding_num = 64  # the length of the padding dimension

        frameWithChirp = np.reshape(mmWave_data, (frame_num, numTxAntennas, numRxAntennas, chirp_num, -1))
        frameWithChirp = np.flip(frameWithChirp, 4)
        print('the shape of the data before rangeFFT is {}'.format(np.shape(frameWithChirp)))

        azimuth_range = np.zeros((frame_num, chirp_num, padding_num, sample_num), dtype=complex)
        for frame in range(frame_num):
            # get 1D range profile->rangeFFT
            windowedBins1D = frameWithChirp[frame] * np.hamming(sample_num)
            rangeFFTResult = np.fft.fft(windowedBins1D)
            # get 2D range-velocity profile->dopplerFFT
            windowedBins2D = rangeFFTResult * np.reshape(np.hamming(chirp_num), (1, 1, -1, 1))
            dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2)
            dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)
            # get 2D range-angle profile->angleFFT
            dopplerResultInDB = np.log10(np.absolute(dopplerFFTResult))
            AOAInput = dopplerResultInDB.reshape(numTxAntennas * numRxAntennas, chirp_num, sample_num)
            azimuth_ant_padded = np.zeros((padding_num, chirp_num, sample_num), dtype=complex)
            azimuth_ant_padded[:len(AOAInput), :] = AOAInput
            azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
            azimuth_fft = np.fft.fftshift(azimuth_fft, axes=0)

            # for chirp in range(chirp_num):
            #     azimuth_range[frame, chirp, :, :] = azimuth_fft[:, chirp, :]

            plt.subplot(221)
            plt.imshow(abs(rangeFFTResult[0][0]))

            plt.subplot(222)
            plt.imshow(abs(dopplerFFTResult[0][0]))

            plt.subplot(223)
            plt.imshow(abs(azimuth_fft[:, 0, :]))
            # azimuth_fft_show = np.zeros((numTxAntennas * numRxAntennas * chirp_num // 4 + 1, sample_num), dtype=complex)
            # for chirp in range(0, chirp_num, 4):
            # # azimuth_fft_show = np.zeros((200, sample_num), dtype=complex)
            # # for chirp in range(100, 150, 1):
            #     azimuth_fft_show[numTxAntennas * numRxAntennas * chirp // 4:numTxAntennas * numRxAntennas * (chirp // 4 + 1), :]\
            #         = azimuth_fft[:, chirp, :]
            #     # azimuth_fft_show[numTxAntennas * numRxAntennas * (chirp - 100):numTxAntennas * numRxAntennas * (chirp - 100 + 1), :] \
            #     #     = azimuth_fft[:, chirp, :]
            # plt.subplot(223)
            # plt.imshow(abs(azimuth_fft_show))

            plt.subplot(224)
            plt.imshow(RGB_data[frame])
            plt.show()
        return azimuth_range[2:]
    else:
        print('there is no data to be processed!')
        return 0