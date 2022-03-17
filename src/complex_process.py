import numpy as np
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt

def clutter_removal(input_val, axis=0): #
    # Reorder the axes
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)
    # Apply static clutter removal
    mean = input_val.mean(0)
    output_val = input_val - mean
    return output_val.transpose(reordering)

def naive_xyz(virtual_ant, num_tx=3, num_rx=4, fft_size=64): #
    assert num_tx > 2, "need a config for more than 2 TXs"
    num_detected_obj = virtual_ant.shape[1]
    azimuth_ant = virtual_ant[[8, 9, 4, 5], :]
    azimuth_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    azimuth_ant_padded[:num_rx, :] = azimuth_ant

    # Process azimuth information
    azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
    k_max = np.argmax(np.log2(np.abs(azimuth_fft)), axis=0)
    peak_1 = np.zeros_like(k_max, dtype=np.complex_)
    for i in range(len(k_max)):
        peak_1[i] = azimuth_fft[k_max[i], i]

    k_max[k_max > (fft_size // 2) - 1] = k_max[k_max > (fft_size // 2) - 1] - fft_size
    wx = 2 * np.pi / fft_size * k_max
    x_vector = wx / np.pi

    # Zero pad elevation
    elevation_ant = virtual_ant[[11, 10, 7, 6], :]
    elevation_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    elevation_ant_padded[:num_rx, :] = elevation_ant

    # Process elevation information
    elevation_fft = np.fft.fft(elevation_ant, axis=0)
    elevation_max = np.argmax(np.log2(np.abs(elevation_fft)), axis=0)  # shape = (num_detected_obj, )
    peak_2 = np.zeros_like(elevation_max, dtype=np.complex_)
    for i in range(len(elevation_max)):
        peak_2[i] = elevation_fft[elevation_max[i], i]

    # Calculate elevation phase shift
    wz = np.angle(peak_1 * peak_2.conj())
    z_vector = wz / np.pi
    ypossible = 1 - x_vector ** 2 - z_vector ** 2
    y_vector = ypossible
    x_vector[ypossible < 0] = 0
    z_vector[ypossible < 0] = 0
    y_vector[ypossible < 0] = 0
    y_vector = np.sqrt(y_vector)
    return x_vector, y_vector, z_vector

def data_process(current_data_index, index):
    if os.path.exists('../data/' + str(current_data_index) + '/' + str(index) + '/mmWave/mmWave.npy') & \
            os.path.exists('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/RGB.npy'):
        mmWave_data = np.load('../data/' + str(current_data_index) + '/' + str(index) + '/mmWave/mmWave.npy')
        print(
            'The syn mmWave data with the shape of {} has been successfully loaded from ../data/{}/{}/mmWave/mmWave.npy'.format(
                np.shape(mmWave_data), current_data_index, index)
        )
        RGB_data = np.load('../data/' + str(current_data_index) + '/' + str(index) + '/RGB/RGB.npy')
        RGB_data = np.flip(RGB_data[:, :, :, [2, 1, 0, 3]], 2)
        print(
            'The syn RGB data with the shape of {} has been successfully loaded from ../data/{}/{}/RGB/RGB.npy'.format(
                np.shape(RGB_data), current_data_index, index)
        )

        sample_num = 256  # Sample Length
        chirp_num = 128  # Chirp Total
        frame_num = len(mmWave_data)  # Frame Total
        FrameTime = 0.1  # the period of one frame
        LightVelocity = 3e8  # Speed of Light
        FreqStart = 60e9  # Start Frequency
        slope = 60.012e12  # Frequency slope
        SampleRate = 4400e3  # Sample rate
        Bandwidth = slope * sample_num / SampleRate  # Sweep Bandwidth
        WaveLength = LightVelocity / FreqStart  # Wave length
        numTxAntennas = 3  # the number of the Tx Antenna
        numRxAntennas = 4  # the number of the Rx Antenna
        padding_num = 64  # the length of the padding dimension

        frameWithChirp = np.reshape(mmWave_data, (frame_num, numTxAntennas, numRxAntennas, chirp_num, -1))
        frameWithChirp = np.flip(frameWithChirp, 4)
        print('the shape of the data before rangeFFT is {}'.format(np.shape(frameWithChirp)))

        azimuth_range = np.zeros((frame_num, chirp_num, padding_num, sample_num), dtype=complex)
        for frame in range(frame_num):
            print(frame)
            # get 1D range profile->rangeFFT
            windowedBins1D = frameWithChirp[frame] * np.hamming(sample_num)
            rangeFFTResult = np.fft.fft(windowedBins1D)
            rangeFFTResult = clutter_removal(rangeFFTResult, axis=2)
            # get 2D range-velocity profile->dopplerFFT
            windowedBins2D = rangeFFTResult * np.reshape(np.hamming(chirp_num), (1, 1, -1, 1))
            dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2)
            dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)
            # get 2D range-angle profile->angleFFT
            dopplerResultSumAllAntenna = np.sum(dopplerFFTResult, axis=(0, 1))
            dopplerResultInDB  = np.log10(np.absolute(dopplerResultSumAllAntenna))
            # filter out the bins which are too far from radar
            dopplerResultInDB[:, 15:] = -100

            cfarResult = np.zeros(dopplerResultInDB.shape, bool)
            top_size = 32
            energyThre128 = np.partition(dopplerResultInDB.ravel(), 128 * 256 - top_size - 1)[
                128 * 256 - top_size - 1]
            cfarResult[dopplerResultInDB > energyThre128] = True

            det_peaks_indices = np.argwhere(cfarResult == True)
            R = det_peaks_indices[:, 1].astype(np.float64)
            V = (det_peaks_indices[:, 0] - chirp_num // 2).astype(np.float64)
            R *= LightVelocity / (2 * Bandwidth)
            V *= WaveLength / (2 * FrameTime)
            energy = dopplerResultInDB[cfarResult == True]

            AOAInput = dopplerFFTResult[:, :, cfarResult == True]
            AOAInput = AOAInput.reshape(12, -1)
            x_vec, y_vec, z_vec = naive_xyz(AOAInput)
            x, y, z = x_vec * R, y_vec * R, z_vec * R
            pointCloud = np.concatenate((x, y, z, V, energy, R))
            pointCloud = np.reshape(pointCloud, (6, -1))
            pointCloud = pointCloud[:, y_vec != 0]

            fig = plt.figure()
            # ax1 = fig.add_subplot(211, projection='3d')
            # ax1.scatter(x, y, z)
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # min_v = min(energy)
            # max_v = max(energy)
            # print(min_v)
            # print(max_v)
            min_v = 0
            max_v = 7
            color = [plt.get_cmap("seismic", 100)(int(float(i - min_v) / (max_v - min_v) * 100)) for i in energy]
            ax1 = fig.add_subplot(211)
            plt.set_cmap(plt.get_cmap("seismic", 100))
            im = ax1.scatter(x, z, c=color)
            fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x, pos: int(x*(max_v-min_v)+min_v)))
            plt.xlabel('X')
            plt.ylabel('Z')

            ax2 = fig.add_subplot(212)
            ax2.imshow(RGB_data[frame])
            plt.show()
        return pointCloud
    else:
        print('there is no data to be processed!')
        return 0