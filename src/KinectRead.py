import numpy as np
import cv2
import time
import os

current_data_index = 11  # indicate the current loaded data index

# ...............get the mmWave data and RGB data................
if os.path.exists('../data/' + str(current_data_index) + '/mmWave/mmWave.npy') & \
        os.path.exists('../data/' + str(current_data_index) + '/RGB/RGB.npy') & \
        os.path.exists('../data/' + str(current_data_index) + '/RGB/Depth.npy'):
    RGB_syn_data = np.load('../data/' + str(current_data_index) + '/RGB/RGB.npy')
    Depth_syn_data = np.load('../data/' + str(current_data_index) + '/RGB/Depth.npy')
    for i in range(len(RGB_syn_data)):
        name = "../images/RGB" + str(i) + ".jpg"
        cv2.imwrite(name, RGB_syn_data[i])
    for i in range(len(Depth_syn_data)):
        name = "../images/Depth" + str(i) + ".jpg"
        cv2.imwrite(name, Depth_syn_data[i])
    mmWave_syn_data = np.load('../data/' + str(current_data_index) + '/mmWave/mmWave.npy')
    print('The syn RGB data has been successfully loaded from ./data/' + str(current_data_index) + '/RGB/RGB.npy')
    print('The shape of the RGB data is {}'.format(np.shape(RGB_syn_data)))
    print('The syn Depth data has been successfully loaded from ./data/' + str(current_data_index) + '/RGB/Depth.npy')
    print('The shape of the Depth data is {}'.format(np.shape(Depth_syn_data)))
    print('The syn FFT data has been successfully loaded from ./data/' + str(current_data_index) + '/mmWave/mmWave.npy')
    print('The shape of the mmWave data is {}'.format(np.shape(mmWave_syn_data)))
else:
    # ...............get the RGB image data.............
    typ = np.dtype((np.uint8, (1080, 1920, 4)))
    RGB_data = np.fromfile('../data/' + str(current_data_index) + '/RGB/RGBdata.txt', dtype=typ)
    RGB_data = RGB_data[:, :, :, [2, 1, 0, 3]]
    print('The shape of the raw RGB data is {}'.format(np.shape(RGB_data)))
    # for i in range(len(RGB_data)):
    #     name = "./images/RGB" + str(i) + ".jpg"
    #     cv2.imwrite(name, RGB_data[i])
    # ..................................................

    # ...............get the Depth image data.............
    typ = np.dtype((np.uint8, (424, 512, 3)))
    Depth_data = np.fromfile('../data/' + str(current_data_index) + '/RGB/Depdata.txt', dtype=typ)
    print('The shape of the raw Depth data is {}'.format(np.shape(Depth_data)))
    # for i in range(len(Depth_data)):
    #     name = "./images/Depth" + str(i) + ".jpg"
    #     cv2.imwrite(name, Depth_data[i])
    # ..................................................

    # ...............get the RGB image timestamp, the same as Depth........
    RGB_timestamp_raw = np.loadtxt('../data/' + str(current_data_index) + '/RGB/timestamp.txt', dtype=str)
    RGB_timestamp = np.empty(len(RGB_timestamp_raw), dtype=float)
    for i in range(len(RGB_timestamp_raw)):
        ymdhms = RGB_timestamp_raw[i][0] + '-' + RGB_timestamp_raw[i][1] + '-' + RGB_timestamp_raw[i][2] + ' ' \
                 + RGB_timestamp_raw[i][3] + ':' + RGB_timestamp_raw[i][4] + ':' + RGB_timestamp_raw[i][5]
        ms = RGB_timestamp_raw[i][6]
        RGB_timestamp[i] = time.mktime(time.strptime(ymdhms, "%Y-%m-%d %H:%M:%S")) + float(ms) / 1000.
    # ..................................................

    dfile = '../data/' + str(current_data_index) + '/mmWave/'
    timestamp = ""
    frame_slot = 0.034
    for root, dirs, files in os.walk(dfile):
        for name in files:
            if 'bin ' in name + ' ':
                dfile = dfile + name
                timestamp, _, _ = name.split('_')
                timestamp = float(timestamp)

    Bandwidth = 3657.8e6  # Sweep Bandwidth
    frame_num = 1000  # Frame total
    sample_num = 256  # Sample Length
    chirp_num = 255  # Chirp Total
    channel_num = 4  # Channel Total
    LightVelocity = 3e8  # Speed of Light
    FreqStart = 77e9  # Start Frequency
    NumRangeFFT = 256  # Range FFT Length


    class AWR1642:
        def __init__(
                self,
                dfile,
                sample_rate,
                num_frame=1000,
                num_chirp=255,
        ):
            num_channel = 4
            x = np.fromfile(dfile, dtype=np.int16)
            x = x.reshape(num_frame, num_chirp, num_channel, -1, 4)  # 2I + 2Q = 4
            # 关于IQ信号，可以参考https://sunjunee.github.io/2017/11/25/what-is-IQ-signal/
            x_I = x[:, :, :, :, :2].reshape(
                num_frame, num_chirp, num_channel, -1
            )  # flatten the last two dims of I data
            x_Q = x[:, :, :, :, 2:].reshape(
                num_frame, num_chirp, num_channel, -1
            )  # flatten the last two dims of Q data
            data = np.array((x_I, x_Q))  # data[I/Q, Frame, Chirp, Channel, Sample]
            self.data = np.transpose(
                data, (0, 3, 1, 2, 4)
            )  # data[I/Q, Channel, Frame, Chirp, Sample]
            self.sample_rate = sample_rate


    mmWave_Data = AWR1642(dfile, 2099)

    index_RGB = 0
    max_syn_der = 0.02
    RGB_syn_data = []
    Depth_syn_data = []
    mmWave_syn_data = []
    for frame in range(frame_num):
        mmWave_cur_timestamp = timestamp + frame_slot * frame
        while (index_RGB < len(RGB_timestamp) - 1) & (RGB_timestamp[index_RGB] - mmWave_cur_timestamp < 0):
            index_RGB = index_RGB + 1
        if index_RGB >= len(RGB_timestamp) - 1:
            break
        if RGB_timestamp[index_RGB] - mmWave_cur_timestamp > max_syn_der:
            continue
        else:
            sigRangeFFT = np.zeros((chirp_num, sample_num * 4), dtype=complex)
            for channel in range(channel_num):
                # get time_series mmWave signal
                sigReceive = np.zeros((chirp_num, sample_num), dtype=complex)
                for chirp in range(chirp_num):
                    for sample in range(sample_num):
                        sigReceive[chirp][sample] = complex(0, 1) * mmWave_Data.data[0, channel, frame, chirp, sample] + mmWave_Data.data[1, channel, frame, chirp, sample]

            index_RGB = index_RGB + 1
            RGB_syn_data.append(RGB_data[index_RGB])
            Depth_syn_data.append(Depth_data[index_RGB])
            mmWave_syn_data.append(sigRangeFFT)
    np.save('../data/' + str(current_data_index) + '/RGB/RGB.npy', RGB_syn_data)
    np.save('../data/' + str(current_data_index) + '/RGB/Depth.npy', Depth_syn_data)
    np.save('../data/' + str(current_data_index) + '/mmWave/mmWave.npy', mmWave_syn_data)
    print('The shape of the syn RGB signal is {}, and has been successfully saved in ../data/'.format(np.shape(RGB_syn_data))
          + str(current_data_index) + '/RGB/RGB.npy')
    print('The shape of the syn Depth signal is {}, and has been successfully saved in ../data/'.format(
        np.shape(Depth_syn_data))
          + str(current_data_index) + '/Depth/Depth.npy')
    print('The shape of the syn FFT signal is {}, and has been successfully saved in ../data/'.format(np.shape(mmWave_syn_data))
          + str(current_data_index) + '/mmWave/mmWave.npy')
# ..................................................
