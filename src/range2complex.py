import numpy as np

B = 1798.92e6  # Sweep Bandwidth
sample_num = 64  # Sample Length
chirp_num = 255  # Chirp Total
channel_num = 4  # Channel Total
c = 3e8  # Speed of Light
f0 = 77e9  # Start Frequency
NumRangeFFT = 64  # Range FFT Length

for current_data_index in range(1, 11):  # indicate the current loaded data index
    num_per_act = 10  # indicate the data folder number of one action
    for index in range(1, num_per_act + 1):
        mmWave_syn_data = np.load('../data/' + str(current_data_index) + '/mmWave/mmWave' + str(index) + '.npy')
        print(
            'The syn mmWave data with the shape of {} has been successfully loaded from ../data/{}/mmWave/mmWave{}.npy'.format(
                np.shape(mmWave_syn_data), current_data_index, index)
        )
        mmWave_data = np.zeros((channel_num, len(mmWave_syn_data), chirp_num, sample_num), dtype=complex)
        for frame in range(len(mmWave_syn_data)):
            for channel in range(channel_num):
                for chirp in range(chirp_num):
                    sigRangeFFT = mmWave_syn_data[frame][chirp][channel * sample_num: (channel + 1) * sample_num]
                    sigRangeWin = np.fft.ifft(sigRangeFFT, NumRangeFFT)
                    sigReceive = np.divide(sigRangeWin, np.hamming(sample_num).T)
                    mmWave_data[channel, frame, chirp, :] = sigReceive
        np.save('../data/' + str(current_data_index) + '/mmWave/mmWave' + str(index) + '_raw_complex.npy', mmWave_data)
        print(
            'The syn mmWave data with the shape of {} has been successfully saved in ../data/{}/mmWave/mmWave{}_raw_complex.npy'.format(
                np.shape(mmWave_syn_data), current_data_index, index)
        )