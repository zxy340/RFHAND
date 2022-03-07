import numpy as np

Bandwidth = 3657.8e6  # Sweep Bandwidth
sample_num = 64  # Sample Length
chirp_num = 255  # Chirp Total
channel_num = 4  # Channel Total
LightVelocity = 3e8  # Speed of Light
FreqStart = 77e9  # Start Frequency
slope = 29.982e12  # Frequency slope
SampleRate = 2099e3  # Sample rate
NumRangeFFT = 64  # Range FFT Length
FrameTime = 0.034  # the period of one frame
ChirpTime = FrameTime / chirp_num  # the internal of the chirp
WaveLength = LightVelocity / FreqStart  # Wave length
numTxAntennas = 1  # the number of the Tx Antenna
numRxAntennas = 4  # the number of the Rx Antenna
distance = 0.8  # approximate distance between hand and radar

dres = LightVelocity / (2 * Bandwidth)
print('the range resolution is {}'.format(dres))
dmax = SampleRate * LightVelocity / (2 * slope)
print('the max range is {}'.format(dmax))
vres = WaveLength / (2 * FrameTime)
print('the velocity resolution is {}'.format(vres))
vmax = WaveLength / (4 * ChirpTime)
print('the max velocity is {}'.format(vmax))
angleResmax = 180 * (2 / numRxAntennas) / np.pi
print('the angle max resolution is {}'.format(angleResmax))