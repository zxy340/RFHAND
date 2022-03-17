import numpy as np

sample_num = 256  # Sample Length
chirp_num = 128  # Chirp Total
channel_num = 4  # Channel Total
LightVelocity = 3e8  # Speed of Light
FreqStart = 60e9  # Start Frequency
slope = 60.012e12  # Frequency slope
SampleRate = 4400e3  # Sample rate
NumRangeFFT = 256  # Range FFT Length
FrameTime = 0.1  # the period of one frame
Bandwidth = slope * sample_num / SampleRate  # Sweep Bandwidth
ChirpTime = FrameTime / chirp_num  # the internal of the chirp
WaveLength = LightVelocity / FreqStart  # Wave length
numTxAntennas = 3  # the number of the Tx Antenna
numRxAntennas = 4  # the number of the Rx Antenna
distance = 0.36  # approximate distance between hand and radar
AntennaDistance = 0.5 * WaveLength

dres = LightVelocity / (2 * Bandwidth)
print('the range resolution is {}'.format(dres))
dmax = SampleRate * LightVelocity / (2 * slope)
print('the max range is {}'.format(dmax))
vres = WaveLength / (2 * FrameTime)
print('the velocity resolution is {}'.format(vres))
vmax = WaveLength / (4 * ChirpTime)
print('the max velocity is {}'.format(vmax))
angleResmax = 180 * (WaveLength / (numRxAntennas * AntennaDistance)) / np.pi
print('the angle max resolution is {}'.format(angleResmax))