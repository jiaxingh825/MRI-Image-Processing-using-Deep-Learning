import numpy as np
import matplotlib.pyplot as plt
import h5py
import common

def readSignal(fileName):
    f = h5py.File(fileName)
    volGroup = list(f["data"].keys())
    sigList = list(f["data"][volGroup[0]].keys())
    count = len(sigList)
    sig = np.empty((count,16,12800))
    noise = np.empty((count,2,12800))
    sigDic = {}
    for i in range(0, count-1):
        sig[i] = np.array(f["data"][volGroup[0]][sigList[i]])[:16,]
        sigDic[sigList[i]] = np.array(f["data"][volGroup[0]][sigList[i]])
        noise[i] = np.array(f["data"][volGroup[0]][sigList[i]])[-2:,]
    return sig,noise,sigDic
sqWave = 'C:/Users/Admin/Desktop/JiaxingData/sampleData_June4_2024/DataCapture_10kHzSqWave_n30dB_TR320_acq100_rep300.h5'
baseline = 'C:/Users/Admin/Desktop/JiaxingData/sampleData_June4_2024/DataCapture_Baseline_TR20_acq5_rep6000.h5'
sig2,noise2,sigDic2 = readSignal(sqWave)

signal = sig2[250,0,:]
fft = np.fft.fft(signal)
ps = 20*np.log10(np.abs(fft)**2)

sampling_frequency = common.getRate(sqWave)
freqs = np.fft.fftfreq(len(signal),1/sampling_frequency)

plt.plot(freqs[:len(freqs)//2], ps[:len(freqs)//2])
#plt.plot(freqs[:len(freqs)//2],np.abs(fft)[:len(freqs)//2])
plt.show()
