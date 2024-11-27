import numpy as np
import matplotlib.pyplot as plt
import h5py
import common
import DataProcessing as dp

date = '0816'
noiseOnly,b = common.readAllAcqs('E:/JiaxingData/EMINoise/0816/NoiseSignal.h5')
noiseOnly = dp.ConvergeComplexR(dp.SplitComplexR(noiseOnly))
noiseImage,b = common.readAllAcqs('E:/JiaxingData/EMINoise/0816/NoiseImage.h5')
noiseImage = dp.ConvergeComplexR(dp.SplitComplexR(noiseImage))
baseline,b = common.readAllAcqs('E:/JiaxingData/EMINoise/0816/BaselineImage.h5')
baseline = dp.ConvergeComplexR(dp.SplitComplexR(baseline))

i1 = 0
i2 = 17

noiseOnly = noiseOnly[i1,i2]
noise = noiseImage[i1,i2]
noiseImage = noiseImage[i1,i2] - baseline[i1,i2]
print(noiseOnly.shape)
print(noiseImage.shape)


fft1 = np.fft.fft(noiseOnly)
fft2 = np.fft.fft(noise)
ps1 = 20*np.log10(np.abs(fft1)**2)
ps2 = 20*np.log10(np.abs(fft2)**2)

sampling_frequency1 = common.getRate('E:/JiaxingData/EMINoise/0816/NoiseSignal.h5')
sampling_frequency2 = common.getRate('E:/JiaxingData/EMINoise/0816/NoiseImage.h5')
freqs1 = np.fft.fftfreq(len(noiseOnly),1/sampling_frequency1)
freqs2 = np.fft.fftfreq(len(noiseOnly),1/sampling_frequency2)

plt.plot(freqs1[:len(freqs1)//2],np.abs(fft1)[:len(freqs1)//2])
plt.plot(freqs2[:len(freqs2)//2],np.abs(fft2)[:len(freqs2)//2])
plt.legend(['noise only','noise map'])
plt.savefig('C:/Users/Admin/Desktop/signalExamination/'+date+'/abs(fft)_'+str(i1)+'_'+str(i2)+'.png')
#plt.plot(freqs[:len(freqs)//2],np.abs(fft)[:len(freqs)//2])
plt.show()
