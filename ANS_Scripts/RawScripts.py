import h5py
import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
import DataProcessing as dp
import common
import TrainAndTest as tt

flip0,b = common.readAllAcqs('E:/JiaxingData/EMINoise/1209/AMSquareFA0_1.h5')
flip0 = dp.ConvergeComplexR(dp.SplitComplexR(flip0))
sampling_frequency = common.getRate('E:/JiaxingData/EMINoise/1209/AMSquareFA0_1.h5')
freqs = np.fft.fftfreq(len(flip0),1/sampling_frequency)

plt.plot(np.real(flip0[0,0]),'lightblue')
plt.plot(np.real(flip0[0,1]),'g')
plt.plot(np.real(flip0[0,2]),'g')
plt.plot(np.real(flip0[0,3]),'lightblue')
plt.plot(np.real(flip0[0,4]),'blue')
plt.plot(np.real(flip0[0,5]),'orange')
plt.plot(np.real(flip0[0,6]),'lightblue')
plt.plot(np.real(flip0[0,7]),'orange')
plt.plot(np.real(flip0[0,8]),'blue')
plt.plot(np.real(flip0[0,9]),'lightblue')
plt.plot(np.real(flip0[0,10]),'lightblue')
plt.plot(np.real(flip0[0,11]),'orange')
plt.plot(np.real(flip0[0,12]),'g')
plt.plot(np.real(flip0[0,13]),'lightblue')
plt.plot(np.real(flip0[0,14]),'g')
plt.plot(np.real(flip0[0,15]),'orange')
plt.plot(np.real(flip0[0,16]),'red')
plt.plot(np.real(flip0[0,17]),'pink')
#plt.xlim(9500,10500)
#plt.ylim(0,45000)
plt.grid()
plt.legend(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17'])
    #plt.savefig('E:/JiaxingData/EMINoise/'+date+'/spectrumPlot/'+mode+type+trial+'_absFFT_'+str(i1)+'_'+str(i2)+'.png')
    #plt.plot(freqs[:len(freqs)//2],np.abs(fft)[:len(freqs)//2])
plt.show()

