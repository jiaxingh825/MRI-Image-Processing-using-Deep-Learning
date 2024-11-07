import h5py
import numpy as np
import keras
import sys
import tensorflow.keras.layers as tfl
from keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt
import pandas as pd
#sys.path.append("E:/JiaxingData/ActiveNoiseSensingModel")
import DataProcessing as dp
sys.path.append("C:/code/mri/commonLib")
import common
from keras.metrics import MeanSquaredError
from keras.layers import ReLU
import matplotlib.pyplot as plt

def toKSpace(allData,fname, table_name="acq", datasrc="data"):
    fh = h5py.File(fname, "r")
    desc = common.getDataDescriptions(fh, table_name)
    fh.close()
    nSlices = int(desc["slice"].max()+1)
    ny = int(desc["phase"].max()+1)
    nx = allData.shape[2]
    nChannels = allData.shape[1]
 
    #allociate space for kSpace data
    kSpace = np.zeros((nx,ny,nSlices,nChannels),dtype=np.complex64)
    #loop through slices and phase encodes put data into kSpace array
    for sliceIndex in range(nSlices):
        sliceDesc = desc.loc[desc.slice==sliceIndex]
        for ind in sliceDesc.index:
            phaseIndex = int(sliceDesc['phase'][ind])
            kSpace[:,phaseIndex,sliceIndex,:]=np.transpose(allData[ind,:,:],[1,0])
    
    return kSpace

def toImg(kSpace):
    im = common.f2d(kSpace,axes=[0,1])
    #Simple Sum of Square Channel Combinations
    sos = np.sum(np.power(np.abs(im),2),axis=im.ndim-1)
    return sos

def ConvergeComplex(data):
    numPoints = int(data.shape[2])
    numChannels = int(data.shape[1])
    newData = np.zeros((data.shape[0],numChannels//2,numPoints),dtype=np.complex64)
    for i in range(0,(numChannels//2)):
        for k in range(0,numPoints):
            newData[:,i,k] = data[:,2*i,k]+data[:,2*i+1,k]*1j
    return newData

sample,d = common.readAllAcqs("E:/JiaxingData/EMINoise/0708/NoiseFlash2D_sagittal.h5")
#sampleNoise = dp.SplitComplex(sample[:,16:18,:])
sample = dp.SplitComplex(sample[:,:16,:])
sampleNoise = np.load('E:/JiaxingData/ActiveNoiseSensingModel/img/noiseTest.npy')
sampleMRI = np.load('E:/JiaxingData/ActiveNoiseSensingModel/img/sigTest.npy')

baseline,d = common.readAllAcqs("E:/JiaxingData/EMINoise/0708/BaselineFlash2D_1.h5")
baseline = dp.SplitComplex(baseline[:,:16,:])

sampleNoise = np.expand_dims(sampleNoise,3)
# load in model
Model = keras.models.load_model('E:/JiaxingData/ActiveNoiseSensingModel/img/Model_2_16_0715.h5',
                                custom_objects={'mse': MeanSquaredError(), 'ReLU': ReLU})
predicted = Model.predict(sampleNoise)
corrected = sampleMRI - predicted
# Calculate MAE
mse = np.mean(np.square(corrected - sampleMRI))
print("Mean square Error on sample Image:", mse)
corrected = ConvergeComplex(corrected)
sample = ConvergeComplex(sample)
baseline = ConvergeComplex(baseline)
sampleCorrected = np.zeros((1600,16,320),dtype=np.complex64)
sampleCorrected[0:1280] = sample[0:1280]
sampleCorrected[1280:1600] = corrected
CleanK = toKSpace(sampleCorrected,"E:/JiaxingData/EMINoise/0708/NoiseFlash2D_sagittal.h5")
NoiseK = toKSpace(sample,"E:/JiaxingData/EMINoise/0708/NoiseFlash2D_sagittal.h5")
baselineK = toKSpace(baseline,'E:/JiaxingData/EMINoise/0708/BaselineFlash2D_1.h5')
plt.imshow(np.abs(CleanK[:,:,1,1]))
CleanImage = toImg(CleanK)
NoiseImage = toImg(NoiseK)
baselineImage = toImg(baselineK)

#max= np.max(CleanImage)/5
#min=np.min(CleanImage)
#common.PlotSlices(CleanImage,min,max)
plt.imshow(np.abs(np.fft.ifftshift(np.fft.ifft2(CleanImage[:,:,9]))))#, cmap='gray', vmin=min, vmax=max)
plt.axis('off')
plt.show()
max= np.max(NoiseImage)/5
min=np.min(NoiseImage)
#common.PlotSlices(NoiseImage,min,max)
plt.imshow(np.abs(np.fft.ifftshift((np.fft.ifft2(NoiseImage[:,:,9])))))#, cmap='gray', vmin=min, vmax=max)
plt.axis('off')
plt.show()

max= np.max(baselineImage)/5
min=np.min(baselineImage)
#common.PlotSlices(baselineImage,min,max)
plt.imshow(np.abs((np.fft.ifftshift(np.fft.ifft2(baselineImage[:,:,9])))))#, cmap='gray', vmin=min, vmax=max)
plt.axis('off')
plt.show()


def plotFFTComparsion(corrected,sampleMRI,baseline):
    fft1 = np.fft.fft(corrected)
    fft2 = np.fft.fft(sampleMRI)
    fft3 = np.fft.fft(baseline)
    plt.plot(abs(fft3),"-y",label="baseline")
    plt.plot(abs(fft2),"-b",label="original")
    plt.plot(abs(fft1),"-r",label="clean")
    plt.legend(loc="upper right")
    plt.show()
    
#plotFFTComparsion(corrected[0,0],sampleMRI[0,0],baseline[1280,0])