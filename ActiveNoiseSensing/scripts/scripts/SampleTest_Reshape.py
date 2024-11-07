import h5py
import numpy as np
import keras
import tensorflow.keras.layers as tfl
from keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt
import pandas as pd
import DataProcessing as dp
import common
import shutil
from keras.metrics import MeanSquaredError
from keras.layers import ReLU
import matplotlib.pyplot as plt

# Convert data into 4-D K space
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

# Convert 4D K Space into image space and combine 16 channels
def toImg(kSpace):
    im = common.f2d(kSpace,axes=[0,1])
    #Simple Sum of Square Channel Combinations
    sos = np.sum(np.power(np.abs(im),2),axis=im.ndim-1)
    return sos

# Reverse the process of splitComplex
# Reconstruct the data by put the coressponding imaginary part next to the real part
def ConvergeComplex(data):
    numPoints = int(data.shape[2])
    numChannels = int(data.shape[1])
    newData = np.zeros((data.shape[0],numChannels//2,numPoints),dtype=np.complex64)
    for i in range(0,(numChannels//2)):
        for k in range(0,numPoints):
            newData[:,i,k] = data[:,2*i,k]+data[:,2*i+1,k]*1j
    return newData

def ConvergeComplexR(data):
    numPoints = int(data.shape[2])
    numChannels = int(data.shape[3])
    newData = np.zeros((data.shape[0],numPoints,numChannels),dtype=np.complex64)
    for k in range(0,numPoints):
        for i in range(numChannels):
            newData[:,k,i] = data[:,0,k,i]+data[:,1,k,i]*1j
    return np.transpose(newData,(0,2,1))

# Return the max-min normalization of the data
def normalized(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

# plot the data in the frequency domain 
def plotFFTComparsion(corrected,sampleMRI,baseline,norm = 0):
    for i in range(16):
        c= corrected[:,i].flatten()
        s = sampleMRI[:,i].flatten()
        b = baseline[:,i].flatten()
        fft1 = np.fft.fft(c)
        fft2 = np.fft.fft(s)
        fft3 = np.fft.fft(b)
        if norm == 0:
            plt.plot(abs(fft2),"-b",label="original")
            plt.plot(abs(fft1),"-r",label="clean")
            plt.plot(abs(fft3),"-g",label="baseline")
            plt.legend(loc="upper right")
            plt.title(f"Channel {i}")
            #plt.savefig(f"E:/JiaxingData/ActiveNoiseSensingModel/ImgResults/0731/object_32_128/FFT/Channel_{i}.png")
            plt.show()
        else:
            fft1 = normalized(fft1)
            fft2 = normalized(fft2)
            fft3 = normalized(fft3)
            plt.plot(abs(fft2),"-b",label="original")
            plt.plot(abs(fft1),"-r",label="clean")
            plt.plot(abs(fft3),"-g",label="baseline")
            plt.legend(loc="upper right")
            plt.title(f"Channel {i} normalized")
            #plt.savefig(f"C:/Users/Admin/Desktop/object0708_FFT/Channel_{i}_Normalizaed.png")
            plt.show()
            
#plot the test samples in image and k space
def plotSamples(CleanImage,NoiseImage,baselineImage):
    fig,axs = plt.subplots(4,3)
    axs = axs.flatten()
    max= np.max(NoiseImage)/50
    min=np.min(NoiseImage)
    axs[0].imshow(NoiseImage[:,:,4], cmap='gray', vmin=min, vmax=max)
    axs[0].set_title('Noisy Image sample 1')
    max= np.max(CleanImage)/1
    min=np.min(CleanImage)
    axs[1].imshow(CleanImage[:,:,4], cmap='gray', vmin=min, vmax=max)
    axs[1].set_title('cleaned Image sample 1')
    max= np.max(baselineImage)/1
    min=np.min(baselineImage)
    axs[2].imshow(baselineImage[:,:,4], cmap='gray', vmin=min, vmax=max)
    axs[2].set_title('Baseline Image sample 1')
    axs[3].imshow(np.abs(np.fft.ifftshift(np.fft.ifft2(NoiseImage[:,:,4]))))
    axs[3].set_title('Noisy K Space sample 1')
    axs[4].imshow(np.abs(np.fft.ifftshift(np.fft.ifft2(CleanImage[:,:,4]))))
    axs[4].set_title('cleaned K Space sample 1')
    axs[5].imshow(np.abs(np.fft.ifftshift(np.fft.ifft2(baselineImage[:,:,4]))))
    axs[5].set_title('Baseline K Space sample 1')
    max= np.max(NoiseImage)/50
    min=np.min(NoiseImage)
    axs[6].imshow(NoiseImage[:,:,9], cmap='gray', vmin=min, vmax=max)
    axs[6].set_title('Noisy Image sample 2')
    max= np.max(CleanImage)/3
    min=np.min(CleanImage)
    axs[7].imshow(CleanImage[:,:,9], cmap='gray', vmin=min, vmax=max)
    axs[7].set_title('cleaned Image sample 2')
    max= np.max(baselineImage)/3
    min=np.min(baselineImage)
    axs[8].imshow(baselineImage[:,:,9], cmap='gray', vmin=min, vmax=max)
    axs[8].set_title('Baseline Image sample 2')
    axs[9].imshow(np.abs(np.fft.ifftshift(np.fft.ifft2(NoiseImage[:,:,9]))))
    axs[9].set_title('Noisy K Space sample 2')
    axs[10].imshow(np.abs(np.fft.ifftshift(np.fft.ifft2(CleanImage[:,:,9]))))
    axs[10].set_title('cleaned K Space sample 2')
    axs[11].imshow(np.abs(np.fft.ifftshift(np.fft.ifft2(baselineImage[:,:,9]))))
    axs[11].set_title('Baseline K Space sample 2')
    plt.axis('off')
    #plt.savefig("E:/JiaxingData/ActiveNoiseSensingModel/ImgResults/0716/object/IMG/sample.png")
    plt.show()

def storePrediction(fname,corrected):
    destination_file = 'E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0816'

    # use the shutil.copyobj() method to copy the contents of source_file to destination_file
    shutil.copy(fname, destination_file)
    f1 = h5py.File(fname, 'r+')     # open the file
    desc = common.getDataDescriptions(f1, table_name='acq')
    for i, row in desc.iterrows():
        data_string = "{:016x}".format(int(row.uid))
        data = f1['data'+data_string]       # load the data
        data[...] = corrected[i]                      # assign new values to data
    f1.close() 

# Load original sample data !!!Change File Paths !!!
sample,d = common.readAllAcqs('E:/JiaxingData/EMINoise/0731/sub/run2.h5')
sample = dp.SplitComplex(sample[:,:16,:])
sample = ConvergeComplex(sample)
# Load testing data !!!Change File Paths !!!
sampleNoise = np.load('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0731/noiseTest0731S.npy')
sampleMRI = np.load('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0731/sigTest0731S.npy')
# Load baseline data !!!Change File Paths !!!
baseline,d = common.readAllAcqs('E:/JiaxingData/EMINoise/0731/sub/baseline.h5')
baseline = dp.SplitComplex(baseline[:,:16,:])
length = baseline.shape[0]
subLength = int(length*0.8)
nPoints = baseline.shape[2]

# load in model and calculate the predicition !!!Change File Paths !!!
Model = keras.models.load_model('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0731/Model_0731_16_S.h5',
                                custom_objects={'mse': MeanSquaredError(), 'ReLU': ReLU})
predicted = Model.predict(sampleNoise)
corrected = sampleMRI - predicted
corrected = ConvergeComplexR(corrected)
baseline = ConvergeComplex(baseline)
sampleMRI = ConvergeComplexR(sampleMRI)
#mse = np.mean(np.square(corrected - baseline))#[subLength:length]))
#print("Mean square Error on sample Image:", mse)
# Initialize array for cleaned data
#sampleCorrected = np.zeros((length,16,nPoints),dtype=np.complex64)
#sampleCorrected[0:subLength] = sample[0:subLength]
#sampleCorrected[subLength:length] = corrected

# Convert data into 3D k-space and image space !!!Change File Paths !!!
NoiseMapK = toKSpace(ConvergeComplexR(predicted),'E:/JiaxingData/EMINoise/0731/sub/run2.h5')
CleanK = toKSpace(corrected,'E:/JiaxingData/EMINoise/0731/sub/run2.h5')
NoiseK = toKSpace(sampleMRI,'E:/JiaxingData/EMINoise/0731/sub/run2.h5')
BaselineK = toKSpace(baseline,'E:/JiaxingData/EMINoise/0731/sub/run2.h5')
NoiseMap = toImg(NoiseMapK)
CleanImage = toImg(CleanK)
print(CleanImage.shape)
NoiseImage = toImg(NoiseK)
print(NoiseImage.shape)
BaselineImage = toImg(BaselineK)
BaselineImage = BaselineImage[:,:,1]



max= np.max(BaselineImage)/2.5
min=np.min(BaselineImage)
common.PlotSlices(BaselineImage,min,max)


max= np.max(NoiseImage)/500
min=np.min(NoiseImage)
#common.PlotSlices(NoiseImage,min,max)
max= np.max(CleanImage)/200
min=np.min(CleanImage)
#common.PlotSlices(CleanImage,min,max)
max= np.max(NoiseMap)/500
min=np.min(NoiseMap)
#common.PlotSlices(NoiseMap,min,max)
#plotSamples(CleanImage,NoiseImage,BaselineImage)
#plotFFTComparsion(corrected,sampleMRI,baseline)