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
from keras.layers import ReLU
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError


#Algorithm for learning rate deacy
def lrDeacy(lr,epoch):
    return tf.keras.callbacks.LearningRateScheduler(lr*0.9**(epoch//4))

# learning rate uodate callback
def learningRateUpdate():
    return tf.keras.callbacks.LearningRateScheduler(lrDeacy)


# early stopping callback, stop the training if the validation loss stop redcuing
ESC = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   # Metric to be monitored
    patience=5,           # Number of epochs to wait for improvement
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored metric
)

def plotHist(name, hist,date,modelName):
    hist = hist.history[name]
    epochs = range(1,len(hist)+1)

    plt.plot(epochs,hist)
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.savefig(f'E:/JiaxingData/ActiveNoiseSensingModel/ImgResults/'+date+'/'+modelName+name+'.png')
    plt.show()




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
def plotFFTComparsion(corrected,sampleMRI,baseline,date):
    fig,axs = plt.subplots(4,3)
    axs = axs.flatten()
    for i in range(16):
        c= corrected[:,i].flatten()
        s = sampleMRI[:,i].flatten()
        b = baseline[:,i].flatten()
        fft1 = np.fft.fft(c)
        fft2 = np.fft.fft(s)
        fft3 = np.fft.fft(b)
        if norm == 0:
            axs[i].plot(abs(fft2),"-b",label="original")
            axs[i].plot(abs(fft1),"-r",label="clean")
            axs[i].plot(abs(fft3),"-g",label="baseline")
            axs[i].legend(loc="upper right")
            axs[i].title(f"Channel {i}")
            axs[i].show()
    plt.savefig(f'E:/JiaxingData/ActiveNoiseSensingModel/ImgResults/'+date+'/FFT_Comparsions.png')
            
#plot the test samples in image and k space
def plotSamples(CleanImage,NoiseImage,baselineImage,date):
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
    #plt.savefig("E:/JiaxingData/ActiveNoiseSensingModel/ImgResults/'+date+'/object/IMG/sample.png")
    plt.show()

def storePrediction(fPath, fName, corrected):

    # use the shutil.copyobj() method to copy the contents of source_file to destination_file
    shutil.copy((fPath+fName+'.h5'), fPath+fName+'New.h5')
    f1 = h5py.File(fPath+fName+'New.h5', 'r+')     # open the file
    desc = common.getDataDescriptions(f1, table_name='acq')
    volGroup = list(f1['data'].keys())
    for i, row in desc.iterrows():
        data_string = "{:016x}".format(int(row.uid))
        data = f1['data'][volGroup[0]][data_string]       # load the data
        data[0:16] = corrected[i]                      # assign new values to data
    f1.close()

def plotAll(BaselineImage,NoiseImage,CleanImage,NoiseMap):
    max= np.max(BaselineImage)/1
    min=np.min(BaselineImage)
    common.PlotSlices(BaselineImage,min,max)
    max= np.max(NoiseImage)/5000
    min=np.min(NoiseImage)
    common.PlotSlices(NoiseImage,min,max)
    max= np.max(CleanImage)/1
    min=np.min(CleanImage)
    common.PlotSlices(CleanImage,min,max)
    max= np.max(NoiseMap)/500
    min=np.min(NoiseMap)
    common.PlotSlices(NoiseMap,min,max)