import h5py
import numpy as np
import keras
import tensorflow.keras.layers as tfl
from keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt
import pandas as pd
import DataProcessing as dp
import common

def readSignal(fileName,l,dataType = "data"):
    f = h5py.File(fileName)
    volGroup = list(f[dataType].keys())
    sigList = list(f[dataType][volGroup[0]].keys())
    count = len(sigList)
    sig = []
    noise = []
    
    #sigDic = {}
    for i in range(0, count-1):
        np.append(sig,np.array(f[dataType][volGroup[0]][sigList[i]])[:16,],axis=1)
        #sigDic[sigList[i]] = np.array(f[dataType][volGroup[0]][sigList[i]])
        np.append(noise,np.array(f[dataType][volGroup[0]][sigList[i]])[-2:,],axis=1)
    return sig,noise

def transferAveraging(noiseSignal, MRISignal):
    transfer = np.zeros((16,2),dtype= np.complex64)
    sum  = np.zeros((16,2),dtype= np.complex64)
    l = noiseSignal.shape[0]

    for i in range(l):
        transfer = np.dot(MRISignal[i],np.linalg.pinv(noiseSignal[i]))
        sum += transfer

    return sum/l

def transferFlatten(noiseSignal, MRISignal):
    mriSize = MRISignal.shape
    noiseSize = noiseSignal.shape
    mri = MRISignal.transpose(1, 0, 2).reshape(mriSize[1],mriSize[0]*mriSize[2])
    noise = noiseSignal.transpose(1, 0, 2).reshape(noiseSize[1],noiseSize[0]*noiseSize[2])
    transfer = np.zeros((16,2),dtype= np.complex64)
    transfer = np.dot(mri,np.linalg.pinv(noise))
    return transfer

# Load Data
# noise channel from noise only scans
signal,b = common.readAllAcqs('E:/JiaxingData/EMINoise/0827/NoiseSignal_32_256.h5')
noiseSignal = signal[:,16:18,:]

# MRI channel from noise only scans
MRISignal = signal[:,0:16,:]

signal2,b = common.readAllAcqs('E:/JiaxingData/EMINoise/0827/0Flip.h5')
noiseSignal2 = signal2[:,16:18,:]

# MRI channel from noise only scans
MRISignal2 = signal2[:,0:16,:]


# Noise channel from  Noisy MRI scans
noisyImg,b = common.readAllAcqs('E:/JiaxingData/EMINoise/0827/NoisyImg.h5')
imageNoise = noisyImg[:,16:18,:]
# Noisy Image, image channel from MRI scans
image = noisyImg[:,0:16,:]
# baseline Image
baseline,b = common.readAllAcqs('E:/JiaxingData/EMINoise/0827/BaselineImg.h5')
baseline = baseline[:,0:16,:]
#(16,2)*(2,512) = (16,512)
# (16,512)*(512,2) = (16,2)

noiseSignal1 = imageNoise
MRISignal1 = image-baseline

transfer = transferFlatten(noiseSignal, MRISignal)
transfer1 = transferFlatten(noiseSignal1, MRISignal1)
transfer2 = transferFlatten(noiseSignal2, MRISignal2)
scale = transfer1/transfer2
print(scale)

pred = np.zeros(image.shape,dtype= np.complex64)
len = image.shape[0]
for k in range(len):
    pred[k] = np.dot(transfer,imageNoise[k])

cleaned = image - pred

mse = np.mean(np.square(cleaned - baseline))
print("Mean square Error on sample Image:", mse)

BaselineImage = common.toImg(common.toKSpace(baseline,'E:/JiaxingData/EMINoise/0827/BaselineImg.h5'))
NoiseImage = common.toImg(common.toKSpace(image,'E:/JiaxingData/EMINoise/0827/NoisyImg.h5'))
CleanImage = common.toImg(common.toKSpace(cleaned,'E:/JiaxingData/EMINoise/0827/NoisyImg.h5'))
NoiseMap = common.toImg(common.toKSpace(pred,'E:/JiaxingData/EMINoise/0827/NoisyImg.h5'))

max= np.max(BaselineImage)/2
min=np.min(BaselineImage)
common.PlotSlices(BaselineImage,min,max)
max= np.max(NoiseImage)/5000
min=np.min(NoiseImage)
common.PlotSlices(NoiseImage,min,max)
max= np.max(CleanImage)/20
min=np.min(CleanImage)
common.PlotSlices(CleanImage,min,max)
max= np.max(NoiseMap)/500
min=np.min(NoiseMap)
common.PlotSlices(NoiseMap,min,max)