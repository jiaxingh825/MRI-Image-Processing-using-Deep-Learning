import h5py
import numpy as np
import keras
import tensorflow.keras.layers as tfl
from keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt
import pandas as pd
import DataProcessing as dp
import common
import TrainAndTest as tt

def readSignal(fileName,l,dataType = "data"):
    f = h5py.File(fileName)
    volGroup = list(f[dataType].keys())
    sigList = list(f[dataType][volGroup[0]].keys())
    count = len(sigList)
    sig = np.array(f[dataType][volGroup[0]][sigList[0]])[:16,]
    noise = np.array(f[dataType][volGroup[0]][sigList[0]])[-2:,]

    for i in range(1, count-1):
        np.append(sig,np.array(f[dataType][volGroup[0]][sigList[i]])[:16,],axis=1)
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
    transfer = np.zeros((16,2))#,dtype= np.complex64)
    transfer = np.dot(mri,np.linalg.pinv(noise))
    return transfer

def transferFlattenSingle(noiseSignal, MRISignal):
    mriSize = MRISignal.shape
    noiseSize = noiseSignal.shape
    mri = MRISignal.transpose(1, 0, 2).reshape(mriSize[1],mriSize[0]*mriSize[2])
    noise = noiseSignal.transpose(1, 0, 2).reshape(noiseSize[1],noiseSize[0]*noiseSize[2])
    transfer = np.zeros((16,2))#,dtype= np.complex64)
    transfer = np.dot(mri,np.linalg.pinv(noise))
    return transfer

noise = 'C:/Active_Noise_Sensing/EMINoise/0827/NoiseSignal_32_256.h5'
img = 'C:/Active_Noise_Sensing/EMINoise/0827/NoisyImg.h5'
baseline = 'C:/Active_Noise_Sensing/EMINoise/0827/BaselineImg.h5'

# Load Data
# noise channel from noise only scans
signal,b = common.readAllAcqs(noise)
noiseSignal = signal[:,16:18,:]

# MRI channel from noise only scans
MRISignal = signal[:,0:16,:]



# Noise channel from  Noisy MRI scans
noisyImg,b = common.readAllAcqs(img)
imageNoise = noisyImg[:,16:18,:]
# Noisy Image, image channel from MRI scans
image = noisyImg[:,0:16,:]
# baseline Image
baseline,b = common.readAllAcqs(baseline)
baseline = baseline[:,0:16,:]
#(16,2)*(2,512) = (16,512)
# (16,512)*(512,2) = (16,2)

noiseSignal1 = imageNoise
MRISignal1 = image-baseline

noiseSignal2 = np.append(noiseSignal,noiseSignal1,axis = 0)
MRISignal2 = np.append(MRISignal,MRISignal1,axis = 0)

#transfer = transferFlatten(noiseSignal, MRISignal)
print(noiseSignal1.shape)
transfer1 = transferFlatten(noiseSignal1, MRISignal1)
#transfer2 = transferFlatten(noiseSignal2, MRISignal2)
#scale = transfer1/transfer2
#print(scale)

pred = np.zeros(image.shape)#,dtype= np.complex64)
len = image.shape[0]
for k in range(len):
    pred[k] = np.dot(transfer1,imageNoise[k])

cleaned = image - pred

mse = np.mean(np.square(cleaned - baseline))
print("Mean square Error on sample Image:", mse)

BaselineImage = common.toImg(common.toKSpace(baseline,img))
NoiseImage = common.toImg(common.toKSpace(image,img))
CleanImage = common.toImg(common.toKSpace(cleaned,img))
NoiseMap = common.toImg(common.toKSpace(pred,img))

tt.storePrediction('C:/Active_Noise_Sensing/EMINoise/0827/','NoisyImg',cleaned)


#tt.plotAll(BaselineImage,NoiseImage,CleanImage,NoiseMap)
