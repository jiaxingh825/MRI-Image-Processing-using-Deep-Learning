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

# Calculate the transfer function by averging the transfer function of each acquision
# Inputs: 
#       noiseSignal: array containing the data from noise channels
#       MRISignal: array containing the data from head coil channels
# Outputs:
#       sum/l: the averaged transfer function 
def transferAveraging(noiseSignal, MRISignal):
    transfer = np.zeros((16,2),dtype= np.complex64)
    sum  = np.zeros((16,2),dtype= np.complex64)
    l = noiseSignal.shape[0]

    for i in range(l):
        transfer = np.dot(MRISignal[i],np.linalg.pinv(noiseSignal[i]))
        sum += transfer

    return sum/l

# Calculate the transfer function by Vectorization of the data set
# Inputs: 
#       noiseSignal: array containing the data from noise channels
#       MRISignal: array containing the data from head coil channels
# Outputs:
#       transfer: the overall transfer function 
def transferFlatten(noiseSignal, MRISignal,noisePreScan = None):
    mriSize = MRISignal.shape
    noiseSize = noiseSignal.shape
    mri = MRISignal.transpose(1, 0, 2).reshape(mriSize[1],mriSize[0]*mriSize[2])
    noise = noiseSignal.transpose(1, 0, 2).reshape(noiseSize[1],noiseSize[0]*noiseSize[2])
    if noisePreScan is not None:
        noisePreSize = noisePreScan.shape
        noisePreScan = noisePreScan.reshape(noisePreSize[1],noisePreSize[0]*noisePreSize[2])
        print(noisePreScan.shape)
        mri = np.concatenate((mri,noisePreScan[0:16]),axis=1)
        noise = np.concatenate((noise,noisePreScan[16:18]),axis=1)
    transfer = np.zeros((16,2),dtype= np.complex64)
    transfer = np.dot(mri,np.linalg.pinv(noise))
    return transfer



# Define file path
date = '0827'
fPath = 'E:/JiaxingData/EMINoise/'+date+'/'
noiseName = 'NoiseSignal_32_256'
imgName = 'NoisyImg'
baselineName = 'BaselineImg'
noise = fPath+noiseName+'.h5'
img = fPath+imgName+'.h5'
baseline = fPath+baselineName+'.h5'


noisePreScan,b = common.readAllAcqs(img,table_name='noise')
noisePreScan = dp.ConvergeComplexR(dp.SplitComplexR(noisePreScan))
## Load Data 
# noise channel from noise only scans
signal,b = common.readAllAcqs(noise)
noiseSignal = dp.ConvergeComplexR(dp.SplitComplexR(signal[:,16:18,:]))

# MRI channel from noise only scans
MRISignal = dp.ConvergeComplexR(dp.SplitComplexR(signal[:,0:16,:]))

# Noise channel from  Noisy MRI scans
noisyImg,b = common.readAllAcqs(img)
print(noisyImg.shape)
imageNoise = dp.ConvergeComplexR(dp.SplitComplexR(noisyImg[:,16:18,:]))
# Noisy Image, image channel from MRI scans
image = dp.ConvergeComplexR(dp.SplitComplexR(noisyImg[:,0:16,:]))
# baseline Image
baseline,b = common.readAllAcqs(baseline)
baseline = baseline[:,0:16,:]
baseline = dp.ConvergeComplexR(dp.SplitComplexR(baseline))
#(16,2)*(2,512) = (16,512)
#(16,512)*(512,2) = (16,2)

noiseSignal1 = imageNoise
print(noiseSignal1.shape)
MRISignal1 = image-baseline

transfer1 = transferFlatten(noiseSignal1, MRISignal1)
print(transfer1)
pred = np.zeros(image.shape,dtype= np.complex64)
len = image.shape[0]
for k in range(len):
    pred[k] = np.dot(transfer1,imageNoise[k])

cleaned = image - pred

mse = np.mean(np.square(cleaned - baseline))
print("Mean square Error on sample Image:", mse)
BaselineImage = dp.toImg(dp.toKSpace(baseline,img))
NoiseImage = dp.toImg(dp.toKSpace(image,img))
CleanImage = dp.toImg(dp.toKSpace(cleaned,img))
NoiseMap = dp.toImg(dp.toKSpace(pred,img))

print(dp.complexRearrangement(cleaned).shape)
tt.storePrediction(fPath,'NoisyImg',dp.complexRearrangement(cleaned))
tt.storePrediction18To16(fPath, 'NoisyImg')


tt.plotAll(BaselineImage,NoiseImage,CleanImage,NoiseMap)
