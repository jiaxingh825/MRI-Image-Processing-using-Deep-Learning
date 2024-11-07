import h5py
import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
import common
import tensorflow as tf
import tensorflow.keras.layers as tfl
import json

# double the size of the dimension of channels by adding phase (angel) data
def AddPhase(data):
    numSamples = data.shape[0]
    numChannels = data.shape[1]
    newData = np.zeros((numSamples,2*numChannels,data.shape[2]))
    for i in range(0,numSamples):
        for j in range(0,numChannels):
            signal = data[i,j,:]
            phase = np.angle(np.fft.fft(signal))
            newData[i,2*j,:] = signal
            newData[i,2*j+1,:] = phase 
    return newData


# Prepare the data for model training
# the input files are noise with image recordings
# The output sig data contains baseline MRI channel
# The output noise channel contains all 18 channels from Noisy sample
# Before running this function, knock off ".view(np.complex64)" in common.readDataByUID
def directDSP(noise,baseline):
    baseline,a = common.readAllAcqs(baseline)
    baseline = baseline[:,0:16]
    sample,b = common.readAllAcqs(noise)
    l = sample.shape[0]
    l1 = int(l*0.6)
    l2 = int(l*0.8)
    
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0723/sigTrainingS.npy',SplitComplex(baseline[0:l1]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0723/noiseTrainingS.npy',SplitComplex(sample[0:l1]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0723/sigValS.npy',SplitComplex(sample[l1:l2]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0723/noiseValS.npy',SplitComplex(baseline[l1:l2]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0723/sigTestS.npy',SplitComplex(sample[l2:l]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0723/noiseTestS.npy',SplitComplex(baseline[l2:l]))
    


# read signal from h5py file(not the raw data)
# common.readALLacqs is a more consistent way of reading
# fileName file path and name
# l length of the data in x axis
# dataType usually "rawData" or "data" 
def readSignal(fileName,l,dataType = "data"):
    f = h5py.File(fileName)
    volGroup = list(f[dataType].keys())
    sigList = list(f[dataType][volGroup[0]].keys())
    count = len(sigList)
    sig = np.empty((count,16,l))
    noise = np.empty((count,2,l))
    #sigDic = {}
    for i in range(0, count-1):
        sig[i] = np.array(f[dataType][volGroup[0]][sigList[i]])[:16,]
        #sigDic[sigList[i]] = np.array(f[dataType][volGroup[0]][sigList[i]])
        noise[i] = np.array(f[dataType][volGroup[0]][sigList[i]])[-2:,]
    return sig,noise

# randomly segmenting the data set
# sig signal
# num size of segmented data
#return segmented data and remaining data
def dataSeg(sig,num):
    i = np.random.choice(sig.shape[0],num,replace=False)
    a = set(range(sig.shape[0]))
    r = list(a-set(i))
    seg = sig[i]
    rem = sig[r]
    return seg,rem



# double the size of channels and half the number of x-axis points to separate the real and imaginary part
def SplitComplex(data):
    numPoints = int(data.shape[2])
    numChannels = data.shape[1]
    newData = np.zeros((data.shape[0],2*numChannels,numPoints//2))
    for i in range(numChannels):
        for j in range(numPoints//2):
            newData[:,2*i,j] = data[:,i,2*j]
            newData[:,2*i+1,j] = data[:,i,2*j+1]
    return newData


# Prepare the data for model training
# the input files are noise only recordings, no MRI data in the MRI channel as well
def processSignalFile(noise):
    sig1,noise1 = readSignal(noise,640)
    #sig2,noise2 = readSignal(baseline,640)

    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0715/sigTraining'+str(i)+'.npy',SplitComplex(sig1[0:4500]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0715/noiseTraining'+str(i)+'.npy',SplitComplex(noise1[0:4500]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0715/sigVal'+str(i)+'.npy',SplitComplex(sig1[4500:5250]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0715/noiseVal'+str(i)+'.npy',SplitComplex(noise1[4500:5250]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0715/sigTest'+str(i)+'.npy',SplitComplex(sig1[5250:6000]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0715/noiseTest'+str(i)+'.npy',SplitComplex(noise1[5250:6000]))


# Prepare the data for model training
# the input files are noise with image recordings
# Before running this function, knock off ".view(np.complex64)" in common.readDataByUID
def processImgFile(noise,baseline):
    baseline,a = common.readAllAcqs(baseline)
    sample,b = common.readAllAcqs(noise)
    sig1 = sample[:,:16,:]-baseline[:,:16,:]
    noise1 = sample[:,16:18,:]
    l = sig1.shape[0]
    l1 = int(l*0.6)
    l2 = int(l*0.8)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0723/sigTraining.npy',SplitComplex(sig1[0:l1]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0723/noiseTraining.npy',SplitComplex(noise1[0:l1]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0723/sigVal.npy',SplitComplex(sig1[l1:l2]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0723/noiseVal.npy',SplitComplex(noise1[l1:l2]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0723/sigTest.npy',SplitComplex(sig1[l2:l]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/0723/noiseTest.npy',SplitComplex(noise1[l2:l]))