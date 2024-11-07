import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import common
import os

def get_filepaths(folder_path):
    filepaths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            filepaths.append(filepath)
    return filepaths

def SplitComplex(data):
    numPoints = int(data.shape[2])
    numChannels = data.shape[1]
    newData = np.zeros((data.shape[0],2*numChannels,numPoints//2))
    for i in range(numChannels):
        for j in range(numPoints//2):
            newData[:,2*i,j] = data[:,i,2*j]
            newData[:,2*i+1,j] = data[:,i,2*j+1]
    return newData


def SplitComplexR(data):
    numPoints = int(data.shape[2])
    numChannels = data.shape[3]
    newData = np.zeros((data.shape[0],2,numPoints//2,numChannels))
    for j in range(numPoints//2):
        newData[:,0,j] = data[:,0,2*j]
        newData[:,1,j] = data[:,0,2*j+1]
    return newData

# double the size of channels and half the number of x-axis points to separate the real and imaginary part
# Input data is a 4-D K-Space data
def SplitComplex3D(data):
    numPoints = int(data.shape[3])
    numChannels = data.shape[2]
    newData = np.zeros((data.shape[0],data.shape[1],2*numChannels,numPoints//2))
    for i in range(0,numChannels):
        for j in range(0,numPoints//2):
            newData[:,:,2*i,j] = data[:,:,i,2*j]
            newData[:,:,2*i+1,j] = data[:,:,i,2*j+1]
    return newData



def processFullMRIFileR(noise,baseline,date,testNoise=None):
    baseline,a = common.readAllAcqs(baseline)
    sample,b = common.readAllAcqs(noise)
    sig1 = sample[:,:16,:]-baseline[:,:16,:]
    noise1 = sample[:,16:18,:]
    sig1 = np.expand_dims(sig1,3)
    sig1 = np.transpose(sig1,(0,3,2,1))
    noise1 = np.expand_dims(noise1,3)
    noise1 = np.transpose(noise1,(0,3,2,1))
    l = sig1.shape[0]
    l1 = int(l*0.8)
    sigTrain = SplitComplexR(sig1[0:l1])
    noiseTrain = SplitComplexR(noise1[0:l1])
    sigVal = SplitComplexR(sig1[l1:l])
    noiseVal = SplitComplexR(noise1[l1:l])
    sigTest = SplitComplexR(sig1)
    noiseTest = SplitComplexR(noise1)
    #np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sig0731S.npy',sigTest)
    #np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noise0731S.npy',noiseTest)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigTraining.npy',sigTrain)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseTraining.npy',noiseTrain)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigVal.npy',sigVal)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseVal.npy',noiseVal)
    if testNoise != None :
        test,c = common.readAllAcqs(testNoise)
        sigT = test[:,:16,:]
        noiseT = test[:,16:18,:]
        sigT = np.expand_dims(sigT,3)
        sigT = np.transpose(sigT,(0,3,2,1))
        noiseT = np.expand_dims(noiseT,3)
        noiseT = np.transpose(noiseT,(0,3,2,1))
        sigTest = SplitComplexR(sigT)
        noiseTest = SplitComplexR(noiseT)
        np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigTest.npy',sigTest)
        np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseTest.npy',noiseTest)    
    return(sigTrain,noiseTrain,sigVal,noiseVal,sigTest,noiseTest)


def processFullMRIFileRFA0(noise, date, testNoise=None):
    sample,b = common.readAllAcqs(noise)
    sig1 = sample[:,:16,:]
    noise1 = sample[:,16:18,:]
    sig1 = np.expand_dims(sig1,3)
    sig1 = np.transpose(sig1,(0,3,2,1))
    noise1 = np.expand_dims(noise1,3)
    noise1 = np.transpose(noise1,(0,3,2,1))
    l = sig1.shape[0]
    l1 = int(l*0.8)
    l2 = int(l*0.8)
    sigTrain = SplitComplexR(sig1[0:l1])
    noiseTrain = SplitComplexR(noise1[0:l1])
    sigVal = SplitComplexR(sig1[l1:l])
    noiseVal = SplitComplexR(noise1[l1:l])
    sigTest = SplitComplexR(sig1)
    noiseTest = SplitComplexR(noise1)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigTrainingFA0.npy',SplitComplexR(sig1[0:l1]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseTrainingFA0.npy',SplitComplexR(noise1[0:l1]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigValFA0.npy',SplitComplexR(sig1[l1:l]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseValFA0.npy',SplitComplexR(noise1[l1:l]))
    if testNoise != None :
        test,c = common.readAllAcqs(testNoise)
        sigT = test[:,:16,:]
        noiseT = test[:,16:18,:]
        sigT = np.expand_dims(sigT,3)
        sigT = np.transpose(sigT,(0,3,2,1))
        noiseT = np.expand_dims(noiseT,3)
        noiseT = np.transpose(noiseT,(0,3,2,1))
        np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigTest.npy',SplitComplexR(sigT))
        np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseTest.npy',SplitComplexR(noiseT))    
    return(sigTrain,noiseTrain,sigVal,noiseVal,sigTest,noiseTest)
    
def directDSPR(noise,baseline):
    baseline,a = common.readAllAcqs(baseline)
    baseline = baseline[:,0:16]
    sample,b = common.readAllAcqs(noise)
    sample = np.expand_dims(sample,3)
    sample = np.transpose(sample,(0,3,2,1))
    baseline = np.expand_dims(baseline,3)
    baseline = np.transpose(baseline,(0,3,2,1))
    l = sample.shape[0]
    l1 = int(l*0.6)
    l2 = int(l*0.8)
    
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/sig0716S.npy',SplitComplexR(baseline[0:l]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/noise0716S.npy',SplitComplexR(sample[0:l]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/sigTraining0716S.npy',SplitComplexR(baseline[0:l1]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/noiseTraining0716S.npy',SplitComplexR(sample[0:l1]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/sigVal0716S.npy',SplitComplexR(baseline[l1:l2]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/noiseVal0716S.npy',SplitComplexR(sample[l1:l2]))
    #np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/sigTest0716S.npy',SplitComplexR(baseline[l2:l]))
    #np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/noiseTest0716S.npy',SplitComplexR(sample[l2:l]))

# Prepare the data for model training
# the input files are noise with image recordings in format (nx,ny,nSlices,nChannels)
# The output data are in 4D k space format (nSlices,ny,nchannels,nx)
# Before running this function, knock off ".view(np.complex64)" in common.readDataByUID
def storeKSpace(noise,baseline):
    baseline = common.getKspace(baseline)
    sample = common.getKspace(noise)
    sig1 = np.transpose((sample[:,:,:,:16]-baseline[:,:,:,:16]),(2,1,3,0))
    noise1 = np.transpose(sample[:,:,:,16:18],(2,1,3,0))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/imgKSpace/sigTrainingS.npy',SplitComplex3D(sig1[0:6]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/imgKSpace/noiseTrainingS.npy',SplitComplex3D(noise1[0:6]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/imgKSpace/sigValS.npy',SplitComplex3D(sig1[6:8]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/imgKSpace/noiseValS.npy',SplitComplex3D(noise1[6:8]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/imgKSpace/sigTestS.npy',SplitComplex3D(sig1[8:10]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/imgKSpace/noiseTestS.npy',SplitComplex3D(noise1[8:10]))

# Prepare the data for model training
# Inputs: 
#       noise/baselineFolder: folderpath of the folder that contains the noise (low resolution) and baseline (high resolution) files
#       subset: determine if the image is sperated into sub images to enlarge the data size
#       subWidth/subHeight: height and width of the subimage
#       trainingSize/valSize: the proportion of training and validation data in total data, test data size is 1-trainingSize-valSize
# The output data are in 3d Image space format (ny,nx,nSlices)  
def processingImage(noiseFolder,baselineFolder,subset = False, subWidth=0, subHeight=0,trainingSize=0.6,valSize=0.2):
    noisefiles = get_filepaths(noiseFolder)
    baselinefiles = get_filepaths(baselineFolder)
    noise = []
    baseline = []
    for item in noisefiles:
        noise = np.append(noise,common.recon(item), axis =2)
    for file in baselinefiles:
        baseline = np.append(baseline, common.recon(file), axis = 2)
    if subset:
        newNoise = subImageSet(noise,subWidth,subHeight)
        newBaseline = subImageSet(baseline,subWidth,subHeight)
    size = newNoise.shape
    size1 = size*trainingSize
    size2 = size*(trainingSize + valSize)
    # !!!Change filePaths!!!
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0827/sigTraining.npy',newBaseline[:,:,0:size1])
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0827/noiseTraining.npy',newNoise[:,:,0:size1])
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0827/sigVal.npy',newBaseline[:,:,size1:size2])
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0827/noiseVal.npy',newNoise[:,:,size1:size2])
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/sigTest.npy',newBaseline[:,:,size2:size])
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/noiseTest.npy',newNoise[:,:,size2:size])  

# Creating image subsets
# Inputs: 
#       images:Original Images 
#       subsetWidth/subsetHeight: height and width of the subimage 
# Output: image subsets        
def subImageSet(images, subsetWidth,subsetHeight):
    size = images.shape
    newImages = []
    for k in range(size[2]):
        for i in range(size[1]//subsetHeight):
            hStart = i*subsetHeight
            hEnd = (i+1)*subsetHeight
            for j in range(size[0]//subsetWidth):
                wStart = j*subsetWidth
                wEnd = (j+1)*subsetWidth
                newImages.append(images[hStart:hEnd,wStart:wEnd,k])
    
    return newImages

#Define path of the data file
noise = 'E:/JiaxingData/EMINoise/0827/0Flip.h5' #'E:/JiaxingData/EMINoise/0716/FlashNoiseRaw.h5',16/SubjRawNoisy
test = 'E:/JiaxingData/EMINoise/0827/NoisyImg.h5'
baseline = 'E:/JiaxingData/EMINoise/0827/BaselineImg.h5' #'E:/JiaxingData/EMINoise/0716/FlashBaselineRaw.h5', SubjRawBaseline
#E:/JiaxingData/EMINoise/0731/Baseline32_256.h5 E:/JiaxingData/EMINoise/0731/Baseline16_128.h5 E:/JiaxingData/EMINoise/0731/Baseline32_128.h5
#E:/JiaxingData/EMINoise/0731/Noise16_128_1.h5 E:/JiaxingData/EMINoise/0731/Noise32_128_1.h5 E:/JiaxingData/EMINoise/0731/Noise32_256_1.h5
#E:/JiaxingData/EMINoise/0731/sub/baseline.h5
#E:/JiaxingData/EMINoise/0731/sub/run1.h5

#processSignalFile(noise,baseline)

#processImgFile(noise,baseline)

#processFullMRIFileR(test,baseline)
              
#directDSPR(noise,baseline)

#storeKSpace(noise,baseline)

