import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import common
import os

# extract all file_paths in a folder
# Inputs: 
#       folder_path: the target folder path
# Outputs:
#       filepaths: an array of string which store all folderpaths
def get_filepaths(folder_path):
    filepaths = []
    for root, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            filepaths.append(filepath)
    return filepaths

# Re-arrange the data by doubling the channel size for splitting read/imagery parts
# Inputs: 
#       data: input dataset in shape (number of samples,number of channels, number of data points)
# Outputs:
#       newData: rearranged dataset in shape (number of samples, 2*number of channels, number of data points)
def SplitComplex(data):
    numPoints = int(data.shape[2])
    numChannels = data.shape[1]
    newData = np.zeros((data.shape[0],2*numChannels,numPoints//2))
    for i in range(numChannels):
        for j in range(numPoints//2):
            newData[:,2*i,j] = data[:,i,2*j]
            newData[:,2*i+1,j] = data[:,i,2*j+1]
    return newData

# Re-arrange the data by adding extra dimension for splitting read/imagery parts
# Inputs: 
#       data: input dataset in shape (number of samples,number of channels, number of data points)
# Outputs:
#        newData: rearranged dataset in shape (number of samples, 2, number of channels, number of data points)
def SplitComplexR(data):
    numPoints = int(data.shape[2])
    numChannels = data.shape[1]
    new = np.transpose(np.expand_dims(data,3),(0,3,2,1))
    newData = np.zeros((data.shape[0],2,numPoints//2,numChannels))
    for j in range(numPoints//2):
        newData[:,0,j] = data[:,0,2*j]
        newData[:,1,j] = data[:,0,2*j+1]
    return newData

# Process MRI recordings into npy file for training and validating active noise sensing model. 
# Testing files are generated based on the whole data file if it is not specified otherwise
# Inputs: 
#       noise: Path of MRI Image raw data file with noise
#       baseline: Path of MRI baseline Image raw data
#       date: the collecting date of the data 
#       testNoise: Path of a separated MRI Image raw data file with noise for testing 
# Outputs:
#        newData: rearranged dataset in shape (number of samples, 2, number of channels, number of data points)
def processFullMRIFileR(noise,baseline,date,testNoise=None):
    baseline,a = common.readAllAcqs(baseline)
    sample,b = common.readAllAcqs(noise)
    sig1 = sample[:,:16,:]-baseline[:,:16,:]
    noise1 = sample[:,16:18,:]
    l = sig1.shape[0]
    l1 = int(l*0.8)
    sigTrain = SplitComplexR(sig1[0:l1])
    noiseTrain = SplitComplexR(noise1[0:l1])
    sigVal = SplitComplexR(sig1[l1:l])
    noiseVal = SplitComplexR(noise1[l1:l])
    sigTest = SplitComplexR(sig1)
    noiseTest = SplitComplexR(noise1)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigTraining.npy',sigTrain)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseTraining.npy',noiseTrain)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigVal.npy',sigVal)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseVal.npy',noiseVal)
    if testNoise != None :
        test,c = common.readAllAcqs(testNoise)
        sigT = test[:,:16,:]
        noiseT = test[:,16:18,:]
        sigTest = SplitComplexR(sigT)
        noiseTest = SplitComplexR(noiseT)
        np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigTest.npy',sigTest)
        np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseTest.npy',noiseTest)
    else:
        np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigTest.npy',sigTest)
        np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseTest.npy',noiseTest)   
    return(sigTrain,noiseTrain,sigVal,noiseVal,sigTest,noiseTest)

# Process MRI recordings into npy file for training and validating active noise sensing model. 
# Testing files are generated based on the whole data file if it is not specified otherwise
# Inputs: 
#       noise: Path of MRI Image raw data file with noise
#       baseline: Path of MRI baseline Image raw data
#       date: the collecting date of the data 
#       testNoise: Path of a separated MRI Image raw data file with noise for testing 
# Outputs:
#        newData: rearranged dataset in shape (number of samples, 2, number of channels, number of data points)
def processFullMRIFileRFA0(noise, date, testNoise=None):
    sample,b = common.readAllAcqs(noise)
    sig1 = sample[:,:16,:]
    noise1 = sample[:,16:18,:]
    l = sig1.shape[0]
    l1 = int(l*0.8)
    sigTrain = SplitComplexR(sig1[0:l1])
    noiseTrain = SplitComplexR(noise1[0:l1])
    sigVal = SplitComplexR(sig1[l1:l])
    noiseVal = SplitComplexR(noise1[l1:l])
    sigTest = SplitComplexR(sig1)
    noiseTest = SplitComplexR(noise1)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigTrainingFA0.npy',sigTrain)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseTrainingFA0.npy',noiseTrain)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigValFA0.npy',sigVal)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseValFA0.npy',noiseVal)
    if testNoise != None :
        test,c = common.readAllAcqs(testNoise)
        sigT = test[:,:16,:]
        noiseT = test[:,16:18,:]
        sigTest = SplitComplexR(sigT)
        noiseTest = SplitComplexR(noiseT)
        np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigTest.npy',sigTest)
        np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseTest.npy',noiseTest)    
    else:
        np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/sigTest.npy',sigTest)
        np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/noiseTest.npy',noiseTest)
    return(sigTrain,noiseTrain,sigVal,noiseVal,sigTest,noiseTest)

# Process MRI recordings into npy file for training and validating direct clean signal predicting model. 
# Testing files are generated based on the whole data file if it is not specified otherwise
# Inputs: 
#       noise: Path of MRI Image raw data file with noise
#       baseline: Path of MRI baseline Image raw data
#       date: the collecting date of the data 
#       testNoise: Path of a separated MRI Image raw data file with noise for testing 
# Outputs:
#        newData: rearranged dataset in shape (number of samples, 2, number of channels, number of data points)
def directDSPR(noise,baseline,date):
    baseline,a = common.readAllAcqs(baseline)
    baseline = baseline[:,0:16]
    sample,b = common.readAllAcqs(noise)
    l = sample.shape[0]
    l1 = int(l*0.6)
    l2 = int(l*0.8)
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/'+date+'/sig.npy',SplitComplexR(baseline[0:l]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/'+date+'/noise.npy',SplitComplexR(sample[0:l]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/'+date+'/sigTraining.npy',SplitComplexR(baseline[0:l1]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/'+date+'/noiseTraining.npy',SplitComplexR(sample[0:l1]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/'+date+'/sigVal.npy',SplitComplexR(baseline[l1:l2]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/'+date+'/noiseVal.npy',SplitComplexR(sample[l1:l2]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/'+date+'/sigTest.npy',SplitComplexR(baseline[l2:l]))
    np.save('E:/JiaxingData/ActiveNoiseSensingModel/data/dsp/'+date+'/noiseTest.npy',SplitComplexR(sample[l2:l]))


# Prepare the data for model training from raw data with simple reconstruction
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
# Output: 
#       newImages: Image subsets        
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

# Convert current dataset into 2d k space signal 
# Inputs: 
#       allData: Provided data with shape (number of samples, number of channels, number of points) in complex64 format
#       fname: Path of corresponding h5py file to reconstrut 2d k space in the correct order
# Outputs:
#        kSpace: An stack of 2d k space slices
def toKSpace(allData,fname):
    fh = h5py.File(fname, "r")
    desc = common.getDataDescriptions(fh, "acq")
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

# Convert current 2d k space signal stacks into image stacks
# Inputs: 
#       kSpace: Input 2d k space array
# Outputs:
#        sos: reconstructed img stacks
def toImg(kSpace):
    im = common.f2d(kSpace,axes=[0,1])
    #Simple Sum of Square Channel Combinations
    sos = np.sum(np.power(np.abs(im),2),axis=im.ndim-1)
    return sos
