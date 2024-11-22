import h5py
import json
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


# Learning Rate deacy algorithm 
# Inputs: 
#       lr: learning_rate
#       epoch: epoch number
# Outputs:
#        the updated learning rate
def lrDeacy(lr,epoch):
    return tf.keras.callbacks.LearningRateScheduler(lr*0.9**(epoch//4))

# Learning Rate updating scheduler
# Inputs: 
# Outputs:
#        Learning Rate updating scheduler
def learningRateUpdate():
    return tf.keras.callbacks.LearningRateScheduler(lrDeacy)


# Early stopping callback, stop the training if the validation loss stop redcuing
# Inputs: 
# Outputs:
#        Early stopping scheduler
def earlyStopUpdate():
    ESC = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',   # Metric to be monitored
        patience=5,           # Number of epochs to wait for improvement
        restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored metric
    )
    return ESC

# Early stopping callback, stop the training if the validation loss stop redcuing
# Inputs: 
# Outputs:
#        Early stopping scheduler
def plotHist(name, hist,date,modelName):
    hist = hist.history[name]
    epochs = range(1,len(hist)+1)

    plt.plot(epochs,hist)
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.savefig(f'E:/JiaxingData/ActiveNoiseSensingModel/ImgResults/'+date+'/'+modelName+name+'.png')
    plt.show()


# Return the max-min normalization of the data
# Inputs: 
#       data
# Outputs:
#       normalized data
def normalized(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

# plot the data in the frequency domain with baseline, noisy (original), and cleaned as comparsion for all channels 
# Inputs: 
#       corrected: cleaned data fft
#       sampleMRI: noisy data ftt
#       baseline: baseline data fft
#       date: date of recording of the given data

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
            
# plot samples of baseline, original, and noisy image with their k space image
# Inputs: 
#       corrected: cleaned data fft
#       sampleMRI: noisy data ftt
#       baseline: baseline data fft
#       date: date of recording of the given data
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
    plt.savefig('E:/JiaxingData/ActiveNoiseSensingModel/ImgResults/'+date+'/object/IMG/sample.png')
    plt.show()

# store the corrected k space signal into a copy of the original h5py file in the same folder
# Inputs: 
#       fPath: Folder path that contains the targeting file 
#       fName: the file name without postfix
#       corrected: the corrected data to be stored
def storePrediction(fPath, fName, corrected):
    # use the shutil.copyobj() method to copy the contents of source_file to destination_file
    shutil.copy((fPath+fName+'.h5'), fPath+fName+'New.h5')
    f1 = h5py.File(fPath+fName+'New.h5', 'r+')     # open the file
    desc = common.getDataDescriptions(f1, table_name='acq')
    volGroup = list(f1['data'].keys())
    for i, row in desc.iterrows():
        data_string = "{:016x}".format(int(row.uid))
        data = f1['data'][volGroup[0]][data_string]       # load the data
        #del f1['data'][volGroup[0]][data_string]
        
        data[0:16] = corrected[i]                      # assign new values to data
        #path = 'data/'+volGroup[0]+'/'+data_string
        #f1.create_dataset(path, data=corrected[i])
    f1.close()

# plot all slices of baseline, noisy, cleaned, and noise map image
# Inputs: 
#       BaselineImage:
#       NoiseImage
#       CleanImage
#       NoiseMap: date of recording of the given data
def plotAll(BaselineImage,NoiseImage,CleanImage,NoiseMap):
    max= np.max(BaselineImage)/1
    min=np.min(BaselineImage)
    common.PlotSlices(BaselineImage,min,max)
    max= np.max(NoiseImage)/500
    min=np.min(NoiseImage)
    common.PlotSlices(NoiseImage,min,max)
    max= np.max(CleanImage)/70
    min=np.min(CleanImage)
    common.PlotSlices(CleanImage,min,max)
    max= np.max(NoiseMap)/500
    min=np.min(NoiseMap)
    common.PlotSlices(NoiseMap,min,max)

# make a copy of the original h5py file in the same folder
# excluding the extra noise channels and updating parameter, attributes, etc.
# Inputs: 
#       fPath: Folder path that contains the targeting file 
#       fName: the file name without postfix
def storePrediction18To16(fPath, fName):
    # open the files
    subset_channels = np.arange(2,18)
    fs = h5py.File(fPath+fName+'New.h5', 'r')
    fd = h5py.File(fPath+fName+'New16.h5','w')
    # copy and change some root attributes
    subset_inds = CopyRootAttributes(fs,fd,subset_channels)
    # copy groups we don't need to change
    fs.copy('metaData',fd)
    fs.copy('status',fd)
    fs.copy('store',fd)
    # copy specific groups to change
    CopyHeaderSummary(fs,fd,subset_inds)
    CopyDataGroup(fs,fd,subset_inds)
    CopyDataGroup(fs,fd,subset_inds,"rawData")
    # close the files
    fs.close()
    fd.close()


def LoadParameters(f):
    g = f['/']
    pstr = g.attrs['Parameters']
    pmap = json.loads(pstr)
    return pmap

def ConvertStringToIntList(stringList):
    # get rid of enclosing brackets
    tempVal = stringList[1:-1]
    # break up based on comma
    tempVal = tempVal.split(',')
    outVal = []
    for s in tempVal:
        outVal.append(int(s))
    return outVal

def ConvertIntListToString(intList):
    outStr = "["
    for s in intList:
        outStr = outStr + str(s) + ","
    # remove last comma, add on end bracket
    outStr = outStr[:-1] + "]"
    return outStr

def UpdateParameters(pmap,channelSubset):
    # get the channels from the pmap, 
    # check that subset is actually a subset
    ch_int = ConvertStringToIntList(pmap['ActiveChannels'])
    ch_gain = ConvertStringToIntList(pmap['ActiveChannelsGain'])
    
    # now check all substrings and make sure in the list, 
    # get the index
    subsetInds = []
    for s in channelSubset:
        foundVal = False
        for i,v in enumerate(ch_int):
            if s==v:
                subsetInds.append(i)
                foundVal = True
                break
        
        assert foundVal, "subset channel "+str(s)+" is not in the active channels list"
    # make a subset gain based on the subset indices
    subsetGain = [ch_gain[i] for i in subsetInds]
    # convert channel subset and gain subset to strings
    ch_ss_str = ConvertIntListToString(channelSubset)
    g_ss_str = ConvertIntListToString(subsetGain)
    # put back into json
    pmap['ActiveChannels'] = ch_ss_str
    pmap['ActiveChannelsGain'] = g_ss_str
    return subsetInds

def CopyRootAttributes(fsource,fdest,subset_channels):
    pmap = LoadParameters(fsource)
    subset_inds = UpdateParameters(pmap,subset_channels)
    fdest.attrs.create("Parameters",json.dumps(pmap))
    rootAtts = ["DataManagerRunId","DataManagerRunIdInHex"]
    for attName in rootAtts:    
        fdest.attrs.create(attName,fsource.attrs[attName])
    return subset_inds

def CopyHeaderSummary(fsource,fdest,subset_inds):
    gName = "headerSummary"
    # create headerSummary group
    fdest.create_group(gName)
    # get the list of sub groups from the previous file
    hs_subgroups = fsource[gName].keys()
    attNames = ["channelIndices","max","min"]
    for subgroup in hs_subgroups:
        gs = fsource[gName+"/"+subgroup]
        g = fdest[gName].create_group(subgroup)
        gs.copy("timestamp",g)
        # get the peak table, we need to subselect
        peakTable = gs["peakTable"]
        print(peakTable.shape)
        # create the peak table
        g.create_dataset("peakTable",data=peakTable[:,subset_inds])
        # copy the peak table attributes
        for attName in attNames:
            temp = peakTable.attrs[attName]
            g["peakTable"].attrs.create(attName,temp[subset_inds])
        
def CopyDataGroup(fsource,fdest,subset_inds,groupName="data"):
    attAllNames = ["acquisitionId","runId","temporalIndex","timestamp"]
    attSubsetNames = ["adcPeakValues","channelIndices"]
    fdest.create_group(groupName)
    subgroups = fsource[groupName].keys()
    for subgroup in subgroups:
        sg_s = fsource[groupName][subgroup]
        sg = fdest[groupName].create_group(subgroup)
        dsets = sg_s.keys()
        for dset in dsets:
            dset_s = sg_s[dset]
            # for now we just copy the subset of data
            data_clean = dset_s[subset_inds,:]

            sg.create_dataset(dset,data=data_clean)
            for attName in attSubsetNames:
                temp = dset_s.attrs[attName]
                sg[dset].attrs.create(attName,temp[subset_inds])
            for attName in attAllNames:
                temp = dset_s.attrs[attName]
                sg[dset].attrs.create(attName,temp)