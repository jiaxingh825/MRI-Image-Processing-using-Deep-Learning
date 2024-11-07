import h5py
import numpy as np
import keras
import sys
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("C:/code/mri/commonLib")
import common
import tensorflow as tf
import tensorflow.keras.layers as tfl

def readSignal(fileName,dataType = "data"):
    f = h5py.File(fileName)
    volGroup = list(f[dataType].keys())
    sigList = list(f[dataType][volGroup[0]].keys())
    count = len(sigList)
    print(count)
    sig = np.empty((16,0))
    noise = np.empty((2,0))
    
    #sigDic = {}
    for i in range(0, count-1):
        np.append(sig,np.array(f[dataType][volGroup[0]][sigList[i]])[:16,],axis=1)
        #sigDic[sigList[i]] = np.array(f[dataType][volGroup[0]][sigList[i]])
        np.append(noise,np.array(f[dataType][volGroup[0]][sigList[i]])[-2:,],axis=1)
    return sig,noise

#baseline,b = common.readAllAcqs('E:/JiaxingData/EMINoise/0827/NoiseSignal_32_256.h5')
sig,noise = readSignal('E:/JiaxingData/EMINoise/0827/NoisyImg.h5')
#baseline = common.toImg(common.toKSpace(baseline,'E:/JiaxingData/EMINoise/0816/NoiseSignal.h5'))
print(sig.shape)
