import h5py
import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
import DataProcessing as dp
import common
import TrainAndTest as tt

noise = 'E:/JiaxingData/EMINoise/0827/NoiseSignal_32_256.h5'
img = 'E:/JiaxingData/EMINoise/0827/NoisyImg.h5'
baseline = 'E:/JiaxingData/EMINoise/0827/BaselineImg.h5'

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
# (16,512)*(512,2) = (16,2)

noiseSignal1 = imageNoise
MRISignal1 = image-baseline
noisePreScan,b = common.readAllAcqs(img,table_name='noise')
print(noiseSignal1.shape)
print(dp.ConvergeComplexR(dp.SplitComplexR(noisePreScan)).shape)

