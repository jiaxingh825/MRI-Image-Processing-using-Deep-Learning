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

noise = 'C:/EMINoise/0827/NoiseSignal_32_256.h5'
img = 'C:/EMINoise/0827/NoisyImg.h5'
baseline = 'C:/EMINoise/0827/BaselineImg.h5'

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


transfer = np.zeros((16,2),dtype= np.complex64)
transfer[0,0] = 0.143154+0.257166j
transfer[0,1] = 0.210251+0.26535j
transfer[1,0] = 0.166184+0.0161324j
transfer[1,1] = 0.0565163-0.0160277j
transfer[2,0] = -0.0198619-0.06122289j
transfer[2,1] = 0.0528602+0.0219627j
transfer[3,0] = 0.0215794-0.0824657j
transfer[3,1] = -0.040052+0.0313773j
transfer[4,0] = 0.0115192+0.0432668j
transfer[4,1] = -0.113523-0.0287192j
transfer[5,0] = 0.12287+0.283884j
transfer[5,1] = -0.368663-0.204291j
transfer[6,0] = -0.108182-0.214316j
transfer[6,1] = -0.165642-0.367538j
transfer[7,0] = -0.142239-0.147934j
transfer[7,1] = -0.0367343-0.00641452j
transfer[8,0] = 0.0720214-1.01172j
transfer[8,1] = -0.168139-0.786305j
transfer[9,0] = -0.161785-0.0863037j
transfer[9,1] = 0.162094+0.289849j
transfer[10,0] = 0.437264+0.121794j
transfer[10,1] = -0.2058-0.0508986j
transfer[11,0] = -0.222648+0.202678j
transfer[11,1] = 0.345868-0.455761j
transfer[12,0] = -0.162989-0.282801j
transfer[12,1] = 0.674313+0.0165195j
transfer[13,0] = -0.126767-2.22061j
transfer[13,1] = 1.9792+0.519963j
transfer[14,0] = -0.0116197+0.479963j
transfer[14,1] = 0.217842+1.59354j
transfer[15,0] = 0.428799+0.311938j
transfer[15,1] = 0.014348-0.3882j

print(transfer.shape)

pred = np.zeros(image.shape,dtype= np.complex64)
len = image.shape[0]
for k in range(len):
    pred[k] = np.dot(transfer,imageNoise[k])

cleaned = image - pred

mse = np.mean(np.square(cleaned - baseline))
print("Mean square Error on sample Image:", mse)
BaselineImage = dp.toImg(dp.toKSpace(baseline,img))
NoiseImage = dp.toImg(dp.toKSpace(image,img))
CleanImage = dp.toImg(dp.toKSpace(cleaned,img))
NoiseMap = dp.toImg(dp.toKSpace(pred,img))

print(dp.complexRearrangement(cleaned).shape)
#tt.storePrediction('C:/EMINoise/0827/','NoisyImg',dp.complexRearrangement(cleaned))


tt.plotAll(BaselineImage,NoiseImage,CleanImage,NoiseMap)