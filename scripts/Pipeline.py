import h5py
import numpy as np
import keras
import sys
import matplotlib.pyplot as plt
import pandas as pd
import sys
import common
import DataProcessing as dp
import TrainModel as m
import TrainAndTest as tt
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt

### Data processing

## Define file paths
date = '0827'
homePath = 'E:/JiaxingData'
noise = homePath+'/EMINoise/'+date+'/0Flip.h5'
baseline = homePath+'/EMINoise/'+date+'/BaselineImg.h5'
test = ''
## 
#processSignalFile(noise,baseline)
sigTrain,noiseTrain,sigVal,noiseVal,sigTest,noiseTest = dp.processFullMRIFileR(noise,baseline,date)        
#directDSPR(noise,baseline)


### Model training
bs = 16
epoch_num = 400
lr = 0.0002

N = noiseTrain.shape[2]
model = m.get_model(N)

# learning rate uodate callback
LRC = tt.learningRateUpdate()

# early stopping callback, stop the training if the validation loss stop redcuing
ESC = tt.earlyStopUpdate
model.summary()
model.compile(optimizer=Adam(learning_rate = lr), loss = "mse",metrics = [RootMeanSquaredError()])
history = model.fit(noiseTrain, sigTrain, epochs = epoch_num, 
                    validation_data = (noiseVal,sigVal),
                    callbacks = [LRC,ESC], batch_size = bs)
# format: model_xxx(batch size)_xxx(feature s.a.image,signal only, flip angle 0)_x(subject(s) or object(o))
modelName = 'Model'
#!!!Change File Paths !!!
model.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/'+date+'/'+modelName+'.h5')
tt.plotHist("loss",history,modelName)
tt.plotHist("val_loss",history,modelName)



### Testing
predicted = model.predict(noiseTest)
corrected = sigTest - predicted
corrected = tt.ConvergeComplexR(corrected)
baseline = tt.ConvergeComplex(baseline)
sampleMRI = tt.ConvergeComplexR(sigTest)
mse = np.mean(np.square(corrected - baseline))#[subLength:length]))
print("Mean square Error on sample Image:", mse)
# Initialize array for cleaned data
#sampleCorrected = np.zeros((length,16,nPoints),dtype=np.complex64)
#sampleCorrected[0:subLength] = sample[0:subLength]
#sampleCorrected[subLength:length] = corrected

# Convert data into 3D k-space and image space !!!Change File Paths !!!
NoiseMap = dp.toImg(dp.toKSpace(tt.ConvergeComplexR(predicted),noise))
CleanImg = dp.toImg(dp.toKSpace(corrected,noise))
NoisyImg = dp.toImg(dp.toKSpace(sampleMRI,noise))
BaselineImg = dp.toImg(dp.toKSpace(baseline,baseline))

tt.plotAll(BaselineImg,NoisyImg,CleanImg,NoiseMap)
tt.plotSamples(CleanImg,NoisyImg,BaselineImg,date)
tt.plotFFTComparsion(corrected,sampleMRI,baseline,date)