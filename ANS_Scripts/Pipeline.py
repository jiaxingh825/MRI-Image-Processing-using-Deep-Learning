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
fPath = 'E:/JiaxingData/EMINoise/'+date+'/'
noiseName = '0Flip'
baselineName = 'BaselineImg'
testName = ''
noise = fPath+noiseName+'.h5'
baseline = fPath+baselineName+'.h5'
test = None

## pre-processing data
sigTrain,noiseTrain,sigVal,noiseVal,sigTest,noiseTest = dp.processFullMRIFileR(noise,baseline,date,test)



### Model training
bs = 16
epoch_num = 400
lr = 0.0002

N = noiseTrain.shape[2]
model = m.get_model(N)

def lrDeacy(epoch):
    return lr*0.9**(epoch//4)
# learning rate uodate callback
LRC = tf.keras.callbacks.LearningRateScheduler(lrDeacy)

# early stopping callback, stop the training if the validation loss stop redcuing
ESC = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   # Metric to be monitored
    patience=5,           # Number of epochs to wait for improvement
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored metric
    )

model.summary()
model.compile(optimizer=Adam(learning_rate = lr), loss = "mse",metrics = [RootMeanSquaredError()])
history = model.fit(noiseTrain, sigTrain, epochs = epoch_num, 
                    validation_data = (noiseVal,sigVal),
                    callbacks = [LRC,ESC], batch_size = bs)

# model name format: 
# model_xxx(batch size)_xxx(feature s.a.image,signal only, flip angle 0)_x(subject(s) or object(o))
modelName = 'Model'
model.save('E:/JiaxingData/DeepLearningModels/data/ANS/'+date+'/'+modelName+'.h5')

#plot corresponding loss and validation loss
tt.plotHist("loss",history,modelName)
tt.plotHist("val_loss",history,modelName)



### Testing
# predicting the noise map using the model obtained above
predicted = model.predict(noiseTest)
corrected = sigTest - predicted
mse = np.mean(np.square(corrected - baseline))#[subLength:length]))
print("Mean square Error on sample Image:", mse)
# Initialize array for cleaned data
#sampleCorrected = np.zeros((length,16,nPoints),dtype=np.complex64)
#sampleCorrected[0:subLength] = sample[0:subLength]
#sampleCorrected[subLength:length] = corrected

# Convert data into complex values and then k-space and image space
corrected = tt.ConvergeComplexR(corrected)
baseline = tt.ConvergeComplex(baseline)
sampleMRI = tt.ConvergeComplexR(sigTest)
NoiseMap = dp.toImg(dp.toKSpace(tt.ConvergeComplexR(predicted),noise))
CleanImg = dp.toImg(dp.toKSpace(corrected,noise))
NoisyImg = dp.toImg(dp.toKSpace(sampleMRI,noise))
BaselineImg = dp.toImg(dp.toKSpace(baseline,baseline))

# Creating new h5py file that stores the noise removed data
tt.storePrediction(fPath,noiseName,dp.complexRearrangement(corrected))
tt.storePrediction18To16(fPath, noiseName)

# plot the results
tt.plotAll(BaselineImg,NoisyImg,CleanImg,NoiseMap)
tt.plotSamples(CleanImg,NoisyImg,BaselineImg,date)
tt.plotFFTComparsion(corrected,sampleMRI,baseline,date)
