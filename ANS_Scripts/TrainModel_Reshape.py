import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt

#batch size, iteration number, learning rate
bs = 16
epoch_num = 400
lr = 0.0002

def get_model(N):
    input = tf.keras.Input(shape = (2,N,2))
    x = tfl.Conv2D(128, (11,11), strides=1, padding='same')(x)#kernel_regularizer=l2(0.01)xx
    x = tfl.BatchNormalization()(x)
    x = tfl.ReLU()(x)
    #x = tfl.Dropout(0.2)(x)
    x = tfl.Conv2D(64,(9,9),strides=1, padding = "same" )(x)
    x = tfl.BatchNormalization()(x)
    x = tfl.ReLU()(x)
    #x = tfl.Dropout(0.2)(x)
    x = tfl.Conv2D(64,(5,5),strides=1, padding = "same" )(x)
    x = tfl.BatchNormalization()(x)
    x = tfl.ReLU()(x)
    #x = tfl.Dropout(0.2)(x)
    x = tfl.Conv2D(32,(1,1),strides=1, padding = "same" )(x)
    x = tfl.BatchNormalization()(x)
    x = tfl.ReLU()(x)
    #x = tfl.Dropout(0.2)(x)
    x = tfl.Conv2D(16,(7,7),strides=1, padding = "same" )(x)
    #x = tfl.Flatten()(x)
    #x = tfl.Dense((16*512))(x)
    #x = tfl.Reshape((32,256))(x)
    model = tf.keras.Model(input, outputs=x)
    return model

#Algorithm for learning rate deacy
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

#Load Datasets, test chaannel 0(0:2),3(6:8),12(24:26)
NoiseCoilTrain = np.load('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0827/noiseTrainingI.npy')
MRICoilTrain = np.load('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0827/sigTrainingI.npy')
NoiseCoilVal = np.load('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0827/noiseValI.npy')
MRICoilVal = np.load('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0827/sigValI.npy')

#Test on small dataset
#NoiseCoilTrain = np.concatenate((NoiseCoilTrain[0:100],NoiseCoilTrain[4500:4600]),axis = 0)
#MRICoilTrain = np.concatenate((MRICoilTrain[0:100],MRICoilTrain[4500:4600]),axis = 0)
#NoiseCoilVal = np.concatenate((NoiseCoilVal[0:100],NoiseCoilVal[4500:4600]),axis = 0)
#MRICoilVal = np.concatenate((MRICoilVal[0:100],MRICoilVal[4500:4600]),axis = 0)

N = NoiseCoilTrain.shape[2]

model = get_model(N)
model.summary()
model.compile(optimizer=Adam(learning_rate = lr), loss = "mse",metrics = [RootMeanSquaredError()])
history = model.fit(NoiseCoilTrain, MRICoilTrain, epochs = epoch_num, 
                    validation_data = (NoiseCoilVal,MRICoilVal),
                    callbacks = [LRC,ESC], batch_size = bs)
#model.save('E:/JiaxingData/ActiveNoiseSensingModel/data/reshape/0827/Model_I.h5')

def plotHist(name, hist):
    hist = hist.history[name]
    epochs = range(1,len(hist)+1)

    plt.plot(epochs,hist)
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.show()
    
#plotHist("val_loss",history)