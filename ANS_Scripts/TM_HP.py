import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt
import keras_tuner
from keras_tuner import RandomSearch
from keras.regularizers import l2

#batch size
epoch_num = 400


def get_model(hp):
    #learning rate
    lr = hp.Float('learning_rate', min_value=0.00001, max_value=0.1)
    input = tf.keras.Input(shape = (4,320,1))
    RType = hp.Choice("regularization_type",["L2","DP"])
    with hp.conditional_scope("regularization_type", ["L2"]):
        if(RType == "L2"):
            L2 = hp.Float('weight_deacy', min_value=0.00001, max_value=0.1)
            x = tfl.Conv2D(128, (11,11), strides=1, padding='same',kernel_regularizer=l2(L2))(input)#kernel_regularizer=l2(L2)xx
            x = tfl.BatchNormalization()(x)
            x = tfl.ReLU()(x)
            x = tfl.Conv2D(64,(9,9),strides=1, padding = "same",kernel_regularizer=l2(L2) )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.ReLU()(x)
            x = tfl.Conv2D(32,(5,5),strides=1, padding = "same",kernel_regularizer=l2(L2) )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.Conv2D(32,(1,1),strides=1, padding = "same",kernel_regularizer=l2(L2) )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.ReLU()(x)
            x = tfl.Conv2D(16,(7,7),strides=1, padding = "same",kernel_regularizer=l2(L2) )(x)
    with hp.conditional_scope("regularization_type", ["DP"]):
        if(RType == "DP"):
            dp = hp.Float('dropout', min_value=0.0, max_value=0.7,step=0.05)
            x = tfl.Conv2D(128, (11,11), strides=1, padding='same')(input)#kernel_regularizer=l2(L2)xx
            x = tfl.BatchNormalization()(x)
            x = tfl.ReLU()(x)
            x = tfl.Dropout(dp)(x)
            x = tfl.Conv2D(64,(9,9),strides=1, padding = "same" )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.ReLU()(x)
            x = tfl.Dropout(dp)(x)
            x = tfl.Conv2D(32,(5,5),strides=1, padding = "same" )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.ReLU()(x)
            x = tfl.Dropout(dp)(x)
            x = tfl.Conv2D(32,(1,1),strides=1, padding = "same" )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.ReLU()(x)
            x = tfl.Dropout(dp)(x)
            x = tfl.Conv2D(16,(7,7),strides=1, padding = "same" )(x)
    x = tfl.Flatten()(x)
    x = tfl.Dense((16*640))(x)
    x = tfl.Reshape((32,320))(x)
    model = tf.keras.Model(input, outputs=x)
    model.compile(optimizer=Adam(learning_rate = lr), loss = "mse",metrics = [RootMeanSquaredError()])
    return model

tuner = RandomSearch(
    get_model,
    objective='val_loss',
    max_trials=20,
    directory= "E:/JiaxingData/ActiveNoiseSensingModel",
    project_name="ParameterSeach0708"
    
)

def lrDeacy(epoch):
    return lr*0.95**(epoch//4)
# learning rate uodate callback
LRC = tf.keras.callbacks.LearningRateScheduler(lrDeacy)

# early stopping callback
ESC = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   # Metric to be monitored
    patience=10,           # Number of epochs to wait for improvement
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored metric
)

NoiseCoilTrain = np.load('E:/JiaxingData/ActiveNoiseSensingModel/img/noiseTraining.npy')
MRICoilTrain = np.load('E:/JiaxingData/ActiveNoiseSensingModel/img/sigTraining.npy')
NoiseCoilVal = np.load('E:/JiaxingData/ActiveNoiseSensingModel/img/noiseVal.npy')
MRICoilVal = np.load('E:/JiaxingData/ActiveNoiseSensingModel/img/sigVal.npy')

#NoiseCoilTrain = dataNormalization(NoiseCoilTrain)
#MRICoilTrain = dataNormalization(MRICoilTrain)
#NoiseCoilVal = dataNormalization(NoiseCoilVal)
#MRICoilVal = dataNormalization(MRICoilVal)

# add axis for conv
NoiseCoilTrain = np.expand_dims(NoiseCoilTrain,3)
MRICoilTrain = np.expand_dims(MRICoilTrain,3)
NoiseCoilVal = np.expand_dims(NoiseCoilVal,3)
MRICoilVal = np.expand_dims(MRICoilVal,3)

tuner.search(NoiseCoilTrain, MRICoilTrain, epochs=40,
             validation_data = (NoiseCoilVal,MRICoilVal),batch_size = 16,callbacks = [ESC])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
tuner.results_summary(num_trials=5)
# Build the best model
#model = tuner.hypermodel.build(best_hps)

#model.summary()

#history = model.fit(NoiseCoilTrain, MRICoilTrain, epochs = epoch_num, validation_data = (NoiseCoilVal,MRICoilVal),
                    #callbacks = [LRC,ESC], batch_size = bs)
# model.save('E:/JiaxingData/ActiveNoiseSensingModel/Model_2_16.h5')
#plotHist("val_loss",history)
