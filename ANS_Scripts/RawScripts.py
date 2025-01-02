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

noise = np.squeeze(common.recon('E:/JiaxingData/EMINoise/1209/AMSineFA77_3.h5'))
print(noise.shape)
max= np.max(noise)/12000
min=np.min(noise)
common.PlotSlices(noise,min,max)

