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

noisePre,b = common.readAllAcqs('E:/JiaxingData/EMINoise/0827/NoisyImg.h5',table_name="noise")
print(noisePre.shape)
noisePre = dp.ConvergeComplexR(dp.SplitComplexR(noisePre))
print(noisePre.shape)

