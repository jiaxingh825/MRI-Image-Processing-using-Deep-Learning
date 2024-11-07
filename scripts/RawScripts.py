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


baseline,b = common.readAllAcqs('C:/Active_Noise_Sensing/EMINoise/0827/NoisyImg.h5')
baseline1,b = common.readAllAcqs('C:/Active_Noise_Sensing/EMINoise/0827/NoisyImgNew.h5')
#baseline = common.toImg(common.toKSpace(baseline,'E:/JiaxingData/EMINoise/0816/NoiseSignal.h5'))
print(baseline1[0,0,0:5])
print(baseline[0,0,0:5])
