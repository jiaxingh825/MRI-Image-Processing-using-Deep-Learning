import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle as pk 
import tensorflow as tf
import function_Bnak as fb
from model import get_Model 
from patchify import patchify, unpatchify

load_data = True # either load images from pkl or create pkls from file path
load_model = True # False # load tf model from saved file

# image and patch dimensions
HR_image_size = 512 
LR_image_size = int(HR_image_size / 2)
patch_size = 128
num_patches = int((512 / patch_size) ** 2)

# downsampling for LR images by factor of 2 (in both x and y dimensions)
sample_idx_start = int(HR_image_size / 4) # 128
sample_idx_end = int(HR_image_size - (HR_image_size / 4)) # 384

print(f'Image info: HR image size = {HR_image_size}, LR image size = {LR_image_size}, patch size = {patch_size}')

T1filenames = glob.glob('data\**\T1Sag\OutputImages*\*.h5')
T2filenames = glob.glob('data\**\T2Ax\OutputImages*\*.h5')

#print(T1filenames)
#print(T2filenames)

HR_pkl = 'HR_data.pkl'
LR_pkl = 'LR_data.pkl'

model_output = 'sr_model.keras'
history_pkl = 'training_history.pkl'

HR_T1Data, HR_T2Data = [], []
HR_T1Patches, HR_T2Patches = [], [] 

LR_T1Data, LR_T2Data = [], []
LR_T1Patches, LR_T2Patches = [], []

####################################
########## PREPARING DATA ##########
####################################

# get HR image data
if load_data: # load T1 and T2 data from pkl
    print('Loading data...')

    print('--> Reading HR data from pickle')
    with open(HR_pkl, 'rb') as handle:  
        HR_T1Data, HR_T2Data, HR_T1Patches, HR_T2Patches = pk.load(handle)
    
    print('--> Reading LR data from pickle')
    with open(LR_pkl, 'rb') as handle:
        LR_T1Data, LR_T2Data, LR_T1Patches, LR_T2Patches = pk.load(handle)


else: # get data from h5 images
    print('Formatting data...')
    
    # HR T1 image data
    print('--> Preparing HR T1 images')
    for filename in T1filenames:
        with h5py.File(filename, 'r') as f:
            keys = list(f.keys())

            for key in keys:

                data = list(f[key])

                for x in data:
                    ds_obj = f[key][x]
                    ds_arr = f[key][x][()]
                    HR_T1Data.append(ds_arr)
                    HR_T1Patches = fb.create_patches(ds_arr, patch_size, HR_T1Patches)


    # HR T2 image data
    print('--> Preparing HR T2 images')
    for filename in T2filenames:
        with h5py.File(filename, 'r') as f:
            keys = list(f.keys())

            for key in keys:

                data = list(f[key])

                for x in data:
                    ds_obj = f[key][x]
                    ds_arr = f[key][x][()]
                    HR_T2Data.append(ds_arr)
                    HR_T2Patches = fb.create_patches(ds_arr, patch_size, HR_T2Patches)


    # T1Data = np.rot90(np.array(T1Data), k=3, axes=(1,2))
    # T2Data = np.rot90(np.array(T2Data), k=3, axes=(1,2))
    print('--> Creating HR matrices')
    HR_T1Data = np.array(HR_T1Data)
    HR_T2Data = np.array(HR_T2Data)
    HR_T1Patches = np.array(HR_T1Patches)
    HR_T2Patches = np.array(HR_T2Patches)


    print('--> Writing to HR data pickle')
    with open(HR_pkl, 'wb') as handle: # save data in pkl
        pk.dump([HR_T1Data, HR_T2Data, HR_T1Patches, HR_T2Patches], handle, protocol=pk.HIGHEST_PROTOCOL)


    # LR T1 image data
    print('--> Downsampling LR T1 images')
    for i in range(HR_T1Data.shape[0]):
        LR_imData,lost_imData = fb.create_LR_image(HR_T1Data[i])
        LR_T1Data.append(LR_imData)
        LR_T1Patches = fb.create_patches(LR_imData, int(patch_size/2), LR_T1Patches)

    
    # LR T2 image data
    print('--> Downsampling LR T2 images')
    for i in range(HR_T2Data.shape[0]):
        LR_imData,lost_imData = fb.create_LR_image(HR_T2Data[i])
        LR_T2Data.append(LR_imData)
        LR_T2Patches = fb.create_patches(LR_imData, int(patch_size/2), LR_T2Patches)


    print('--> Creating LR matrices')
    LR_T1Data = np.array(LR_T1Data)
    LR_T2Data = np.array(LR_T2Data)
    LR_T1Patches = np.array(LR_T1Patches)
    LR_T2Patches = np.array(LR_T2Patches)

    
    print('--> Writing to LR data pickle')
    with open(LR_pkl, 'wb') as handle: # save data in pkl
        pk.dump([LR_T1Data, LR_T2Data, LR_T1Patches, LR_T2Patches], handle, protocol=pk.HIGHEST_PROTOCOL)


# data for full model
"""
HR_data = np.concatenate((HR_T1Data, HR_T2Data))
HR_patches = np.concatenate((HR_T1Patches, HR_T2Patches))
LR_data = np.concatenate((LR_T1Data, LR_T2Data))
LR_patches = np.concatenate((LR_T1Patches, LR_T2Patches))
"""

# data for reduced model
HR_data = HR_T2Data
HR_patches = HR_T2Patches
LR_data = LR_T2Data
LR_patches = LR_T2Patches

print('\n')
print(f'HR T1 images: {HR_T1Data.shape}, LR T1 images: {LR_T1Data.shape}')
print(f'HR T2 images: {HR_T2Data.shape}, LR T2 images: {LR_T2Data.shape}')
print(f'HR T1 patches: {HR_T1Patches.shape}, LR T1 patches: {LR_T1Patches.shape}')
print(f'HR T2 patches: {HR_T2Patches.shape}, LR T2 patches: {LR_T2Patches.shape}')
print(f'HR data: {HR_data.shape}, LR data: {LR_data.shape}')
print(f'HR patches: {HR_patches.shape}, LR patches: {LR_patches.shape}')
print('\n')



sr_model = get_Model() 
sr_model.summary()


####################################
########## MODEL TRAINING ##########
####################################

sr_model = get_Model() 
sr_model.summary()

if load_model:
    print('Loading model...')
    sr_model = tf.keras.models.load_model('run1/sr_model.keras') # reconstruct trained model from file

else:
    print('Starting model training...')
    sr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error')

    # filter patches to train only with ones where > 50% patch contain signal information
    train_indices = []
    for i in range(HR_patches.shape[0]):
        if np.count_nonzero(HR_patches[i]) > int((patch_size ** 2) / 2):
            train_indices.append(i)

    # reshape input
    X_train, Y_train = np.expand_dims(LR_patches[train_indices], 3), np.expand_dims(HR_patches[train_indices], 3)
    print(f'X: {X_train.shape}, Y: {Y_train.shape}')

    batch_size = 20
    epochs = 50

    # training
    history = sr_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # save model and its history
    sr_model.save(model_output)

    with open(history_pkl, 'wb') as handle:
        pk.dump(history.history, handle, protocol=pk.HIGHEST_PROTOCOL)


##############################
########## PLOTTING ##########
##############################

print('Plotting...')
fb.plot_output([HR_T1Data[150], HR_T1Data[450], HR_T1Data[1450]], 'new_output.png')

print('Done.')