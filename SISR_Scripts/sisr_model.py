import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle as pk 
import tensorflow as tf
import function_Bnak
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


######################################
########## HELPER FUNCTIONS ##########
######################################

# function to divide HR/LR images into patches for training
def create_patches(data_arr, size, output_arr):
    # divide each image into patches
    patches = patchify(data_arr, (size,size), step=size)

    # add patches to output array
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            output_arr.append(patches[i, j, :, :])

    return output_arr


# function to desample HR image into LR image
def create_LR_image(HR_image):
    # convert to k space
    HR_k_data = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(HR_image, axes=(-2,-1)), norm='ortho'), axes=(-2,-1))
    # sample from centre
    LR_k_data = HR_k_data[sample_idx_start:sample_idx_end, sample_idx_start:sample_idx_end]
    # reconvert to image space
    LR_imData = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(LR_k_data, axes=(-2,-1)), norm='ortho'), axes=(-2,-1)))

    return LR_imData


# function to get lost kspace data from desampling
def get_lost_img_data(HR_image):
    # convert to k space
    HR_k_data = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(HR_image, axes=(-2,-1)), norm='ortho'), axes=(-2,-1))
    # replace centre frequencies with zeros
    HR_k_data[sample_idx_start:sample_idx_end, sample_idx_start:sample_idx_end] = np.zeros((LR_image_size,LR_image_size), dtype=complex)
    # reconvert to image space
    imData = np.abs(np.fft.fftshift(np.fft.ifft2(HR_k_data, norm='ortho'), axes=(-2,-1)))

    return imData


# function to generate HR image using zero-padding in k-space
def create_zero_padded_image(LR_image):
    # convert to k space
    LR_k_data = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(LR_image, axes=(-2,-1)), norm='ortho'), axes=(-2,-1))

    # pad array with surrounding zeros
    HR_k_data = np.zeros((HR_image_size,HR_image_size), dtype=complex)
    #HR_k_data[sample_idx_start:sample_idx_end, sample_idx_start:sample_idx_end] = np.fft.ifftshift(LR_k_data, axes=(-2,-1))
    HR_k_data[sample_idx_start:sample_idx_end, sample_idx_start:sample_idx_end] = LR_k_data

    # reconvert to image space
    #HR_imData = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(LR_k_data, axes=(-2,-1)), norm='ortho'), axes=(-2,-1)))
    HR_imData = np.abs(np.fft.fftshift(np.fft.ifft2(HR_k_data, norm='ortho'), axes=(-2,-1)))

    return HR_imData


# function to run inference on LR image to predict HR counterpart using SR model
def create_SR_image(LR_data_arr):
    LR_patches = create_patches(LR_data_arr, int(patch_size/2), [])
    input = np.expand_dims(np.array(LR_patches), 3)
    HR_patches = sr_model.predict(input)

    patch_dim = int(HR_image_size/patch_size)
    HR_patches = np.reshape(HR_patches, (patch_dim, patch_dim, patch_size, patch_size))
    HR_img = unpatchify(HR_patches, (HR_image_size, HR_image_size))

    return HR_img


# function to plot and compare HR/LR/zero-padding/srdensenet images, input as list of 2D images
def plot_output(HR_images, path_for_plot):
    col_names = ['HR', 'LR', 'Lost Signal', 'Zero-padding', 'SRDenseNet', 'Diff (Zero-padding - SR)', 'Diff (HR - SR)', 'Diff (HR - Zero-padding)']
    n_images = len(HR_images)
    n_rows = 2*n_images
    n_cols = len(col_names)
    x, y = 300, 200
    box_size = 64

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    
    for n, HR_imData in enumerate(HR_images):
        # determine corresponding LR, zero-padded, and srdensenet inferred images
        LR_imData = create_LR_image(HR_imData)
        LR_loss = get_lost_img_data(HR_imData)
        HR_zeropad = create_zero_padded_image(LR_imData)
        SR_imData = create_SR_image(LR_imData)

        # rotate image clockwise
        HR_img = np.rot90((HR_imData / HR_imData.max()) * 255, k=3, axes=(0,1))
        LR_img = np.rot90((LR_imData / LR_imData.max()) * 255, k=3, axes=(0,1))
        loss_img = np.rot90((LR_loss / LR_loss.max()) * 255, k=3, axes=(0,1))
        zeropad_img = np.rot90((HR_zeropad / HR_zeropad.max()) * 255, k=3, axes=(0,1))
        sr_img = np.rot90((SR_imData / SR_imData.max()) * 255, k=3, axes=(0,1))

        # find difference between zero-padded and SR image
        diff_img_zeropad = zeropad_img - sr_img
        if np.min(diff_img_zeropad) < 0: # shift to min zero if there are negative values
            diff_img_zeropad -= np.min(diff_img_zeropad)
        diff_img_zeropad = (diff_img_zeropad / diff_img_zeropad.max()) * 255 # scaling values for plotting

        # find difference between HR and SR image
        diff_img_HR = HR_img - sr_img
        if np.min(diff_img_HR) < 0:
            diff_img_HR -= np.min(diff_img_HR)
        diff_img_HR = (diff_img_HR / diff_img_HR.max()) * 255
        print(f'Magnitude of difference b/w HR and SR: {np.sum(diff_img_HR)}')

        # find difference between HR and zero-padded image
        diff_img_HR_zeropad = HR_img - zeropad_img
        if np.min(diff_img_HR_zeropad) < 0:
            diff_img_HR_zeropad -= np.min(diff_img_HR_zeropad)
        diff_img_HR_zeropad = (diff_img_HR_zeropad / diff_img_HR_zeropad.max()) * 255
        print(f'Magnitude of difference b/w HR and zero-padding: {np.sum(diff_img_HR_zeropad)}')

        for i, col_name in enumerate(col_names):
            
            # plot full image and zoomed in image
            if col_name == col_names[0]:
                axs[2*n, i].imshow(HR_img, cmap='gray')
                axs[2*n+1, i].imshow(HR_img[y:int(y+box_size), x:int(x+box_size)], cmap='gray')
            elif col_name == col_names[1]:
                axs[2*n, i].imshow(LR_img, cmap='gray')
                axs[2*n+1, i].imshow(LR_img[int(y/2):int((y+box_size)/2), int(x/2):int((x+box_size)/2)], cmap='gray')
            elif col_name == col_names[2]:
                axs[2*n, i].imshow(loss_img, cmap='gray')
                axs[2*n+1, i].imshow(loss_img[int(y/2):int((y+box_size)/2), int(x/2):int((x+box_size)/2)], cmap='gray')
            elif col_name == col_names[3]:
                axs[2*n, i].imshow(zeropad_img, cmap='gray')
                axs[2*n+1, i].imshow(zeropad_img[y:int(y+box_size), x:int(x+box_size)], cmap='gray')
            elif col_name == col_names[4]:
                axs[2*n, i].imshow(sr_img, cmap='gray')
                axs[2*n+1, i].imshow(sr_img[y:int(y+box_size), x:int(x+box_size)], cmap='gray')
            elif col_name == col_names[5]:
                axs[2*n, i].imshow(diff_img_zeropad, cmap='gray')
                axs[2*n+1, i].imshow(diff_img_zeropad[y:int(y+box_size), x:int(x+box_size)], cmap='gray')
            elif col_name == col_names[6]:
                axs[2*n, i].imshow(diff_img_HR, cmap='gray')
                axs[2*n+1, i].imshow(diff_img_HR[y:int(y+box_size), x:int(x+box_size)], cmap='gray')
            elif col_name == col_names[7]:
                axs[2*n, i].imshow(diff_img_HR_zeropad, cmap='gray')
                axs[2*n+1, i].imshow(diff_img_HR_zeropad[y:int(y+box_size), x:int(x+box_size)], cmap='gray')

            # different box dimensions used for LR image (to highlight zoomed in section)
            if i == 1: 
                box = patches.Rectangle((int(x/2),int(y/2)), int(box_size/2), int(box_size/2), linewidth=1, edgecolor='r', facecolor='none')
            else:
                box = patches.Rectangle((x,y), box_size, box_size, linewidth=1, edgecolor='r', facecolor='none')
            
            axs[2*n, i].add_patch(box)

            # add titles only to the top of the first row
            if n == 0:
                axs[n, i].set_title(col_names[i])


    # remove x and y ticks
    for i in range(n_rows):
        for j in range(n_cols):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
    
    fig.tight_layout()
    fig.savefig(path_for_plot)
    plt.close(fig)

    return


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
                    HR_T1Patches = create_patches(ds_arr, patch_size, HR_T1Patches)


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
                    HR_T2Patches = create_patches(ds_arr, patch_size, HR_T2Patches)


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
        LR_imData = create_LR_image(HR_T1Data[i])
        LR_T1Data.append(LR_imData)
        LR_T1Patches = create_patches(LR_imData, int(patch_size/2), LR_T1Patches)

    
    # LR T2 image data
    print('--> Downsampling LR T2 images')
    for i in range(HR_T2Data.shape[0]):
        LR_imData = create_LR_image(HR_T2Data[i])
        LR_T2Data.append(LR_imData)
        LR_T2Patches = create_patches(LR_imData, int(patch_size/2), LR_T2Patches)


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


######################################
########## MODEL DEFINITION ##########
######################################

# each DB consists of 8 convolutional layers, each producing 16 feature maps
# so each DB produces 128 feature maps total
# layers receive output of all preceding layers as input 
def dense_block(x, num_layers, num_feature_maps):

    for i in range(num_layers):
        # bn = tf.keras.layers.BatchNormalization()(x)
        conv = tf.keras.layers.Conv2D(num_feature_maps, (3,3), padding='same', activation='relu')(x)

        if i == 0:
            x = conv
        else:
            x = tf.keras.layers.concatenate([conv, x])

    return x


# sr denset net cnn of 8 dense blocks
def sr_dense_net(input_shape=(64, 64, 1), num_dense_blocks=8, num_layers=8, num_feature_maps=16):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(num_feature_maps, (3,3), padding='same', activation='relu')(inputs)

    skip_outputs = x

    for _ in range(num_dense_blocks):
        x = dense_block(skip_outputs, num_layers, num_feature_maps)
        skip_outputs = tf.keras.layers.concatenate([x, skip_outputs])
        # x = tf.keras.layers.Conv2D(x.shape[-1] // 2, (1,1), padding='same', activation='relu')(x)
        # x = tf.keras.layers.AveragePooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(256, (1,1), padding='same', activation='relu')(skip_outputs)            # bottleneck layer to reduce to 256
    x = tf.keras.layers.Conv2DTranspose(256, (3,3), strides=2, padding='same', activation='relu')(x)   # deconvolution to upsample
    x = tf.keras.layers.Conv2D(1, (3,3), padding='same')(x)                                            # reduce to single channel output

    model = tf.keras.models.Model(inputs, x)
    return model


sr_model = sr_dense_net() 
sr_model.summary()


####################################
########## MODEL TRAINING ##########
####################################

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
plot_output([HR_T1Data[150], HR_T1Data[450], HR_T1Data[1450]], 'new_output.png')

print('Done.')

