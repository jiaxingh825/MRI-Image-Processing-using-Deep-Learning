import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle as pk 
import tensorflow as tf
from patchify import patchify, unpatchify

# function to divide HR/LR images into patches for training
# Inputs: 
#       data_arr: the arr with current image
#       size: size of each sub images
#       output_arr: the output array which is empty or with previous patches
# Outputs:
#       output_arr: the output array after adding patches from the current array
def create_patches(data_arr, size, output_arr):
    # divide each image into patches
    patches = patchify(data_arr, (size,size), step=size)

    # add patches to output array
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            output_arr.append(patches[i, j])

    return output_arr


# function to desample HR image into LR image by converting to k sapce and remove details from k space
# the removed details are calculated and returned as well
# Inputs: 
#       HR_image: the targeting high resolution image
# Outputs:
#       LR_image: the output low resolution image
#       lost_imData: image which stores lost information from desampling
def create_LR_image(HR_image):
    HR_image_size = HR_image.shape[0]
    sample_idx_start = int(HR_image_size / 4) # 128
    sample_idx_end = int(HR_image_size - (HR_image_size / 4)) # 384
    # convert to k space
    HR_k_data = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(HR_image, axes=(-2,-1)), norm='ortho'), axes=(-2,-1))
    # sample from centre
    LR_k_data = HR_k_data[sample_idx_start:sample_idx_end, sample_idx_start:sample_idx_end]
    lost_k_data = HR_k_data-LR_k_data
    # reconvert to image space
    LR_imData = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(LR_k_data, axes=(-2,-1)), norm='ortho'), axes=(-2,-1)))
    lost_imData = np.abs(np.fft.fftshift(np.fft.ifft2(lost_k_data, norm='ortho'), axes=(-2,-1)))
    return LR_imData,lost_imData


# function to generate HR image using zero-padding in k-space
# Inputs: 
#       LR_image: the targeting low resolution image
#       HR_image_size: the height&width (height = width) of the high resolution size 
# Outputs:
#       HR_image: the output high resolution image after zero-padding
def create_zero_padded_image(LR_image,HR_image_size):
    sample_idx_start = int(HR_image_size / 4) # 128
    sample_idx_end = int(HR_image_size - (HR_image_size / 4)) # 384
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
# Inputs: 
#       LR_data_arr:the targeting high resolution image
#       patch_size:size of each sub image
#       HR_image_size: the height&width (height = width) of the high resolution size
#       model: the SISR model used to predict the high resolution image
# Outputs:
#       HR_img: the super resolution image predicted by the model

def create_SR_image(LR_data_arr,patch_size,HR_image_size,model):
    LR_patches = create_patches(LR_data_arr, int(patch_size/2), [])
    input = np.expand_dims(np.array(LR_patches), 3)
    HR_patches = model.predict(input)

    patch_dim = int(HR_image_size/patch_size)
    HR_patches = np.reshape(HR_patches, (patch_dim, patch_dim, patch_size, patch_size))
    HR_img = unpatchify(HR_patches, (HR_image_size, HR_image_size))

    return HR_img


# function to plot and compare HR/LR/zero-padding/srdensenet images, input as list of 2D images
# Inputs: 
#       HR_images: presenting high resolution images
#       path_for_plot: file path for storing the plot 
def plot_output(HR_images, path_for_plot):
    col_names = ['HR', 'LR', 'Lost Signal', 'Zero-padding', 'SRDenseNet', 'Diff (Zero-padding - SR)', 'Diff (HR - SR)', 'Diff (HR - Zero-padding)']
    n_images = len(HR_images)
    HR_image_size = HR_images.shape[1]
    n_rows = 2*n_images
    n_cols = len(col_names)
    x, y = 300, 200
    box_size = 64

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    
    for n, HR_imData in enumerate(HR_images):
        # determine corresponding LR, zero-padded, and srdensenet inferred images
        LR_imData,LR_loss = create_LR_image(HR_imData)
        HR_zeropad = create_zero_padded_image(LR_imData,HR_image_size)
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