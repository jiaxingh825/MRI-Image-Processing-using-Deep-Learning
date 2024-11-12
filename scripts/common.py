import h5py
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt

def rms(x):
    return np.sqrt(np.mean(np.abs(x) ** 2))

def readAllAcqs(fname, table_name="acq", datasrc="data"):
    fh = h5py.File(fname, "r")
    desc = getDataDescriptions(fh, table_name=table_name)

    data_grp = list(fh.get(datasrc).items())[-1][1]

    res = []
    for ix, row in desc.iterrows():
        res.append(readDataByUID(data_grp, row.uid))
    res = np.array(res)
    fh.close()
    return res, desc

def SumOfSquares(im):
    return np.sum(np.power(np.abs(im),2),axis=im.ndim-1)

def readDataByUID(data_grp, uid):
    data_string = "{:016x}".format(int(uid))
    data = data_grp.get(data_string)[()]
    return data#.view(np.complex64)

def getDesc(fname, table_name="acq"):
    fh = h5py.File(fname, "r")
    desc = getDataDescriptions(fh, table_name=table_name)
    fh.close()
    return desc

def recon(fname, table_name="acq", datasrc="data"):
    #Load in all the data
    allData,desc = readAllAcqs(fname, table_name, datasrc)
    nSlices = int(desc["slice"].max()+1)
    ny = int(desc["phase"].max()+1)
    nx = allData.shape[2]
    nChannels = allData.shape[1]
 
    #allociate space for kSpace data
    kSpace = np.zeros((nx,ny,nSlices,nChannels),dtype=np.complex64)
    #loop through slices and phase encodes put data into kSpace array
    for sliceIndex in range(nSlices):
        sliceDesc = desc.loc[desc.slice==sliceIndex]
        for ind in sliceDesc.index:
            phaseIndex = int(sliceDesc['phase'][ind])
            kSpace[:,phaseIndex,sliceIndex,:]=np.transpose(allData[ind,:,:],[1,0])
    #FT to image space
    im = f2d(kSpace,axes=[0,1])
    #Simple Sum of Square Channel Combinations
    sos = np.sum(np.power(np.abs(im),2),axis=im.ndim-1)
    
    return sos

def reconNoise(fname, table_name="acq", datasrc="data"):
    #Load in all the data
    allData,desc = readAllAcqs(fname, table_name, datasrc)
    avg = allData.shape[0]
    nx = allData.shape[2]
    nChannels = allData.shape[1]
 
    #reshape for k-space data (concatenate avg and nx together)
    kSpace = allData.reshape(avg*nx,nChannels)

    im = np.fft.fftshift(np.fft.fft((kSpace)))
    
    return im

def getRate(fname):
    fh = h5py.File(fname, "r")
    res = fh.get("metaData/receiverSampleRate")[()]
    fh.close()
    return res

def getAttributes(fname, h5path="/", tag="Parameters"):
    fh = h5py.File(fname, "r")
    res = fh.get(h5path).attrs[tag]
    fh.close()
    return res

def getMeasurementStartTimes(fname, table_name="acq"):
    fh = h5py.File(fname, "r")
    res = fh.get("headerSummary").get(table_name).get("timestamp")[()]
    fh.close()
    return res

def getKspace(fname):
    #Load in all the data
    allData,desc = readAllAcqs(fname)
    nSlices = int(desc["slice"].max()+1)
    ny = int(desc["phase"].max()+1)
    nx = allData.shape[2]
    nChannels = allData.shape[1]
 
    #allociate space for kSpace data
    kSpace = np.zeros((nx,ny,nSlices,nChannels),dtype=np.complex64)
    #loop through slices and phase encodes put data into kSpace array
    for sliceIndex in range(nSlices):
        sliceDesc = desc.loc[desc.slice==sliceIndex]
        for ind in sliceDesc.index:
            phaseIndex = int(sliceDesc['phase'][ind])
            kSpace[:,phaseIndex,sliceIndex,:]=np.transpose(allData[ind,:,:],[1,0])
    return kSpace

def getDataDescriptions(fh, table_name="acq"):
    meta = fh.get("metaData")
    grp = meta.get("dataDescriptions/{}".format(table_name))

    cols = grp.get("columnDescriptions")
    uids = grp.get("uidList")
    table = grp.get("dataTable")

    try:
        col_map = {}
        for c in cols:
            col_no = cols.get(c).attrs["columnIndex"]
            col_map[col_no] = c

        records = []
        for ix in range(len(uids)):
            rec = {"uid": uids[ix]}
            for k, v in col_map.items():
                rec[v] = table[ix, k]
            records.append(rec)
        df = pd.DataFrame(records)

    except TypeError:  # NoneType is not iterable --> col table non existant
        # This happens when we don't have encoding metadata for e.g. the noise acqs
        # or in a FID sequence. In that case just grab the UID list.
        records = uids
        df = pd.DataFrame(records, columns=["uid"])

    return df

def getDataSize(uid, dgrp):
    acq_data = readDataByUID(dgrp, uid)
    return acq_data.shape


def uidAtRow(dtab, row):
    return int(dtab.iloc[row].uid)

def f1d(x, ax=[-1]):
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(x, axes=[ax]), axis=ax), axes=[ax])

def f2d(x, axes=[-1,-2]):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x, axes=axes), axes=axes), axes=axes)

def figmri(data, clim=[None, None], cmap=plt.cm.gray, showbar=True, orient='e',interp='nearest', newfig=False):
    
    if newfig:
        plt.figure()

    if(np.iscomplexobj(data)):
        data = np.abs(data)
    
    data = data.squeeze()
    ndim = data.ndim
    orient = str.lower(orient)
    N = 1
    
    # find a decent reshaping of the array if it is high dimensional
    if ndim > 3:
        data = np.transpose(data, range(0,ndim-3) + [-2, -3, -1] )
        c = list(data.shape)
        print ([np.prod(c[:2]), np.prod(c[2:])])
        data = data.reshape([np.prod(c[:2]), np.prod(c[2:])])
    elif ndim == 3:
        c = list(data.shape)
        N=1
        if orient == 'v':
            N = 1
        elif orient == 'h':
            N = c[0]
        else:
            for k in np.arange(int(np.floor(np.sqrt(c[0]))),0,-1):
                if np.remainder(c[0],k) == 0:
                    N=k
                    break
                    
        data = data.reshape([int(c[0]/N), N] + c[1:] )
        data = np.transpose(data, [0, 2, 1, 3])
        c = data.shape
        data = data.reshape([np.prod(c[:2]), np.prod(c[2:])])
        
    if orient == 't':
        data = np.rot90(data)
    
    plt.imshow(data,cmap=cmap,interpolation=interp)
    plt.clim(clim)
    
    
    
    if showbar:
        plt.colorbar()
        
def PlotSlices(sos, min=None, max=None):
    sos = np.abs(sos)  # Ensure data is in magnitude form
    if sos.ndim == 2:
        # If 2D data, plot the single image
        plt.imshow(sos, cmap='gray', vmin=min, vmax=max)
        plt.axis('off')
        plt.show()
    elif sos.ndim == 3:
        # If 3D data, treat it as a stack of 2D slices
        slices = sos.shape[2]
        cols = 5
        rows = np.ceil(slices / cols).astype(int)

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        for i in range(slices):
            slice_img = np.abs(sos[:, :, i])
            r = i // cols
            c = i % cols
            axes[r, c].imshow(slice_img, cmap='gray', vmin=min, vmax=max)
            axes[r, c].axis('off')

        for j in range(i + 1, rows * cols):
            fig.delaxes(axes.flat[j])

        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("Input sos must be 2D or 3D")
    



