import numpy as np
import matplotlib.pyplot as plt
import h5py
import common
import DataProcessing as dp
from scipy import stats
from prettytable import PrettyTable

#Plot a random frequency encoding line of the sample 
def plotAcqSamples(noiseImage,flip0,noisePre,noisePre0,date,mode,type,trial):
    
    i1 = np.random.randint(0,len(noiseImage))
    i2 = np.random.randint(16,18)
    
    #noiseOnly = noiseOnly[i1,i2]
    noise = noiseImage[i1,i2]
    #noiseImage = noiseImage[i1,i2] - baseline[i1,i2]
    flip0 = flip0[i1,i2]
    noisePre = noisePre[0,i2,0:512]
    noisePre0 = noisePre0[0,i2,0:512]

    #fft1 = np.fft.fft(noiseOnly)
    fft2 = np.fft.fft(noise)
    fft3 = np.fft.fft(flip0)
    fft4 = np.fft.fft(noisePre)
    fft5 = np.fft.fft(noisePre0)
    #phase1 = np.angle(fft1)
    #ps1 = 20*np.log10(np.abs(fft1)**2)


    #sampling_frequency1 = common.getRate('E:/JiaxingData/EMINoise/0827/NoiseSignal_32_256.h5')
    sampling_frequency2 = common.getRate('E:/JiaxingData/EMINoise/'+date+'/'+mode+type+'FA77_'+trial+'.h5')
    sampling_frequency3 = common.getRate('E:/JiaxingData/EMINoise/'+date+'/'+mode+type+'FA0_'+trial+'.h5')
    #freqs1 = np.fft.fftfreq(len(noiseOnly),1/sampling_frequency1)
    freqs2 = np.fft.fftfreq(len(noise),1/sampling_frequency2)
    freqs3 = np.fft.fftfreq(len(flip0),1/sampling_frequency3)
    freqs4 = np.fft.fftfreq(len(noisePre),1/sampling_frequency2)



    #plt.plot(freqs1[:len(freqs1)//2],np.abs(fft1)[:len(freqs1)//2])
    plt.plot(freqs2[:len(freqs2)//2],np.abs(fft2)[:len(freqs2)//2])
    plt.plot(freqs3[:len(freqs3)//2],np.abs(fft3)[:len(freqs3)//2])
    plt.plot(freqs2[:len(freqs4)//2],np.abs(fft4)[:len(freqs2)//2])
    plt.plot(freqs3[:len(freqs3)//2],np.abs(fft5)[:len(freqs3)//2])
    #plt.xlim(9000, 11000)
    plt.grid()
    plt.legend(['noise map','flip Angle 0','regular noise preScan', '0FlipAngle noise preScan'])
    #plt.savefig('E:/JiaxingData/EMINoise/'+date+'/spectrumPlot/'+mode+type+trial+'_absFFT_'+str(i1)+'_'+str(i2)+'.png')
    #plt.plot(freqs[:len(freqs)//2],np.abs(fft)[:len(freqs)//2])
    plt.show()

def plotAllSamples(noiseImage,flip0,noisePre,noisePre0,date,mode,type,trial):
    noiseSize = noiseImage.shape
    #noiseOnly = noiseOnly[i1,i2]
    noise = noiseImage.transpose(1, 0, 2).reshape(noiseSize[1],noiseSize[0]*noiseSize[2])
    flip0 = flip0.transpose(1, 0, 2).reshape(noiseSize[1],noiseSize[0]*noiseSize[2])
    noisePre = noisePre[0]
    noisePre0 = noisePre0[0]
    for i in [16,17]:
        #fft1 = np.fft.fft(noiseOnly)
        fft2 = np.fft.fft(noise[i])
        fft3 = np.fft.fft(flip0[i])
        fft4 = np.fft.fft(noisePre[i])
        fft5 = np.fft.fft(noisePre0[i])
        #phase1 = np.angle(fft1)
        #ps1 = 20*np.log10(np.abs(fft1)**2)


        #sampling_frequency1 = common.getRate('E:/JiaxingData/EMINoise/0827/NoiseSignal_32_256.h5')
        sampling_frequency2 = common.getRate('E:/JiaxingData/EMINoise/'+date+'/'+mode+type+'FA77_'+trial+'.h5')
        sampling_frequency3 = common.getRate('E:/JiaxingData/EMINoise/'+date+'/'+mode+type+'FA0_'+trial+'.h5')
        #freqs1 = np.fft.fftfreq(len(noiseOnly),1/sampling_frequency1)
        freqs2 = np.fft.fftfreq(len(noise),1/sampling_frequency2)
        freqs3 = np.fft.fftfreq(len(flip0),1/sampling_frequency3)
        freqs4 = np.fft.fftfreq(len(noisePre),1/sampling_frequency2)



        #plt.plot(freqs1[:len(freqs1)//2],np.abs(fft1)[:len(freqs1)//2])
        plt.plot(freqs2[:len(freqs2)//2],np.abs(fft2)[:len(freqs2)//2])
        plt.plot(freqs3[:len(freqs3)//2],np.abs(fft3)[:len(freqs3)//2])
        plt.plot(freqs2[:len(freqs4)//2],np.abs(fft4)[:len(freqs2)//2])
        plt.plot(freqs3[:len(freqs3)//2],np.abs(fft5)[:len(freqs3)//2])
        plt.grid()
        plt.legend(['noise map','flip Angle 0','regular noise preScan', '0FlipAngle noise preScan'])
        plt.savefig('E:/JiaxingData/EMINoise/'+date+'/spectrumPlot/'+mode+type+trial+'_OverallAbsFFT_Channel'+str(i)+'.png')
        #plt.plot(freqs[:len(freqs)//2],np.abs(fft)[:len(freqs)//2])
        #plt.show()

# Calculate the PCC value of two signals, PCC is in the range of 0-1. 
# PCC closer to 1 means a higher relevance between signals. The function is 
# robust to a pair of signal or a pair of a group of signals in an array.
# Inputs: 
#       sig1
#       sig2
# Outputs:
#       real_corr: a single PCC or an array of PCC value for the real parts depends on the input type.
#       imag_corr: a single PCC or an array of PCC value for the imag parts depends on the input type.
def PCCCalculation(sig1,sig2):
    if(sig1.ndim != sig2.ndim):
            print("signal 1 and signal 2 have different dimensions")
    elif(sig1.ndim == 1):
        real_corr, _ = stats.pearsonr(np.real(sig1), np.real(sig2))
        imag_corr, _ = stats.pearsonr(np.imag(sig1), np.imag(sig2))
    else:
         real_corr = np.zeros((sig1.shape[1]))
         imag_corr = np.zeros((sig1.shape[1]))
         for i in range(0, sig1.shape[1]):
              real_corr[i],_ = stats.pearsonr(np.real(sig1[i]), np.real(sig2[i]))
              imag_corr[i], _ = stats.pearsonr(np.imag(sig1[i]), np.imag(sig2[i]))
    return real_corr, imag_corr
              


# Separate the  fourier transform of the signal into different frequency ranges
# Inputs: 
#       sig: the given signal to separate the frequency ranges
#       fd: file directory to extract the sampling frequency 
#       bands: an array which contains lower and upper bounds of different bands
# Outputs:
#       decomposed_bands: the fourier transform of the signal in different bands
def frequencyBandDivision(sig, fd, bands):
    fft = np.abs(np.fft.fft(sig))
    sampling_frequency = common.getRate(fd)
    freqs = np.fft.fftfreq(len(sig),1/sampling_frequency)
    band_mask1 = (freqs >= bands[0,0]) & (freqs < bands[0,1])
    fft1 = fft * band_mask1
    band_mask2 = (freqs >= bands[1,0]) & (freqs < bands[1,1])
    fft2 = fft * band_mask2
    band_mask3 = (freqs >= bands[2,0]) & (freqs < bands[2,1])
    fft3 = fft * band_mask3
    return fft1,fft2,fft3

def TFCalculation(sig,fd,bands,channel=18):
    fft1 = []
    fft2 = []
    fft3 = []
    for i in range(0,18):
        fft1_temp,fft2_temp,fft3_temp = frequencyBandDivision(sig,fd,bands)
        fft1.append(fft1_temp)
        fft2.append(fft2_temp)
        fft3.append(fft3_temp)
    MRISignal1 = fft1[0:16]
    MRISignal2 = fft2[0:16]
    MRISignal3 = fft3[0:16]
    if channel == 18:
        EMISignal1 = fft1[16:18]
        EMISignal2 = fft2[16:18]
        EMISignal3 = fft3[16:18]
    else:
        EMISignal1 = fft1[channel]
        EMISignal2 = fft2[channel]
        EMISignal3 = fft3[channel]
    tf1 = np.zeros((16,2),dtype= np.complex64)
    tf1 = np.dot(MRISignal1,np.linalg.pinv(EMISignal1))
    tf2 = np.zeros((16,2),dtype= np.complex64)
    tf2 = np.dot(MRISignal2,np.linalg.pinv(EMISignal2))
    tf3 = np.zeros((16,2),dtype= np.complex64)
    tf3 = np.dot(MRISignal3,np.linalg.pinv(EMISignal3))
    return tf1,tf2,tf3


def denoising(sig,fd,bands,channel):
    tf1,tf2,tf3 = TFCalculation(sig[0],fd,bands,channel)

    fft1 = []
    fft2 = []
    fft3 = []
    for i in range(0,18):
        fft1_temp,fft2_temp,fft3_temp = frequencyBandDivision(sig[128],fd,bands)
        fft1.append(fft1_temp)
        fft2.append(fft2_temp)
        fft3.append(fft3_temp)
    MRISignal1 = fft1[0:16]
    MRISignal2 = fft2[0:16]
    MRISignal3 = fft3[0:16]
    if channel == 18:
        EMISignal1 = fft1[16:18]
        EMISignal2 = fft2[16:18]
        EMISignal3 = fft3[16:18]
    else:
        EMISignal1 = fft1[channel]
        EMISignal2 = fft2[channel]
        EMISignal3 = fft3[channel]
    denoised = sig[128,0:16]

    pred1 = np.dot(EMISignal1,tf1)
    pred2 = np.dot(EMISignal2,tf2)
    pred3 = np.dot(EMISignal3,tf3)
    pred = np.concatenate((pred1,pred2,pred3),axis=1)
    denoised = denoised - np.fft.ifft(pred)
    return denoised

def mean(sig):
    return np.mean(sig)

def peak(sig):
     return np.max(sig)

def std(sig):
     return np.std(sig, dtype=np.float64)

# before suppression/ after suppression of channel 16/ suppression rate (SR)/ after channel 17/ suppression rate / after channel 16+17/ suppression rate
# mean
# peak
# standard deviation
# PCC
def experiment1DTable(before,sup16,sup17,supComb):
    meanBefore = mean(before)
    peakBefore = peak(before)
    stdBefore = std(before)
    meanSup16 = mean(sup16)
    peakSup16 = peak(sup16)
    stdSup16 = std(sup16)
    SRMeanSup16 = 1 - (meanSup16/meanBefore)
    SRPeakSup16 = 1 - (peakSup16/peakBefore)
    SRStdSup16 = 1 - (stdSup16/stdBefore)
    meanSup17 = mean(sup17)
    peakSup17 = peak(sup17)
    stdSup17 = std(sup17)
    SRMeanSup17 = 1 - (meanSup17/meanBefore)
    SRPeakSup17 = 1 - (peakSup17/peakBefore)
    SRStdSup17 = 1 - (stdSup17/stdBefore)
    meanSupComb = mean(supComb)
    peakSupComb = peak(supComb)
    stdSupComb = std(supComb)
    SRMeanSupComb = 1 - (meanSupComb/meanBefore)
    SRPeakSupComb = 1 - (peakSupComb/peakBefore)
    SRStdSupComb = 1 - (stdSupComb/stdBefore)
    table = PrettyTable(["","Before suppression", "After suppression with Channel 16",
                          "Suppression Rate", "After suppression with Channel 17",
                          "Suppression Rate","After suppression with Channel 16 and 17",
                          "Suppression Rate"])
    table.add_row(["mean",meanBefore,meanSup16,SRMeanSup16,meanSup17,SRMeanSup17,meanSupComb,SRMeanSupComb]) 
    table.add_row(["peak",peakBefore,peakSup16,SRPeakSup16,peakSup17,SRPeakSup17,peakSupComb,SRPeakSupComb]) 
    table.add_row(["standard deviation",stdBefore,stdSup16,SRStdSup16,stdSup17,SRStdSup17,stdSupComb,SRStdSupComb]) 
    print(table)

# step 0: calculate 
# step 1: calculate the pcc of channel 0-15 to 16 and 17
# step 2: calculate transfer factor with frequency band division
# step 3: experiment 1D
def experiment1D(sig,fd):
    # step 1
    real_corr = []
    imag_corr = []
    for i in range(0,16):
        for k in range(16,18):
            real_corr_temp,imag_corr_temp = PCCCalculation(sig[i],sig[k])
            real_corr.append(real_corr_temp)
            imag_corr.append(imag_corr_temp)
    mean_real_corr = np.mean(real_corr,axis=0)
    mean_imag_corr = np.mean(imag_corr,axis=0)
    print(mean_real_corr)
    print(mean_imag_corr)
    count_strong_real_corr = []
    count_strong_imag_corr = []

    for i in range(0,32):
        strong_real_corr = (real_corr[i] > 0.8)
        strong_imag_corr = (imag_corr[i] > 0.8)
        count_strong_real_corr.append(len(strong_real_corr))
        count_strong_imag_corr.append(len(strong_imag_corr))
    
    print(count_strong_real_corr)
    print(count_strong_imag_corr)
    #heat plot
    plt.imshow(real_corr, cmap='viridis', aspect='auto')
    plt.colorbar(label='Intensity')  # Add a color bar
    plt.title("Real part Correlation Heat Map")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

    plt.imshow(imag_corr, cmap='viridis', aspect='auto')
    plt.colorbar(label='Intensity')  # Add a color bar
    plt.title("Imagery part Correlation Heat Map")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

    # step 2
    bands = {[0,5000],[5000,15000],[15000,32000]}
    for i in range(16,18):
        if i == 16:
            sup16 = denoising(sig,fd,bands,i)
        else:
            sup17 = denoising(sig,fd,bands,i)
    
    supcomb = denoising(sig,fd,bands,18)
    experiment1DTable(sig[128,0:16],sup16,sup17,supcomb)


date = '1209'
for mode in ['AM']:
    for type in ['Square','Sine']:
        for trial in ['1','2','3','4','5']:
            #noiseOnly,b = common.readAllAcqs('E:/JiaxingData/EMINoise/1205/FA0Run1.h5')
            #noiseOnly = dp.ConvergeComplexR(dp.SplitComplexR(noiseOnly))
            noiseImage,b = common.readAllAcqs('E:/JiaxingData/EMINoise/'+date+'/'+mode+type+'FA77_'+trial+'.h5')
            noiseImage = dp.ConvergeComplexR(dp.SplitComplexR(noiseImage))
            #baseline,b = common.readAllAcqs('E:/JiaxingData/EMINoise/'+date+'/FA77Baseline.h5')
            #baseline = dp.ConvergeComplexR(dp.SplitComplexR(baseline))
            flip0,b = common.readAllAcqs('E:/JiaxingData/EMINoise/'+date+'/'+mode+type+'FA0_'+trial+'.h5')
            flip0 = dp.ConvergeComplexR(dp.SplitComplexR(flip0))
            noisePre,b = common.readAllAcqs('E:/JiaxingData/EMINoise/'+date+'/'+mode+type+'FA77_'+trial+'.h5',table_name="noise")
            noisePre = dp.ConvergeComplexR(dp.SplitComplexR(noisePre))
            noisePre0,b = common.readAllAcqs('E:/JiaxingData/EMINoise/'+date+'/'+mode+type+'FA0_'+trial+'.h5',table_name="noise")
            noisePre0 = dp.ConvergeComplexR(dp.SplitComplexR(noisePre0))
            
            plotAcqSamples(noiseImage,flip0,noisePre,noisePre0,date,mode,type,trial)

