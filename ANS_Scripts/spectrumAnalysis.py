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
    i2 = np.random.randint(0,16)
    
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
    plt.xlim(9000, 11000)
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
              


def TFCalculation2(sig,channel=18):
    MRISignal = sig[0:16]
    if channel == 18:
        EMISignal = sig[16:18]
        tf = np.zeros((16,2),dtype= np.complex64)
        tf = np.dot(MRISignal,np.linalg.pinv(EMISignal))
    else:
        EMISignal = sig[channel]
        tf = np.zeros((16,1),dtype= np.complex64)
        tf = np.dot(MRISignal,np.linalg.pinv(EMISignal.reshape(1,len(EMISignal))))
    return tf

def denoising2(sig,channel1,channel2=18):
    tf = TFCalculation2(sig[0], channel2)
    denoised = sig[channel1,0:16]
    if channel2 == 18:
        EMISignal = sig[channel1,16:18]
        pred = np.dot(tf,EMISignal)
        denoised = denoised - pred
    else:
        EMISignal = sig[channel1,channel2]
        pred = np.dot(tf,EMISignal.reshape(1,512))
        denoised = denoised - pred
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
                          "Suppression Rate1", "After suppression with Channel 17",
                          "Suppression Rate2","After suppression with Channel 16 and 17",
                          "Suppression Rate3"])
    table.add_row(["mean",meanBefore,meanSup16,SRMeanSup16,meanSup17,SRMeanSup17,meanSupComb,SRMeanSupComb]) 
    table.add_row(["peak",peakBefore,peakSup16,SRPeakSup16,peakSup17,SRPeakSup17,peakSupComb,SRPeakSupComb]) 
    table.add_row(["standard deviation",stdBefore,stdSup16,SRStdSup16,stdSup17,SRStdSup17,stdSupComb,SRStdSupComb]) 
    print(table)

# step 0: calculate 
# step 1: calculate the pcc of channel 0-15 to 16 and 17
# step 2: calculate transfer factor with frequency band division
# step 3: experiment 1D
def experiment1D(sig):

    # step2
    channel1 = 128
    sup16 = denoising2(sig,channel1,16)
    sup17 = denoising2(sig,channel1,17)
    supcomb = denoising2(sig,channel1,18)
    # step3
    experiment1DTable(sig[channel1,0:16],sup16,sup17,supcomb)

# frequency band range: 0-5000 5000-20000 20000-32000
#imag_FA0SineRun1
#real_FA0SineRun1
date = '1209'
for mode in ['AM']:
    for type in ['Square']:
        for trial in ['1','2','3','4','5']:
            #noiseOnly,b = common.readAllAcqs('E:/JiaxingData/EMINoise/1205/FA0Run1.h5')
            #noiseOnly = dp.ConvergeComplexR(dp.SplitComplexR(noiseOnly))
            #noiseImage,b = common.readAllAcqs('E:/JiaxingData/EMINoise/'+date+'/'+mode+type+'FA77_'+trial+'.h5')
            #noiseImage = dp.ConvergeComplexR(dp.SplitComplexR(noiseImage))
            #baseline,b = common.readAllAcqs('E:/JiaxingData/EMINoise/'+date+'/FA77Baseline.h5')
            #baseline = dp.ConvergeComplexR(dp.SplitComplexR(baseline))
            flip0,b = common.readAllAcqs('C:/JiaxingData/EMINoise/'+date+'/0115/'+mode+type+'FA0_'+trial+'.h5')
            flip0 = dp.ConvergeComplexR(dp.SplitComplexR(flip0))
            #noisePre,b = common.readAllAcqs('E:/JiaxingData/EMINoise/'+date+'/'+mode+type+'FA77_'+trial+'.h5',table_name="noise")
            #noisePre = dp.ConvergeComplexR(dp.SplitComplexR(noisePre))
            #noisePre0,b = common.readAllAcqs('E:/JiaxingData/EMINoise/'+date+'/'+mode+type+'FA0_'+trial+'.h5',table_name="noise")
            #noisePre0 = dp.ConvergeComplexR(dp.SplitComplexR(noisePre0))
            
            #plotAcqSamples(noiseImage,flip0,noisePre,noisePre0,date,mode,type,trial)
            experiment1D(flip0)

