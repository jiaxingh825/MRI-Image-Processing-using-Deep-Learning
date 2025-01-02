import numpy as np
import matplotlib.pyplot as plt
import h5py
import common
import DataProcessing as dp

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


# frequency band range: 0-5000 5000-20000 20000-32000
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