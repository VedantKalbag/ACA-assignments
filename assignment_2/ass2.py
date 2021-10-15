import numpy as np
import scipy.signal
import os
import math
from matplotlib import pyplot as plt 

# blockSize=1024
# hopSize=512
# fs=44100

from scipy.io.wavfile import read as wavread
def ToolReadAudio(cAudioFilePath):    
    [samplerate, x] = wavread(cAudioFilePath)    
    if x.dtype == 'float32':        
        audio = x    
    else:        
        # change range to [-1,1)        
        if x.dtype == 'uint8':            
            nbits = 8        
        elif x.dtype == 'int16':            
            nbits = 16        
        elif x.dtype == 'int32':            
            nbits = 32        
        audio = x / float(2**(nbits - 1))    
        # special case of unsigned format    
    if x.dtype == 'uint8':        
        audio = audio - 1.    
    return (samplerate, audio)

# BLOCK AUDIO
def block_audio(x,blockSize,hopSize,fs):    
    # allocate memory    
    numBlocks = math.ceil(x.size / hopSize)    
    xb = np.zeros([numBlocks, blockSize])    
    # compute time stamps    
    t = (np.arange(0, numBlocks) * hopSize) / fs   
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)    
    for n in range(0, numBlocks):        
        i_start = n * hopSize        
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])        
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]    
    return (xb,t)

def calc_stft(xb,fs=44100):
    stft = np.zeros((xb.shape[0],(int(xb[0].shape[0]/2)+1)))
    freqs = np.zeros((xb.shape[0],(int(xb[0].shape[0]/2)+1)))
    window = np.hanning(xb[0].shape[0])
    for i in range(xb.shape[0]):
        block= xb[i]
        # Apply Window to the block
        windowed_block = window * block 
        # Pad zeros equal to length of the windowed block at the end
        #windowed_block = np.append(windowed_block,np.zeros(block.shape[0]))
        stft_blk = np.fft.fft(windowed_block)
        #stft_blk = np.fft.rfft(windowed_block)
        freq=np.fft.fftfreq(block.size,1/fs)
        freqs[i]=freq[:int(block.size/2)+1]
        stft_blk = np.abs(stft_blk)
        #stft_block = stft_blk[int((stft_blk.shape[0])/2):]
        stft_block = stft_blk[:int(((stft_blk.shape[0])/2)+1)]
        stft_db = 10*np.log10(stft_block) # IS THIS CORRECT??
        stft[i]=stft_block#stft_db
    stft = np.array(stft)
    freqs=np.array(freqs)
    return stft,freqs

# Q1-Spectral Centroid

# Q1-RMS
def extract_rms(xb):
    rms = np.zeros(xb.shape[0])
    for i in range(xb.shape[0]):
        block = xb[i]
        r = np.sqrt(np.sum(block**2)/xb.shape[0])
        if r <= 0.00001: # Done to handle case when rms is 0 (for a block of all zeros)
            r = 0.00001
        #rms.append(r)
        rms[i] = r
    #rms=np.array(rms)
    return 20*np.log10(rms)#rms,20*np.log10(rms)

#Q1-ZCR
def extract_zerocrossingrate(xb):
    zcr= np.zeros(xb.shape[0])
    for i in range(xb.shape[0]):
        block = xb[i]
        zero_crossings = np.sum(np.abs(np.diff(np.sign(block)))) / block.shape[0] #np.nonzero(np.diff(block > 0))[0].size
        #zcr.append(zero_crossings)
        zcr[i] = zero_crossings 
    #zcr = np.array(zcr)
    return zcr

#Q1-Spectral Crest
def extract_spectral_crest(xb):
    crest = np.zeros(xb.shape[0])
    stft,freqs = calc_stft(xb,44100)
    for i in range(stft.shape[0]):
        #crest.append((np.max(stft[i])/np.sum(stft[i])))
        crest[i] = np.max(stft[i])/np.sum(stft[i])
    #crest = np.array(crest)
    return crest

#Q1-Spectral Flux
def extract_spectral_flux(xb):
    num_blocks=xb.shape[0] 
    blockSize = xb.shape[1]
    spectral_flux = np.zeros(num_blocks)
    stft,freqs = calc_stft(xb,44100)
    fft_len= stft.shape[1]
    n = 0
    k = 0
    for n in range(1,num_blocks):
        sum_flux = 0
        for k in range(fft_len):
            f = (abs(stft[n, k]) - abs(stft[n-1, k]))**2
            sum_flux += f
        flux = np.sqrt(sum_flux)/((blockSize/2)+1)
        spectral_flux[n] = flux #first flux value will be 0
    return spectral_flux


#Q2-Extract Features
def extract_features(x, blockSize, hopSize, fs):
    xb,ts = block_audio(x,blockSize,hopSize,fs)
    features=np.zeros((5,xb.shape[0]))
    features[0] = extract_spectral_centroid(xb,fs)
    features[1] = extract_rms(xb)
    features[2] = extract_zerocrossingrate(xb)
    features[3] = extract_spectral_crest(xb)
    features[4] = extract_spectral_flux(xb)
    return features

#Q3-Aggregate Features
def aggregate_feature_per_file(features):
    agg_features = np.zeros((2*features.shape[0],1))
    agg_features[0]= np.mean(features[0])
    agg_features[1]= np.std(features[0])
    agg_features[2]= np.mean(features[1])
    agg_features[3]= np.std(features[1])
    agg_features[4]= np.mean(features[2])
    agg_features[5]= np.std(features[2])
    agg_features[6]= np.mean(features[3])
    agg_features[7]= np.std(features[3])
    agg_features[8]= np.mean(features[4])
    agg_features[9]= np.std(features[4])
    return agg_features

#Q4- Get Feature Data
def get_feature_data(path, blockSize, hopSize):
    N=0
    i=0
    for file_name in os.listdir(path):
        if file_name.endswith(".wav"):
            N+=1
    featureData=np.zeros((10,N))
    for file_name in os.listdir(path):
        if file_name.endswith(".wav"):
            #print(file_name)
            fs,x = ToolReadAudio(path+file_name)
            features = extract_features(x,blockSize,hopSize,fs)
            aggFeatures = aggregate_feature_per_file(features)
            for j in range(10):
                featureData[j][i] = aggFeatures[j]
            i+=1
    return featureData

# B1- Normalization
def normalize_zscore(featureData):
    normalized_matrix = np.zeros((featureData.shape[0],featureData.shape[1]))
    for i in range(featureData.shape[0]):
        std = np.std(featureData[i])
        mean = np.mean(featureData[i])
        normalized_matrix[i] = (featureData[i]-mean)/std
    return normalized_matrix