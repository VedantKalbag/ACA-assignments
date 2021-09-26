import numpy as np
import scipy 
import os
from matplotlib import pyplot as plt 

def block_audio(x,blockSize,hopSize,fs):
    #Returns a matrix [numblocks * blocksize] and starttimeinsecs
    a=1
    file_length=x.shape[0]
    # assuming last block does not end at an integral multiple of hop length, ignore the last block, and add 1 to the block count
    # throws error if not cast as int
    #  file length / block size is number of blocks 
    numBlocks=np.ceil(file_length/hopSize).astype(int)
    #numBlocks = np.floor((file_length-blockSize)/hopSize + 1).astype(int) 
    numZeroes= hopSize-(file_length%hopSize)# file_length-[(hopSize*numBlocks) ]
    x=np.append(x,np.zeros(numZeroes))
    i=0
    lst=[]
    ts=[]
    for blk in range(0,numBlocks):
        if i == 0:
            start = 0
            i=i+1
        end = start + blockSize
        lst.append(x[start:end])
        #print(start)
        ts.append(start/fs)
        start = start + hopSize
        #end = end + hopSize
        out = np.array(lst)
        time = np.array(ts)
    return(out,time)

def comp_acf(inputVector, bIsNormalized=False):
    c=[]
    #plt.figure()
    # print(inputVector)
    for block in inputVector:
        #print(block)
        z=np.zeros(block.shape[0])
        x=np.append(block,z)
        start=0
        #print(block)
        correl=[]

        for i in range(0,block.shape[0]):# Offset start of x by i samples
            start = i
            end = start + block.shape[0]
            #print(f"{start},{end}")
            #print(f"start - {start} end - {end}")
            y=x[start:end] #x[start:end]
            #print(y)
            corr = np.dot(y,block)
            if bIsNormalized:
                corr=corr/(np.sqrt(block,block) * np.sqrt(y,y))
                #value = value/np.sqrt(np.matmul(frame, frame)*np.matmul(inputVector, inputVector))
            #print(corr)
            correl.append(corr) # ADD THE SHIFTED DOT PRODUCT TO THE CORRELATION ARRAY FOR EACH BLOCK
        #print(len(correl)
            #plt.plot(np.array(correl))
        c.append(np.array(correl)) # ADD TO LIST WHEN THE BLOCK HAS BEEN FULLY COMPUTED 

    return np.array(c) # OUTPUT NEEDS TO BE AN ARRAY OF CORRELATION ARRAYS
        
def get_f0_from_acf (r, fs):
    f=[]
    p,_=scipy.signal.find_peaks(r)
    #print(p[0])
    calculated_ts=p[0]
    f_calc=fs/calculated_ts
    f.append(f_calc)
    return np.array(f)

def track_pitch_acf(x,blockSize,hopSize,fs):
    a,ts = block_audio(x,blockSize,hopSize,fs)
    #print(a.shape)
    #print(ts.shape)
    f=[]
    c = comp_acf(a)
    #f_calc = get_f0_from_acf(c,fs)
    for blk in c:
        #print(blk)
        f_calc = get_f0_from_acf(blk,fs)
        f.append(f_calc)
        #print(f_calc)
    f0=np.array(f)
    #print(f0.shape)
    return f0,ts

