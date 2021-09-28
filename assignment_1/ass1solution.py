import numpy as np
import scipy.signal
import scipy.io.wavfile
import scipy
import os
from matplotlib import pyplot as plt 

def block_audio(x,blockSize,hopSize,fs):
    # Implement a function block_audio(x,blockSize,hopSize,fs) which returns
    # a matrix xb (dimension NumOfBlocks X blockSize) and a vector timeInSec (dimension NumOfBlocks) 
    # for blocking the input audio signal into overlapping blocks. timeInSec will refer to the start time of each block. 
    # Do not use any helper functions such as stride.
    #print("block function")
    #Returns a matrix [numBlocks * blockSize] and starttimeinsecs [numBlocks]
    a=1
    file_length=x.shape[0]
    numBlocks=np.ceil(file_length/hopSize).astype(int)
    #numBlocks = np.floor((file_length-blockSize)/hopSize + 1).astype(int) # throws error if not cast as int (PUT MORE DETAILED COMMENTS)
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
    # Implement a function comp_acf(inputVector, bIsNormalized) 
    # which computes the autocorrelation function and returns the non-redundant (right) part of the result r. 
    # Normalization is controlled by parameter bIsNormalized. 
    # Do not use any helper functions such as xcorr.
    #print("comp_acf")
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
                corr=corr/np.sqrt(np.dot(block,block) * np.dot(y,y))
                #value = value/np.sqrt(np.matmul(frame, frame)*np.matmul(inputVector, inputVector))
            #print(corr)
            correl.append(corr) # ADD THE SHIFTED DOT PRODUCT TO THE CORRELATION ARRAY FOR EACH BLOCK
        #print(len(correl)
            #plt.plot(np.array(correl))
        c.append(np.array(correl)) # ADD TO LIST WHEN THE BLOCK HAS BEEN FULLY COMPUTED 

    return np.array(c) # OUTPUT NEEDS TO BE AN ARRAY OF CORRELATION ARRAYS
        
def get_f0_from_acf (r, fs):
    # Implement a function get_f0_from_acf (r, fs) that takes the output of comp_acf 
    # and computes and returns the fundamental frequency f0 of that block in Hz 
    # (Textbook ref: 7.3.3.2).
    #print("get_f0")
    f=[]
    p,_=scipy.signal.find_peaks(r) 
    #print(p[0])
    calculated_ts = p[np.argmax(r[p])]
    #calculated_ts=p[0] # THIS METHOD WILL BREAK FOR REAL WORLD SIGNALS
    # To fix this for real world signals, iterate through p, and find the value of r at each of these sample numbers.
    # The max of this value will correspond to the sample number for the fundamental frequency
    #if calculated_ts == 0:
    #   print(r)
    #   print(p)
    #   for ele in p:
    #      print(r[ele],ele)
    #print(calculated_ts)
    f_calc=fs/calculated_ts
    f.append(f_calc)
    #plt.plot(r)
    #plt.plot(p,r[p],"x")
    return np.array(f)

def track_pitch_acf(x,blockSize,hopSize,fs):
    # Implement a 'main' function track_pitch_acf(x,blockSize,hopSize,fs) 
    # that calls the three functions above and returns two vectors f0 and timeInSec.
    #print("track_pitch")
    a,ts = block_audio(x,blockSize,hopSize,fs)
    #print(a.shape)
    #print(ts.shape)
    f=[]
    c = comp_acf(a)
    #f_calc = get_f0_from_acf(c,fs)

    for blk in range(0,c.shape[0]):
        #print(blk)
        f_calc = get_f0_from_acf(c[blk],fs)
        f.append(f_calc)
        #print(f_calc)
    f0=np.array(f)
    #print(f0.shape)
    return f0,ts

def convert_freq2midi(freqInHz):
    # Implement a function convert_freq2midi(freqInHz) that returns a variable pitchInMIDI of the same dimension as freqInHz. 
    # Note that the dimension of freqInHz can be a scalar, a vector, or a matrix. 
    # The conversion is described in Textbook Section: 7.2.3. Assume f(A4) = 440Hz.
    fA4=440
    midi_freq=[]
    for f in freqInHz:
        if f == 0:
            midi_freq.append(0)
        else:
            m = 69 + np.log2(f/fA4)
            midi_freq.append(m)
    return np.array(midi_freq)

def eval_pitchtrack(estimateInHz, groundtruthInHz):
    # Implement a function eval_pitchtrack(estimateInHz, groundtruthInHz) that computes the 
    # RMS of the error in Cent (Textbook Section: 7.2.3)
    # in the pitch domain (not frequency) and returns this as errCentRms. 
    # Note: exclude blocks with annotation = 0

    est_pitch = convert_freq2midi(estimateInHz)
    ground_pitch = convert_freq2midi(groundtruthInHz)
    error_cents = [] 

    for i in range(0,est_pitch.shape[0]):
        if ground_pitch[i] != 0:
            error_cents.append(100 * (ground_pitch[i] - est_pitch[i]))
        else:
            error_cents.append(0)
    l = len(error_cents)
    # TRY TO VECTORIZE RMS CALCULATION 
    square_sum=0
    for error in error_cents:
        #print(error)
        square_sum = square_sum + (error**2)
    return np.sqrt(square_sum/l)

def run_evaluation (complete_path_to_data_folder):
    errorcents={}
    files=0
    errCentRms = 0
    for file_name in os.listdir(complete_path_to_data_folder):
        if file_name.endswith(".wav"):
            files = files+1
            name=file_name[:-4]
            print(name)
            #print(loc+name+'.wav')
            #print(loc+name+'.f0.Corrected.txt')
            sr,x = scipy.io.wavfile.read(complete_path_to_data_folder+name+'.wav')

            lut = np.loadtxt(complete_path_to_data_folder+name+'.f0.Corrected.txt')
            onset_seconds = lut[:,1]
            duration_seconds = lut[:,1]
            pitch_frequency = lut[:,2]
            quantized_frequency = lut[:,3]

            hopSize = np.ceil(x.shape[0]/duration_seconds.shape[0]).astype(int)
            blockSize = 2 * hopSize

            f0,ts = track_pitch_acf(x,blockSize,hopSize,sr)
            err = eval_pitchtrack(f0,pitch_frequency)
            
            errorcents[name] = err
            errCentRms = errCentRms + (err ** 2)
    errCentRms = np.sqrt(errCentRms/files)
    return errCentRms
errCentRms = run_evaluation('assignment_1/trainData/')
print(errCentRms)