import math
import numpy as np
from scipy import signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import recognizer.tools as tools
import recognizer.feature_extraction as fe

if __name__ == "__main__":

    def hz_to_mel(x):
        return 2595*math.log10(1+(x/700));

    def mel_to_hz(y):
        return 700*math.pow(10,y/2595) - 700;


    # HILFSFUNKTIONEN FÜR MEL-FILTERBANK BERECHNUNG
    def left_half_mel(k,arguments):
        begin = arguments[0]; 
        mid = arguments[1];
        end = arguments[2];
        return (2*k - 2*begin)/((end-begin)*(mid-begin));

    def right_half_mel(k,arguments):
        begin = arguments[0]; 
        mid = arguments[1];
        end = arguments[2];
        return (2*end - 2*k)/((end-begin)*(end-mid));

    def apply_index(func,seq,arg,start,end):
        for k in range(start,end):
            seq[k] = func(k,arg)
        return seq


    def get_mel_filters(sampling_rate,freq_idx,n_filters,f_min=0,f_max=8000):

        if type(freq_idx)==float:
            freq_idx = sec_to_samples(freq_idx, sampling_rate)

        mel_max = hz_to_mel(8000);
        test_hz = mel_to_hz(mel_max);


        Ndiv2_1 = freq_idx - 1; 
        lastDFTindex = int(Ndiv2_1);

        FreqInterval = f_max / Ndiv2_1 ;


        melInterval = mel_max/(n_filters+1)

        stuezstelle = np.zeros(n_filters + 2, dtype = np.int32);

        for i in range(n_filters+2):
            stuezstelle[i] = int(round(mel_to_hz(i*melInterval)/FreqInterval)); # for the index calculation

        Hm = np.zeros( (n_filters,freq_idx) );
        for m in range(n_filters):

            func_args = (stuezstelle[m],stuezstelle[m+1],stuezstelle[m+2])
            Hm[m] = apply_index(left_half_mel,Hm[m],func_args,stuezstelle[m],stuezstelle[m+1])
            Hm[m] = apply_index(right_half_mel,Hm[m],func_args,stuezstelle[m+1],stuezstelle[m+2]+1) 

        return Hm

    def apply_mel_filters(abs_spectrum, filterbank):

        K = abs_spectrum.shape[1];
        mel_spectrum = np.zeros( (len(abs_spectrum),len(filterbank)) );

        for t in range ( len(abs_spectrum) ): # t =: frame index
            for m in range(len(filterbank)): # m =: mel filter index
                mel_sum = 0;
                for k in range(K): # k =: frequence domain index
                    mel_sum += filterbank[m][k]*abs_spectrum[t][k];
                mel_spectrum[t][m] = mel_sum;


        return mel_spectrum

    def compute_features(audio_file, window_size=25e-3, hop_size=10e-3, feature_type="STFT", n_filters=24, fbank_fmin=0, fbank_fmax=8000, num_ceps=13):

        sampling_rate, audio_data = scipy.io.wavfile.read(audio_file)

        duration = len(audio_data)/sampling_rate

        frames = fe.make_frames(audio_data, sampling_rate, window_size, hop_size)
        mySTFT = fe.compute_absolute_spectrum(frames)
        frequence = int(mySTFT.shape[1])

        mbank = get_mel_filters(sampling_rate,frequence,n_filters)
        mel_spectrum = apply_mel_filters(mySTFT,mbank)



        if feature_type == "STFT":

            print('STFT selected')

            return  10 * np.log10(mySTFT)

        elif feature_type == "FBANK":
            print('FBANK selected')
            return (mbank,np.log(mel_spectrum))

        else:
            print("ERROR: No valid feature type, returning None ")





    file = 'data/TEST-MAN-AH-3O33951A.wav'  
    filters = 24
    
    mel = compute_features(file,feature_type='FBANK',n_filters=filters)
    
    melBank = mel[0]
    fig = plt.figure(figsize=(25,10))

    for b in range(filters):
        plt.plot(melBank[b])

    fig.show()
    
    plt.imshow(np.swapaxes(mel[1],0,1),origin='lower',aspect='auto')
    plt.ylabel('Mel Frequenzindex')
    plt.xlabel('Frame')
    plt.colorbar()
    plt.show()