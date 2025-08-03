"""Übung 4: Berechnung von MFCCs und deren Ableitungen."""

import math
import numpy as np
from scipy import signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import recognizer.tools as tools
import recognizer.feature_extraction as fe

if __name__ == "__main__":
    
    def compute_cepstrum(mel_spectrum, num_ceps):
        mel_spectrum = np.array(mel_spectrum)
        FullCepstrum = np.zeros(mel_spectrum.shape)

        for tau in range(mel_spectrum.shape[0]):
            FullCepstrum[tau] = fft.dct(np.log(mel_spectrum[tau]),norm='ortho')
            
        cepstrum = np.zeros( (mel_spectrum.shape[0],num_ceps) )
            
        for t in range(cepstrum.shape[0]):
            cepstrum[t] = FullCepstrum[t][:num_ceps]
            

        return cepstrum

    def get_delta(x):
        
        d_cep = np.zeros(x.shape)
        d_cep[0] = x[1] - x[0]
        for tau in range(1,x.shape[0]-1):
            d_cep[tau] = 0.5*(x[tau+1] - x[tau-1])
        d_cep[-1] = x[-1] - x[-2]
        
        return d_cep

    def append_delta(x,delta):
        features = np.zeros((x.shape[0],x.shape[1]+delta.shape[1]))
        for t in range(x.shape[0]):
            features[t][0:x.shape[1]] = x[t]
            features[t][x.shape[1]:features.shape[1]] = delta[t]
        return features
            

    def compute_features(audio_file, window_size=25e-3, hop_size=10e-3, feature_type="STFT", n_filters=24, fbank_fmin=0, fbank_fmax=8000, num_ceps=13):

        sampling_rate, audio_data = scipy.io.wavfile.read(audio_file)
        duration = len(audio_data)/sampling_rate

        frames = tools.make_frames(audio_data, sampling_rate, window_size, hop_size)
        mySTFT = fe.compute_absolute_spectrum(frames)
        frequence = int(mySTFT.shape[1])
        
        mbank = fe.get_mel_filters(sampling_rate,frequence,n_filters)
        mel_spectrum = fe.apply_mel_filters(mySTFT,mbank)
        
        cepstrum = compute_cepstrum(mel_spectrum,num_ceps)
     
        d_cep = get_delta(cepstrum)
        dd_cep = get_delta(d_cep)
           

        if feature_type == "STFT":
            
            return  10 * np.log10(mySTFT)

        
        elif feature_type == "FBANK":

            return np.log(mel_spectrum)
        
        elif feature_type == "MFCC":

            return cepstrum
            
        elif feature_type == "MFCC_d":

            return append_delta(cepstrum,d_cep)
            
        elif feature_type == "MFCC_d_dd":

            return append_delta( append_delta(cepstrum,d_cep), dd_cep )
            
        else:
            print("ERROR: No valid feature type, returning None ")


    file = 'data/TEST-MAN-AH-3O33951A.wav'
    num_ceps = 13
    ceps = fe.compute_features(file,feature_type='MFCC_d_dd',num_ceps=num_ceps)
    plt.imshow(np.swapaxes(ceps,0,1),origin='lower',aspect='auto')
    plt.ylabel('Cepstral Koeffizient')
    plt.xlabel('Frame')
    plt.colorbar()
    plt.show()