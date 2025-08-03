"""Feature extraction utilities for the ASR tutorial.

This module converts waveform audio into feature vectors used for training and
 testing the neural network.  It implements a classic MFCC pipeline including
windowing, spectral analysis, Mel filtering and calculation of delta features.
"""
import math
import numpy as np
from scipy import signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import recognizer.tools as tools
import os
import keras
import keras.models
from keras.models import Sequential
from keras.layers import Dense
from praatio import tgio
import recognizer.hmm as hmm


def make_frames(audio_data, sampling_rate, window_size, hop_size):
    """Slice raw audio into overlapping frames.

    The audio is divided into short segments which are windowed with a Hamming
    function.  Frames are sized to the next power of two to allow an efficient
    FFT in the subsequent processing steps.
    """

    windowSamples = tools.sec_to_samples(window_size, sampling_rate)
    new_window_size = int(math.pow(2, tools.next_pow2(windowSamples)))
    new_hop_size = int(math.ceil(hop_size * sampling_rate))
    data_size = len(audio_data)
    number_of_frames = tools.get_num_frames(data_size, new_window_size, new_hop_size)

    frames = np.zeros(new_window_size * number_of_frames, dtype=np.float)
    hamming_window = np.hamming(new_window_size)
    pad_width = new_window_size
    padded_audio_data = np.pad(np.array(audio_data), (0, pad_width), 'constant', constant_values=0)
    j = 0

    for i in range(number_of_frames):
        frames[i * new_window_size: (i * new_window_size) + new_window_size] = (
            padded_audio_data[j: j + new_window_size] * hamming_window
        )
        j += new_hop_size

    return np.reshape(frames, (number_of_frames, new_window_size))

def compute_absolute_spectrum(frames):
    """Compute the magnitude spectrum for every frame."""

    frameshape = frames.shape
    frequence = int(frameshape[1] / 2)
    mySTFT = np.zeros((frameshape[0], frequence))

    for fr in range(frameshape[0]):
        cpx = scipy.fftpack.fft(frames[fr])
        cpx = np.abs(cpx)
        mySTFT[fr] = cpx[:frequence]

    return mySTFT


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


def get_mel_filters(sampling_rate, freq_idx, n_filters, f_min=0, f_max=8000):
    """Create a Mel filter bank matrix.

    The triangular filters are positioned on the Mel scale and later used to
    compress the linear frequency spectrum into perceptually motivated bands.
    """

    if isinstance(freq_idx, float):
        freq_idx = tools.sec_to_samples(freq_idx, sampling_rate)

    mel_max = tools.hz_to_mel(f_max)
    Ndiv2_1 = freq_idx - 1
    FreqInterval = f_max / Ndiv2_1
    melInterval = mel_max / (n_filters + 1)
    stuezstelle = np.zeros(n_filters + 2, dtype=np.int32)

    for i in range(n_filters + 2):
        stuezstelle[i] = int(round(tools.mel_to_hz(i * melInterval) / FreqInterval))

    Hm = np.zeros((n_filters, freq_idx))
    for m in range(n_filters):
        func_args = (stuezstelle[m], stuezstelle[m + 1], stuezstelle[m + 2])
        Hm[m] = apply_index(left_half_mel, Hm[m], func_args, stuezstelle[m], stuezstelle[m + 1])
        Hm[m] = apply_index(right_half_mel, Hm[m], func_args, stuezstelle[m + 1], stuezstelle[m + 2] + 1)

    return Hm

def apply_mel_filters(abs_spectrum, filterbank):
    """Apply the Mel filter bank to an absolute spectrum."""

    K = abs_spectrum.shape[1]
    mel_spectrum = np.zeros((len(abs_spectrum), len(filterbank)))

    for t in range(len(abs_spectrum)):  # frame index
        for m in range(len(filterbank)):  # mel filter index
            mel_sum = 0
            for k in range(K):  # frequency bin
                mel_sum += filterbank[m][k] * abs_spectrum[t][k]
            mel_spectrum[t][m] = mel_sum

    return mel_spectrum



def compute_cepstrum(mel_spectrum, num_ceps):
    """Calculate MFCCs from a Mel spectrum."""

    mel_spectrum = np.array(mel_spectrum)
    FullCepstrum = np.zeros(mel_spectrum.shape)

    for tau in range(mel_spectrum.shape[0]):
        FullCepstrum[tau] = fft.dct(np.log(mel_spectrum[tau]), norm='ortho')

    cepstrum = np.zeros((mel_spectrum.shape[0], num_ceps))
    for t in range(cepstrum.shape[0]):
        cepstrum[t] = FullCepstrum[t][:num_ceps]

    return cepstrum

def get_delta(x):
    """First order time derivative of cepstral features."""

    d_cep = np.zeros(x.shape)
    d_cep[0] = x[1] - x[0]
    for tau in range(1, x.shape[0] - 1):
        d_cep[tau] = 0.5 * (x[tau + 1] - x[tau - 1])
    d_cep[-1] = x[-1] - x[-2]

    return d_cep

def append_delta(x, delta):
    """Concatenate static and dynamic features."""

    features = np.zeros((x.shape[0], x.shape[1] + delta.shape[1]))
    for t in range(x.shape[0]):
        features[t][0:x.shape[1]] = x[t]
        features[t][x.shape[1]:features.shape[1]] = delta[t]
    return features
        
def add_context(features, left_context, right_context):
    """Append neighbouring frames to provide temporal context."""

    extend_feature = np.zeros((left_context + features.shape[0] + right_context, features.shape[1]))

    for f in range(left_context):
        extend_feature[f] = features[0]
    for f in range(left_context, features.shape[0] + left_context):
        extend_feature[f] = features[f - left_context]
    for f in range(features.shape[0] + left_context, extend_feature.shape[0]):
        extend_feature[f] = features[-1]

    context_feature = np.zeros((features.shape[0], left_context + 1 + right_context, features.shape[1]))

    for f in range(features.shape[0]):
        for k in range(left_context + 1 + right_context):
            context_feature[f][k] = extend_feature[f + k]

    return np.swapaxes(context_feature, 1, 2)


def compute_features(audio_file, window_size=25e-3, hop_size=10e-3, feature_type="STFT", n_filters=24, fbank_fmin=0, fbank_fmax=8000, num_ceps=13):
    """Master function for feature computation.

    Depending on ``feature_type`` this returns plain STFT spectra, filter-bank
    energies or MFCC features with optional derivatives.
    """

    sampling_rate, audio_data = scipy.io.wavfile.read(audio_file)

    frames = make_frames(audio_data, sampling_rate, window_size, hop_size)
    mySTFT = compute_absolute_spectrum(frames)
    frequence = int(mySTFT.shape[1])

    mbank = get_mel_filters(sampling_rate, frequence, n_filters, fbank_fmin, fbank_fmax)
    mel_spectrum = apply_mel_filters(mySTFT, mbank)

    cepstrum = compute_cepstrum(mel_spectrum, num_ceps)
    d_cep = get_delta(cepstrum)
    dd_cep = get_delta(d_cep)

    if feature_type == "STFT":
        return 10 * np.log10(mySTFT)
    elif feature_type == "FBANK":
        return np.log(mel_spectrum)
    elif feature_type == "MFCC":
        return cepstrum
    elif feature_type == "MFCC_d":
        return append_delta(cepstrum, d_cep)
    elif feature_type == "MFCC_d_dd":
        return append_delta(append_delta(cepstrum, d_cep), dd_cep)
    else:
        print("ERROR: No valid feature type, returning None ")

def compute_features_with_context(audio_file, window_size=25e-3, hop_size=10e-3, feature_type='MFCC_d_dd', n_filters=24, fbank_fmin=0, fbank_fmax=8000, num_ceps=13, left_context=4, right_context=4):
    """Convenience wrapper that also adds temporal context."""

    features = compute_features(audio_file, window_size, hop_size, feature_type, n_filters, fbank_fmin, fbank_fmax, num_ceps)

    return add_context(features, left_context, right_context)
