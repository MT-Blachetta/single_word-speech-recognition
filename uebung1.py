import math
import numpy as np
from scipy import signal
import scipy.io.wavfile
import recognizer.tools as tools


if __name__ == "__main__":

    audio_file = 'data/TEST-MAN-AH-3O33951A.wav'
        
    def compute_features(audio_file, window_size=25e-3, hop_size=10e-3, feature_type="STFT", n_filters=24, fbank_fmin=0, fbank_fmax=8000, num_ceps=13)

        sampling_rate, audio_data = scipy.io.wavfile.read(audio_file)
        #duration = len(audio_data)/sampling_rate

        return tools.make_frames(audio_data, sampling_rate, window_size, hop_size)

