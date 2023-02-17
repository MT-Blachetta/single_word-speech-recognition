import recognizer.dnn_recognizer as rec
import recognizer.hmm as HMM
from keras.models import load_model
import numpy as np


if __name__ == "__main__":

    # parameters for the feature extraction
    parameters = {'window_size': 25e-3,
        'hop_size': 10e-3,
        'feature_type': 'MFCC_d_dd',
        'n_filters': 24,
        'fbank_fmin': 0,
        'fbank_fmax': 8000,
        'num_ceps': 13,
        'left_context': 6,
        'right_context': 6}

    # default HMM
    hmm = HMM.HMM()


    # Task 1) load posteriors (will only work with default HMM)
    posteriors = np.load('data/TEST-MAN-AH-3O33951A.npy')

    # calculates transcription from posteriors
    words = hmm.posteriors_to_transcription(posteriors)

    print('OUT: {}'.format(words))  # OUT: ['THREE', 'OH', 'THREE', 'THREE', 'NINE', 'FIVE', 'ONE']   

    # Task 2) use own DNN
    test_audio = 'data/TEST-MAN-AH-3O33951A.wav'

    # load DNN (trained DNN from excercise 6)
    model = load_model('exp/bestDNN.h5')
    testpost = rec.wav_to_posteriors(model, test_audio, parameters)
    wordsB = hmm.posteriors_to_transcription(testpost)
    # TODO run recognizer for test_audio with model and calculate predicted transcription

    print('My own Network OUT: {}'.format(wordsB)) 

 
