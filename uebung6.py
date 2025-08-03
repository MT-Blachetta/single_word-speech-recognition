"""Übung 6: Training eines DNN-Akustikmodells mit den extrahierten Merkmalen."""

import recognizer.feature_extraction as fe
import recognizer.dnn_recognizer as rec
import recognizer.hmm as HMM
import argparse
import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

def train_model(datadir, hidmm, model_dir, parameters, epochs=3):
  
    test_audio = 'data/TEST-MAN-AH-3O33951A.wav'
    test_data = fe.compute_features_with_context(test_audio,**parameters)
    num_states = hidmm.get_num_states()
    
    model = dnn_model( (test_data.shape[1],test_data.shape[2]), num_states)
    
    x_dirs = []
    for root, dirs, files in os.walk(datadir+'/TRAIN/wav'):
        for f in files:
            x_dirs.append(os.path.join(root,f))
    
    x_new = [e.split('.')[0].split('/')[-1] for e in x_dirs]
       
    
    sampleGenerator = generator(x_new,datadir,hidmm, parameters, epochs)
    model.fit_generator(sampleGenerator, steps_per_epoch = len(x_dirs) ,epochs = epochs)
    
    model.save(model_dir)

if __name__ == "__main__":

    # parse arguments
    # data directory, e.g., /media/public/TIDIGITS-ASE
    # call:
    # python uebung6.py <data/dir>
    # e.g., python uebung6.py /media/public/TIDIGITS-ASE
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=str, help='Data dir')
    args = parser.parse_args()

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

    # number of epoches (not too many for the beginning)
    #epochs = 3

    # define a name for the model, e.g., 'dnn'
    model_name = 'bestDNN'
    # directory for the model
    #model_dir = os.path.join('exp', model_name + '.h5')
    #if not os.path.exists('exp'):
    #    os.makedirs('exp')

    # default HMM
    hmm = HMM.HMM()

    # train DNN
    #train_model(args.datadir, hmm, model_dir, parameters, epochs)
    model = load_model(os.path.join('exp', model_name + '.h5'))

    # get posteriors for test file
    post = rec.wav_to_posteriors(model, 'data/TEST-MAN-AH-3O33951A.wav', parameters)
    plt.imshow(post.transpose(), origin='lower')
    plt.xlabel('Frames')
    plt.ylabel('HMM states')
    plt.colorbar()
    plt.show()