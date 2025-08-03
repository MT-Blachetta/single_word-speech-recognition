"""Testskript zur Berechnung der Word Error Rate über das Testset."""

import recognizer.dnn_recognizer as rec
import recognizer.hmm as HMM
import recognizer.tools as tools
from keras.models import load_model
import numpy as np
import os

def test_model(datadir, hmm, model_dir, parameters):
    
    model = load_model(model_dir)
    
    x_dirs = []
    for root, dirs, files in os.walk(datadir+'/TEST/wav'):
        for f in files:
            x_dirs.append(os.path.join(root,f))
    
    t_dirs = []
    for root, dirs, files in os.walk(datadir+'/TEST/lab'):
        for f in files:
            t_dirs.append(os.path.join(root,f))
    
    x_dirs.sort()
    t_dirs.sort()
    
    I_ges = 0
    D_ges = 0
    S_ges = 0
    N_ges = 0
    
    for test in range(len(x_dirs)):
        dnnResult = rec.wav_to_posteriors(model,x_dirs[test],parameters)
        word_sequence = hmm.posteriors_to_transcription(dnnResult)
        #print('------------------------------------------------------')
        print('--' * 40)
        print(x_dirs[test])
        labfile = open(t_dirs[test],'r')
        transcript = labfile.read()
        transcript = transcript.lower()
        transcript = transcript.split()
        print('REF: {}'.format(transcript))
        print('OUT: {}'.format(word_sequence))
        N, D, I, S = tools.needlemann_wunsch(transcript, word_sequence)
        print('I: {}   D: {}   S: {}   N: {}'.format(I,D,S,N) )
        I_ges += I
        D_ges += D
        S_ges += S
        N_ges += N
        WER = 100*((D_ges+I_ges+S_ges)/N_ges)
        print('Current total WER: '+str(WER))
        
    return 100*((D_ges+I_ges+S_ges)/N_ges)
        

'''
if __name__ == "__main__":

    # parse arguments
    # data directory, e.g., /media/public/TIDIGITS-ASE
    # call:
    # python uebung10.py <data/dir>
    # e.g., python uebung6.py /media/public/TIDIGITS-ASE
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=str, help='Data dir')
    args = parser.parse_args()
'''

datadir = 'TIDIGITS-ASE'

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

# define a name for the model, e.g., 'dnn'
model_name = 'bestDNN'

# directory for the model
model_dir = os.path.join('exp', model_name + '.h5')

    # test DNN
wer = test_model(datadir, hmm, model_dir, parameters)
print('--' * 40)
print("Total WER: {}".format(wer))