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
import random
import recognizer.hmm as hmm
import recognizer.feature_extraction as fe



def wav_to_posteriors(model, audio_file, parameters): 
    
    audio_data = fe.compute_features_with_context(audio_file,**parameters)
    
    return model.predict(audio_data)



def generator(x_dirs, datadir, hidmm, parameters, epoch): 
    
    end = len(x_dirs);
    sampling_rate, audio_data = scipy.io.wavfile.read(datadir+'/TRAIN/wav/'+x_dirs[0]+'.wav')
    win_siz = int( math.pow(2,tools.next_pow2( int ( math.ceil( parameters['window_size'] * sampling_rate) ) )) )
    hop_siz = int( math.ceil(parameters['hop_size']*sampling_rate) )
    
    while True:        
        index = 0;
        random.shuffle(x_dirs)
            
        while index < end:
            x_data = fe.compute_features_with_context(datadir+'/TRAIN/wav/'+x_dirs[index]+'.wav',**parameters);
            y_data = tools.praat_file_to_target(datadir+'/TRAIN/TextGrid/'+x_dirs[index]+'.TextGrid', sampling_rate, win_siz , hop_siz , hidmm)
            index += 1
            yield (x_data, y_data)


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



def dnn_model(input_shape, output_shape):

    model = Sequential()
    model.add(keras.layers.Flatten(input_shape = input_shape) )
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(output_shape, activation="softmax"))
    model.compile(loss='categorical_crossentropy',optimizer="adam", metrics=["accuracy"])

    return model