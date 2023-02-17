import recognizer.dnn_recognizer as rec
import recognizer.hmm as hmm
from keras.models import load_model
import os



# Hallo, mit diesem Programm können die den Spracherkenner für englische Ziffern verwenden

# geben sie hier den relativen Pfad zur wav Datei an die gescannt werden soll
inputFile = 'data/TEST-MAN-AH-3O33951A.wav'


model_name = 'bestDNN'
model_dir = os.path.join('exp', model_name + '.h5')
hmm = hmm.HMM()

parameters = {'window_size': 25e-3,
    'hop_size': 10e-3,
    'feature_type': 'MFCC_d_dd',
    'n_filters': 24,
    'fbank_fmin': 0,
    'fbank_fmax': 8000,
    'num_ceps': 13,
    'left_context': 6,
    'right_context': 6}

def wav_to_transcription(wav_filepath,model_dir,hmm,parameters):
    
    model = load_model(model_dir)
    posteriors = rec.wav_to_posteriors(model,wav_filepath,parameters)
    
    return hmm.posteriors_to_transcription(posteriors)

# Die Transcription wird in der Konsole ausgegeben
print ( wav_to_transcription(inputFile,model_dir,hmm,parameters) )


# Hinweis: bestDNN wurde noch nicht ausgebug getestet, versucht auch andere dnn's in den Ordnern