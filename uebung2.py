"""Übung 2: Darstellung des Kurzzeit-Frequenzspektrums (STFT).

Das Skript lädt eine Audiodatei, berechnet das STFT und visualisiert das
Spektrum als Bild.
"""

import recognizer.tools as tools
import recognizer.feature_extraction as fe
import numpy as np
import matplotlib as plt


if __name__ == "__main__":
    
    audio_file = 'data/TEST-MAN-AH-3O33951A.wav'
    stft = fe.compute_features(audio_file, window_size=25e-3, hop_size=10e-3, feature_type="STFT")
    # Visualisiere die spektrale Energie jedes Frames
    plt.imshow(np.swapaxes(stft,0,1),origin='lower',aspect='auto')
    plt.ylabel('Frequenzindex')
    plt.xlabel('Frame')
    plt.colorbar()
    plt.show()
