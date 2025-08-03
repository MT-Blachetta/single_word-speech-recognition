# Einzelwort-Spracherkennung

Dieses Projekt demonstriert Schritt für Schritt den Aufbau eines einfachen
Spracherkennungssystems für einzeln gesprochene englische Ziffern. Die
Implementierung orientiert sich an modernen hybriden Systemen aus Deep Neural
Network (DNN) und Hidden-Markov-Modell (HMM) und eignet sich als Lernressource
für die Grundlagen der automatischen Spracherkennung.

## 1. Von der Wellenform zum Merkmal
1. **Signalaufnahme** – Eine WAV-Datei enthält das digitale Audiosignal.
2. **Framing & Fensterung** – `feature_extraction.make_frames` teilt das
   Signal in überlappende Frames und versieht sie mit einem Hamming‑Fenster.
3. **Spektralanalyse** – `feature_extraction.compute_absolute_spectrum`
   berechnet mittels FFT das Betragspektrum jedes Frames.
4. **Mel-Filterbank** – `feature_extraction.get_mel_filters` und
   `apply_mel_filters` bilden das Spektrum auf die Mel‑Skala ab.
5. **MFCCs** – `feature_extraction.compute_cepstrum` erzeugt die
   Mel-Frequency Cepstral Coefficients, optional ergänzt durch
   erste und zweite Ableitung sowie Kontextfenster.

## 2. Akustisches Modell (DNN)
`dnn_recognizer.dnn_model` definiert ein kleines voll
vernetztes Netz, das aus den Merkmalen pro Frame Wahrscheinlichkeiten für die
HMM-Zustände berechnet. `generator` liest Trainingsdaten on‑the‑fly und
`train_model` speichert das Modell als `*.h5` Datei.

## 3. Sprachmodell und Dekodierung
Das HMM in `hmm.py` modelliert die Übergänge zwischen den Lautzuständen der
Ziffern. Der Viterbi‑Algorithmus (`viterbi`) kombiniert die DNN-Ausgaben mit den
Übergangswahrscheinlichkeiten, und `HMM.posteriors_to_transcription` wandelt die
Zustandsfolge in Wörter um.

## 4. Trainings- und Testscripte
Die Dateien `uebung1.py`–`uebung10.py` führen schrittweise durch den
Erkennungsprozess – von der Feature-Berechnung über das DNN-Training bis zur
Bewertung mit der Word Error Rate (`testing.py`).

### Training
```
python uebung6.py <pfad_zum_datensatz>
```
Das trainierte Modell wird im Verzeichnis `exp/` abgelegt.

### Erkennung eines Beispiels
```
python uebung9.py
```
Lädt das gespeicherte Modell und gibt die erkannte Wortsequenz aus.

### Evaluation auf dem Testset
```
python testing.py
```
Berechnet die Word Error Rate über alle Testdateien.

## 5. Projektstruktur
- `recognizer/feature_extraction.py` – komplette MFCC-Pipeline
- `recognizer/dnn_recognizer.py` – DNN-Definition und Training
- `recognizer/hmm.py` – HMM und Viterbi-Dekodierung
- `recognizer/tools.py` – Hilfsfunktionen
- `uebung*.py` – Übungen, die einzelne Schritte demonstrieren
- `testing.py` – automatisierte Auswertung

## 6. Voraussetzungen
- Python 3 mit `numpy`, `scipy`, `keras`, `matplotlib`, `praatio`
- Der TIDIGITS‑Datensatz (oder ein vergleichbarer Korpus) im Ordner `data/`

Viel Spaß beim Experimentieren mit automatischer Spracherkennung!

