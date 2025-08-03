"""Übung 5: Einführung in Keras anhand des Fashion-MNIST Datensatzes."""

from keras.datasets import fashion_mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    ((trainX, trainY),(testX,testY)) = fashion_mnist.load_data()
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0
    
    #Label-Vektor generieren
    trainY = np_utils.to_categorical(trainY,10)
    testY = np_utils.to_categorical(testY,10)
    
    # Labelnamen
    label_names = ["top", "trouser", "pullover", "dress", "coat", "sandal",
                  "shirt", "sneaker","bag","ankle boot"]
    
    # Anordnung der Daten als Bild
    trainX = trainX.reshape((trainX.shape[0],28,28,1))
    testX = testX.reshape((testX.shape[0],28,28,1))
    
    return ( {"train": {"image":trainX, "label":trainY}, "test": {"image": testX, "label": testY}, "labels": label_names })
	
	
def get_ff_net():
    
    # Leeres Model anlegen
    model = Sequential()
    
    # Lagen hinzufuegen
    model.add(keras.layers.Flatten(input_shape=(28, 28, 1)))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, name="label", activation="softmax"))
    
    # Optimizer hinzufuegen
    model.compile(loss={'label': 'categorical_crossentropy'},optimizer="adam", metrics=["accuracy"])
    
    # Model anzeigen
    model.summary()
    
    return model
	

epochs = 5
model = get_ff_net()
data = get_data()
loss_hist = model.fit(x=data["train"]["image"],y=data["train"]["label"],validation_data=(data["test"]["image"], data["test"]["label"]),epochs=epochs)

# Als h5-Datei speichern

model.save('fmnist_ff.h5')

# Darstellung von Kostenfunktion und Accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), loss_hist.history["loss"], label="Kosten_Trainingsdaten")
plt.plot(np.arange(0, epochs), loss_hist.history["val_loss"], label="Kosten_Validierungsdaten")
plt.plot(np.arange(0, epochs), loss_hist.history["acc"], label="Trainings-Accuracy")
plt.plot(np.arange(0, epochs), loss_hist.history["val_acc"], label="Validierungs-Accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")