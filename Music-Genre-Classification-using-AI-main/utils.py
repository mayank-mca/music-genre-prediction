import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import numpy as np
import cv2
tf.compat.v1.disable_eager_execution()

labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
img_size = 256

def spectrogram(path,base_path):
    filepath,_ = os.path.splitext(path)
    filename = os.path.basename(filepath)
    x , sr = librosa.load(path)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb)
    savepath = f'{base_path}{filename.rsplit(".")[0]}.png'
    plt.title(f"Spectrogram {filename}")
    plt.savefig(savepath)
    plt.close()
    return f'/static/spectrograms/{filename.rsplit(".")[0]}.png'


def wavelet(path,base_path):
    filepath,_ = os.path.splitext(path)
    filename = os.path.basename(filepath)
    x , sr = librosa.load(path)
    librosa.display.waveshow(x)
    savepath =f'{base_path}{filename.rsplit(".")[0]}.png'
    plt.title(f"Wavelet {filename}")
    plt.savefig(savepath)
    plt.close()
    return f'/static/wavelets/{filename.rsplit(".")[0]}.png'


def classify(spectrogram,wavelet,spath):
    print(os.path.join(os.getcwd(),wavelet))
    print(os.path.join(os.getcwd(),spectrogram))
    modelpath = os.path.join(os.getcwd(),'models','500_epoch_simple_lr.cpkt')
    img = os.path.join(spath,os.path.basename(spectrogram))
    print("--------------------",img)
    img_arr = cv2.imread(img)[...,::-1] #convert BGR to RGB format
    data = cv2.resize(img_arr, (img_size, img_size))
    model = Sequential()
    model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(256,256,3)))
    model.add(MaxPool2D())

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(10, activation="softmax"))
    opt = Adam(lr=0.0001)
    model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])
    model.load_weights(modelpath)
    pred = model.predict(np.array([data]))
    pred = pred.reshape(1,-1)[0]
    return pred
