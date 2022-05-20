import os

# Data imports
import tensorflow as tf
from typing import Iterator, List, Union, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow imports
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import History
import tensorflow_addons as tfa

import data_pipeline as dp
import numpy as np
import os
import keras
import random
import cv2
import math
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout

from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

def plot_results(history: History):
    """plots the training history of the model"""
    #adapted from my COMP3888 Capstone Project KoalaAI plotting summary code
    #plot binary accuracy
    plt.plot(history.history["binary_accuracy"])
    plt.plot(history.history["val_binary_accuracy"])
    plt.title('Model Average Accuracy (Binary)')
    plt.ylabel('binary accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("model_accuracy.png")
    plt.show()
    #plot history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("model_loss.png")
    plt.show()

def dense_model():
    
    inputs = layers.Input(shape=(224, 224, 3))
    base = DenseNet121(
                include_top=False, weights='imagenet',
                input_tensor=inputs
                )
    base.trainable = False

    transfer_layers = base.output

    transfer_layers = layers.GlobalAveragePooling2D()(transfer_layers)
    transfer_layers = BatchNormalization()(transfer_layers)
    transfer_layers = Dropout(0.5)(transfer_layers)
    transfer_layers = Dense(1024,activation='relu')(transfer_layers) 
    transfer_layers = Dense(512,activation='relu')(transfer_layers) 
    transfer_layers = Dropout(0.5)(transfer_layers)

    outputs=Dense(40,activation='sigmoid')(transfer_layers)

    model = Model(inputs, outputs, name="Dense")
    model.summary()
    return model

def run(train, val, test):
    model = dense_model()
    radam = tfa.optimizers.RectifiedAdam(learning_rate=0.001)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    optimizer = ranger
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=['binary_accuracy']
        )
    epoch = 20
    history = model.fit(
        train,
        epochs=epoch,
        validation_data=val,
    )
    eval = model.evaluate(test, return_dict=True)
    model.save(os.path.join('models', "celeb_ai.hdf5"))
    return history, eval


#conducts fine tuning after trianing model
def run_fine_tuned_model(train, val, test):
    inputs = layers.Input(shape=(224, 224, 3))
    base = DenseNet121(
                include_top=False, weights='imagenet',
                input_tensor=inputs
                )
    base.trainable = False

    transfer_layers = base.output

    transfer_layers = layers.GlobalAveragePooling2D()(transfer_layers)
    transfer_layers = BatchNormalization()(transfer_layers)
    transfer_layers = Dropout(0.5)(transfer_layers)
    transfer_layers = Dense(1024,activation='relu')(transfer_layers) 
    transfer_layers = Dense(512,activation='relu')(transfer_layers) 
    transfer_layers = Dropout(0.5)(transfer_layers)

    outputs=Dense(40,activation='sigmoid')(transfer_layers)

    model = Model(inputs, outputs, name="Dense")
    radam = tfa.optimizers.RectifiedAdam(learning_rate=0.001)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    optimizer = ranger
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=['binary_accuracy']
        )
    epoch = 10
    history = model.fit(
        train,
        epochs=epoch,
        validation_data=val,
    )
    base.trainable = True
    freeze_layers = len(base.layers) - 8
    for layer in base.layers[:freeze_layers]:
        #only unfreeze top 8 layers
        layer.trainable = False
    radam = tfa.optimizers.RectifiedAdam(learning_rate=0.00001)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    optimizer = ranger
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=['binary_accuracy']
        )

    epoch = 25
    fine_tune_history = model.fit(
        train,
        epochs=epoch,
        initial_epoch=history.epoch[-1],
        validation_data=val,
    )    
    eval = model.evaluate(test, return_dict=True)
    model.save(os.path.join('models', "celeb_ai.hdf5"))

    return fine_tune_history, eval

def main():
    train, val, test = dp.main()
    #history, eval = run(train, val, test)
    history, eval = run_fine_tuned_model(train, val, test)
    plot_results(history) 
    print(history.history)
    print(eval)

if __name__ == "__main__":
    main()