import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, LeakyReLU,Dropout, Convolution2D, GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from tensorflow.keras.preprocessing import image
from sklearn import preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_resnet_v2 import \
    InceptionResNetV2,preprocess_input, decode_predictions
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
def plotImages(images_arr):
    """
    Muestra las imagenes
    """
    
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def model():

    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2)) 
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation="softmax"))

    return model

def modelResNet():
    model_inc = InceptionResNetV2(include_top=False,weights='imagenet',input_tensor=None,input_shape=(150,150,3))
    x = model_inc.output
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    output= Dense(3, activation='softmax', name='softmax')(x)
    modelResNet = Model(inputs=model_inc, outputs=output)

    return modelResNet


def fit(model,trainGen,validationGen,optimizer,steps_per_epoch,epochs,validation_steps):
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)  
    model.compile(optimizer= optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    history=model.fit(trainGen,steps_per_epoch=steps_per_epoch,epochs=epochs,validation_data=validationGen,validation_steps=validation_steps,callbacks=[callback])
    score = model.evaluate_generator(validationGen,600)
    print("Test loss: = ",score[0])
    print("Test accuracy = ",score[1])
    return history

def plot(history):
    #Acuraccy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    #Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



def modelGray(): 
    model = Sequential()
    model.add(Conv2D(32, (3, 3),padding="same", activation='relu', input_shape=(150, 150, 1)))
    model.add(MaxPooling2D((2, 2),padding="same"))
   
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2),padding="same"))
    model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation="softmax"))
        
    return model