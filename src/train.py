import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow 
import keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, LeakyReLU,Dropout, Convolution2D, GlobalAveragePooling2D,BatchNormalization, Input,MaxPool2D
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils, plot_model
from tensorflow.keras.preprocessing import image
from sklearn import preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50



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

#Prueba de modelos:

def model():

    model = Sequential()
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(106, 68, 3)))
    model.add(MaxPooling2D((2, 2),padding="same"))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2),padding="same"))
    model.add(Dropout(0.2)) 

    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2),padding="same"))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2),padding="same"))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation="softmax"))

    return model
def modelGray(): 
    model = Sequential()
    model.add(Conv2D(32, (5, 5),padding="same", activation='relu', input_shape=(106, 68, 1)))
    model.add(MaxPooling2D((2, 2),padding="same"))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2),padding="same"))
    model.add(Dropout(0.2))
        
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2),padding="same"))
    model.add(Dropout(0.2))
    
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3,activation="softmax"))
        
    return model

def modelResNet50():
    model = tensorflow.keras.applications.ResNet101(include_top=False,weights='imagenet',input_tensor=None,input_shape=(106, 68,3))
    input_tensor = Input(shape=(106, 68,3))
    #bn = BatchNormalization()(input_tensor)
    #x = model(bn)
    x = model(input_tensor)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output= Dense(3, activation='softmax')(x)
    modelResNet = Model(inputs=input_tensor, outputs=output)

    return modelResNet

def modelResNet101():
    model = tensorflow.keras.applications.ResNet101(include_top=False,weights='imagenet',input_tensor=None,input_shape=(106, 68,3))
    input_tensor = Input(shape=(106, 68,3))
    bn = BatchNormalization()(input_tensor)
    x = model(bn)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    output= Dense(3, activation='softmax')(x)
    modelResNet = Model(inputs=input_tensor, outputs=output)

    return modelResNet
    
def modelDenseNet121():
    input_tensor = Input(shape=(106,68,3))
    dense = tensorflow.keras.applications.DenseNet121(include_top=False,weights='imagenet',input_tensor=input_tensor,input_shape=(106, 68,3))
    dense.trainable=True
    x = dense.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output= Dense(3, activation='softmax')(x)
    model = Model(inputs=dense.input, outputs=output)
    return model

def get_optimizer(initial_learning_rate,steps_per_epoch):
    return Adam(tensorflow.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate,decay_steps=steps_per_epoch*1000,decay_rate=1,staircase=False))

def fit(model,trainGen,validationGen,optimizer,steps_per_epoch,epochs,validation_steps):
    callback = tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4)  
    model.compile(optimizer= optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    history=model.fit(trainGen,steps_per_epoch=steps_per_epoch,epochs=epochs,validation_data=validationGen,validation_steps=validation_steps,callbacks=[callback])
    score = model.evaluate(validationGen)
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

def classReport(model,validationGen,validsteps):
    Y_pred = model.predict(validationGen, validsteps+1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Classification Report')
    class_labels = list(validationGen.class_indices.keys()) 
    print(classification_report(validationGen.classes, y_pred, target_names=class_labels))
    conf_mat = confusion_matrix(validationGen.classes, y_pred)
    #conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    print('Confusion Matrix')
    print(conf_mat)
    sns.heatmap(conf_mat,annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')