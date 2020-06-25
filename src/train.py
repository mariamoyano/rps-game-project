import pandas as pd
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow 
import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, LeakyReLU,Dropout, Convolution2D, GlobalAveragePooling2D,BatchNormalization, Input,MaxPool2D
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils, plot_model
from tensorflow.keras.preprocessing import image
from sklearn import preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_resnet_v2 import \
    InceptionResNetV2,preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
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


def model():

    model = Sequential()
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(240, 320, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2)) 
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation="softmax"))

    return model
def modelGray(): 
    model = Sequential()
    model.add(Conv2D(32, (5, 5),padding="same", activation='relu', input_shape=(36, 48, 1)))
    model.add(MaxPooling2D((2, 2),padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2),padding="same"))
    model.add(Dropout(0.25))
        
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2),padding="same"))
    model.add(Dropout(0.25))
    
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(3,activation="softmax"))
        
    return model
def modelResNet():
    model_inc = InceptionResNetV2(include_top=False,weights='imagenet',input_tensor=None,input_shape=(240,320,3))
    input_tensor = Input(shape=(240,320,3))
    bn = BatchNormalization()(input_tensor)
    x = model_inc(input_tensor)
    x = Flatten()(x)
    #x = Dropout(0.5)(x)
    #x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output= Dense(3, activation='softmax')(x)
    modelResNet = Model(inputs=input_tensor, outputs=output)

    return modelResNet

def modelResNet50():
    model = tensorflow.keras.applications.ResNet101(include_top=False,weights='imagenet',input_tensor=None,input_shape=(36, 48,3))
    input_tensor = Input(shape=(36, 48,3))
   # bn = BatchNormalization()(input_tensor)
    x = model(input_tensor)
    x = Flatten()(x)
    #x = Dropout(0.5)(x)
    output= Dense(3, activation='softmax')(x)
    modelResNet = Model(inputs=input_tensor, outputs=output)

    return modelResNet

def modelResNet101():
    model = tensorflow.keras.applications.ResNet101(include_top=False,weights='imagenet',input_tensor=None,input_shape=(36, 48,3))
    input_tensor = Input(shape=(36, 48,3))
    bn = BatchNormalization()(input_tensor)
    x = model(bn)
    x = Flatten()(x)
    #x = Dropout(0.5)(x)
    output= Dense(3, activation='softmax')(x)
    modelResNet = Model(inputs=input_tensor, outputs=output)

    return modelResNet
    
def modelDenseNet121(trainGen,validationGen,optimizer,steps,validsteps):
    input_tensor = Input(shape=(36,48,3))
    dense = tensorflow.keras.applications.DenseNet121(include_top=False,weights='imagenet',input_tensor=input_tensor,input_shape=(36, 48,3))
    dense.trainable=True
    x = dense.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output= Dense(3, activation='softmax')(x)
    model = Model(inputs=dense.input, outputs=output)
    optimizer="adam"
    history=fit(model,trainGen,validationGen,optimizer,steps,15,validsteps)
    model.save("modelDense.h5")
    plot(history)
    classReport(model,validationGen,steps)
    return history

def fit(model,trainGen,validationGen,optimizer,steps_per_epoch,epochs,validation_steps):
    callback = tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6)  
    model.compile(optimizer= optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
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





def classReport(model,validationGen,steps):
    Y_pred = model.predict(validationGen, steps)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Classification Report')
    class_labels = list(validationGen.class_indices.keys()) 
    print(classification_report(validationGen.classes, y_pred, target_names=class_labels))
    conf_mat = confusion_matrix(validationGen.classes, y_pred)
    conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    print(conf_mat_normalized)
    sns.heatmap(conf_mat_normalized,annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

batch_size = 32
trainPath='./INPUT/rps/'


agen = ImageDataGenerator(rescale=1./255,
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True,
      fill_mode='nearest',validation_split = 0.1) 


trainGen= agen.flow_from_directory(trainPath,target_size=(36, 48),batch_size=batch_size,class_mode='binary')#,color_mode='grayscale') 

validationGen = agen.flow_from_directory(trainPath,target_size=(36, 48),batch_size=batch_size,class_mode='binary',subset = 'validation')#,color_mode='grayscale')  


steps = trainGen.n // trainGen.batch_size
validsteps = validationGen.n // validationGen.batch_size
print(steps,validsteps)
model=modelResNet101()
optimizer=SGD(lr=0.01)
history=fit(model,trainGen,validationGen,optimizer,steps,10,validsteps)
model.save("modelResNet101.h5")
classReport(model,validationGen,steps)
#model=load_model("modelRN101.h5")
