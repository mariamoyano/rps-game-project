from tensorflow.keras.preprocessing.image import load_img,img_to_array, ImageDataGenerator
from os import walk
import cv2
from scipy import ndimage
import numpy as np

Path='./INPUT/rps/'

def fileList(path,directoryname):
    """
    Lista de los nombres de los ficheros de cada clase
    """
    return next(walk(f"{path}{directoryname}/"))[2]

def loadImgd(path,directory,fileList):
    """
    Método que creara las imagenes
    """
    datagen = ImageDataGenerator(rotation_range=15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,
        horizontal_flip=True,brightness_range=[0.2,1.0],fill_mode='nearest')
    for file in fileList:
        img = load_img(f"{path}{directory}/{file}")
        arr = img_to_array(img)
        arr = arr.reshape((1,) + arr.shape)
        i = 0
        for batch in datagen.flow(arr, batch_size=1,save_to_dir=f"{path}{directory}/", save_prefix=directory+str(i), save_format='png'):
            i += 1
            if i > 20:
                break

def dataAug():
    """
    Crea más imagenes de las 3 clases a partir de data augmentation por si se quiere aumentar el número de imagenes.
    """
    fpaper=fileList(Path,"paper")
    frock=fileList(Path,"rock")
    fscissors=fileList(Path,"scissors")
    loadImgd(Path,"paper",fpaper) 
    loadImgd(Path,"rock",frock) 
    loadImgd(Path,"scissors",fscissors) 

def videoToImg(path,frame,video):
    """
    Pasa video a imagen. Útil para crear nuestro dataset.
    """
    vidcap = cv2.VideoCapture(video)
    r,image = vidcap.read()
    i = 0
    while r:

        cv2.imwrite(f"{path}{frame}/{frame}%d.png" % i, image)    
        r,image = vidcap.read()
        i += 1


#Para obtener mi dataset, he grabado las posiciones y las he pasado a imágenes

#videoToImg(Path,"paper","./INPUT/Videos/paper.mp4")
#videoToImg(Path,"rock","./INPUT/Videos/rock.mp4")
#videoToImg(Path,"scissors","./INPUT/Videos/scissors.mp4")
#videoToImg(Path,"lizard","./INPUT/Videos/lizard.mp4")
#videoToImg(Path,"spock","./INPUT/Videos/spock.mp4")


def cutImages(path,directoryname):
    """
    Recorta las imagenes extraidas del video y las gira como necesito
    En mi caso, fue por que grábe imagenes con varios dispositivos. No es necesario si se graban los videos desde la camara wen.
    """
    files = fileList(path,directoryname)
    for e in files:
        img = cv2.imread(f"{path}{directoryname}/{e}")
        newimg = img[0:720, 0:540]
        rot = ndimage.rotate(newimg,270)
        cv2.imwrite(f"{path}{directoryname}/{e}", rot)
    return 0




#cutImages(Path,"paper")
#cutImages(Path,"rock")
#cutImages(Path,"scissors")
#cutImages(Path,"lizard")
#cutImages(Path,"spock")

