from keras.preprocessing.image import load_img,img_to_array
from os import walk
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from scipy import ndimage
import numpy as np

trainPath='./INPUT/rps/'
testPath='./INPUT/rps-test-set/'

def fileList(path,directoryname):
    """
    Lista de los nombres de los ficheros de cada clase
    """
    return next(walk(f"{path}{directoryname}/"))[2]

def loadImgd(path,directory,fileList):
    """
    Método que creara las imagenes
    """
    datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,
        horizontal_flip=True,vertical_flip=True,brightness_range=[0.2,1.0],fill_mode='nearest')
    for file in fileList:
        img = load_img(f"{path}{directory}/{file}")
        arr = img_to_array(img)
        arr = arr.reshape((1,) + arr.shape)
        i = 0
        for batch in datagen.flow(arr, batch_size=1,save_to_dir=f"{path}{directory}/", save_prefix=directory+str(i), save_format='png'):
            i += 1
            if i > 20:
                break


#fpaper=fileList(trainPath,"paper")
#frock=fileList(trainPath,"rock")
#fscissors=fileList(trainPath,"scissors")
Path='INPUT/Videos/'
def dataAug():
    """
    Crea las imagenes de las 3 clases
    """
    fpaper=fileList('INPUT/Videos/',"paper")
    frock=fileList('INPUT/Videos/',"rock")
    fscissors=fileList('INPUT/Videos/',"scissors")
    loadImgd(Path,"paper",fpaper) 
    loadImgd(Path,"rock",frock) 
    loadImgd(Path,"scissors",fscissors) 

def videoToImg(frame,video):
    """
    Pasa video a imagen.
    """
    vidcap = cv2.VideoCapture(video)
    r,image = vidcap.read()
    i = 0
    while r:

        cv2.imwrite(f"INPUT/Videos/{frame}/{frame}%d.png" % i, image)     # save frame as JPEG file
        r,image = vidcap.read()
        i += 1



#videoToImg("paper","./INPUT/Videos/paper.mp4")
#videoToImg("paper","./INPUT/Videos/paper1.mp4")
#videoToImg("paper","./INPUT/Videos/paper2.mp4")
#videoToImg("paper","./INPUT/Videos/paper3.mp4")

#videoToImg("rock","./INPUT/Videos/rock3.mp4")
#videoToImg("rock","./INPUT/Videos/rock1.mp4")
#videoToImg("rock","./INPUT/Videos/rock2.mp4")
#videoToImg("rock","./INPUT/Videos/rock.mp4")

#videoToImg("scissors","./INPUT/Videos/scissors1.mp4")
#videoToImg("scissors","./INPUT/Videos/scissors2.mp4")
#videoToImg("scissors","./INPUT/Videos/scissors3.mp4")
#videoToImg("scissors","./INPUT/Videos/scissors.mp4")

def cutImages(path,directoryname):
    """
    Recorta las imagenes extraidas del video y las gira como necesito
    """
    files = fileList(path,directoryname)
    for e in files:
        img = cv2.imread(f"{path}{directoryname}/{e}")
        newimg = img[0:720, 0:720]
        rot = ndimage.rotate(newimg,270)
        cv2.imwrite(f"{path}{directoryname}/{e}", newimg)
    return 0
cutImages("./INPUT/Videos/","paper")
cutImages("./INPUT/Videos/","rock")
cutImages("./INPUT/Videos/","scissors")

def imgToFFT(path,directoryname):
    """
    FFT
    """
    files = fileList(path,directoryname)
    for e in files:
        img = cv2.imread(f"{path}{directoryname}/{e}",0).astype(np.float32)/255
   
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        rows, cols = img.shape
        crow,ccol = rows/2 , cols/2
        fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        cv2.imwrite(f"./INPUT/rpsfft/{directoryname}/{e}", img_back)
        
    return 0  

#imgToFFT("./INPUT/Videos/","paper")
#imgToFFT("./INPUT/Videos/","rock")
#imgToFFT("./INPUT/Videos/","scissors")