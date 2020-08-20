import os
import cv2

#path='./INPUT/rps/'

def fileList(path,directoryname):
    """
    Lista de los nombres de los ficheros de cada clase
    """
    return next(os.walk(f"{path}{directoryname}/"))[2]


def get_data(path,class_name):
    """
    Pasa video a imagen. Graba desde la camara, pulsar q para detener. Útil para crear nuestro dataset.
    He grabado en diferentes escenarios con fondos e iluminación diferentes.
    """
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    i = 0
    print("Comprobando ficheros existentes...")
    while(True):
        ret,frame = video.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRAY)
        image = cv2.flip(frame, 1)
        img = cv2.rectangle(image,(180,25),(460,455),(255,0,55),3)
        
        if os.path.isfile(f"{path}{class_name}/{class_name}{i}.png"):
            
            i+=10
        else:
            cv2.imshow('frame',img)
            cv2.imwrite(f"{path}{class_name}/{class_name}%d.png" % i, image)    
            i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
        
    video.release()
    cv2.destroyAllWindows()
#get_data(path,"paper")
#get_data(path,"rock")
#get_data(path,"scissors")

def videoToImg(path,frame,video):
    """
    Pasa video a imagen. Útil para crear nuestro dataset. Algunos videos los he grabado desde el movil.
    """
    video = cv2.VideoCapture(video)
    r,image = video.read()
    i = 0
    while r:
        if os.path.isfile(f"{path}{class_name}/{class_name}{i}.png"):
            
            i+=10
        else:
            cv2.imwrite(f"{path}{frame}/{frame}%d.png" % i, image)    
            r,image = video.read()
            i += 1



#videoToImg(path,"paper","./INPUT/Videos/paper.mp4")
#videoToImg(path,"rock","./INPUT/Videos/rock.mp4")
#videoToImg(path,"scissors","./INPUT/Videos/scissors.mp4")



def cutImages(path,directoryname):
    """
    Recorta las imagenes extraidas del video y las gira como necesito
    En mi caso, fue por que grábe imagenes con varios dispositivos. No es necesario si se graban los videos desde la camara wen.
    """
    files = fileList(path,directoryname)
    for e in files:
        img = cv2.imread(f"{path}{directoryname}/{e}")
        img = cv2.resize(img,(640,480))
        newimg = img[28:452, 183:457]
        cv2.imwrite(f"{path}{directoryname}/{e}", cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY))
        
    return 0

#cutImages(path,"paper")
#cutImages(path,"rock")
#cutImages(path,"scissors")


