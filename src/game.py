import random
from tkinter import Tk, Label, Frame, BOTTOM,LEFT,RIGHT, ttk,messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
from playsound import playsound
import threading
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model=load_model( 'modelResNet101.h5')
#model=load_model("RPS_model.h5")
#model= load_model("modelA.h5")
#model = load_model("modelResNet.h5")
#model = load_model("modelAug.h5")
#model = load_model("modelAug4.h5")
winSound= "/home/maria/GIT/rps-game-project/INPUT/win.mp3"
tieSound= "/home/maria/GIT/rps-game-project/INPUT/tie.wav"
loseSound= "/home/maria/GIT/rps-game-project/INPUT/lose.mp3"

def play(yourChoice):
    options={0:"PAPER",1:"ROCK" ,2:"SCISSORS"}
    pcOption=random.randint(0,2)
    result =f"PC:{options[pcOption]} vs YOU: {options[yourChoice]}"
    if(yourChoice == pcOption):
        return (result+"\n\n\n    TIE!!",tieSound)
    elif yourChoice==0 and pcOption==1 or yourChoice==1 and pcOption==2 or yourChoice==2 and pcOption==0:
        return (result+"\n\n\n    YOU WIN!!!",winSound)
    else:
        return (result+"\n\n\n    YOU LOSE!!",loseSound)
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

window = Tk()
window.title("PAPER, ROCK OR SCISSORS")
window.config(background="black")
lmain = Label(window)
lmain.pack()

frame = Frame(window)
frame.pack()

def show():

    retval, frame = video.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show)

def predict():
    retval,frame = video.read()
    img=cv2.resize(frame,(48,36))
    pred = model.predict(np.array([img]))
    print("Predicted:",pred)

    if max(pred[0]) == (pred[0])[0]:
        yourChoice=0
        x=play(yourChoice)
        threading.Thread(target=playsound, args=(x[1],)).start()
        messagebox.showinfo(message=x[0], title="Game results")
    elif max((pred[0])) == (pred[0])[1]:
        yourChoice=1
        x=play(yourChoice)
        threading.Thread(target=playsound, args=(x[1],)).start()
        messagebox.showinfo(message=x[0], title="Game results")
    elif max((pred[0])) == (pred[0])[2]:
        yourChoice=2
        x=play(yourChoice)
        threading.Thread(target=playsound, args=(x[1],)).start()
        messagebox.showinfo(message=x[0], title="Game results")
    else:
        print("Please, try again")
        
     
    
style = ttk.Style()
style.theme_use('alt')
style.configure('TButton', background = 'gray', foreground = 'white', width = 20, borderwidth=3, focusthickness=3, focuscolor='none')
style.map('TButton', background=[('active','red')])

button = ttk.Button(frame, text="QUIT", command=window.destroy)
button.pack(side=RIGHT)
playbutton = ttk.Button(frame,text="PLAY",command=predict)
playbutton.pack(side=LEFT)

show()

window.mainloop()

