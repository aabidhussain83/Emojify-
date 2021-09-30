import tkinter as tk
from tkinter import *
import numpy as np
import cv2
from PIL import Image, ImageTk
import os

from keyring.backends import null

from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading

from tensorflow.python.ops.signal.shape_ops import frame

emotion_model=Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024,activation='relu'))
emotion_model.add(Dropout(0.25))
emotion_model.add(Dense(7,activation='softmax'))
emotion_model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

cur_path=os.path.dirname(os.path.abspath(__file__))

emoji_dist={0:"./emojis/angry.png",2:"./emojis/disgust.png",2:"./emojis/fear.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surprise.png"}
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
num=0
show_text=[0]
global frame_number


#cap1=cv2.VideoCapture("C:/Users/91993/Desktop/acting3.mp4")
cap1 = cv2.VideoCapture(0)

def show_subject():


    if not cap1.isOpened():
        print("Camera opening failed ! ")
    global frame_number

    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))


    # if frame_number > length:
    #     exit()
    # frame_number += 1



    flag1,frame1=cap1.read()



    bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)




    for (x, y, w, h) in num_faces:
        global show_text
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex

    #frame1 = cv2.resize(frame1, (600, 500))

    if flag1 is None:
        print ("Major error !: ")
    elif flag1:
        global last_frame1
        last_frame1=frame1.copy()
        pic=cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img=Image.fromarray(pic)
        imgtk=ImageTk.PhotoImage(image=img)
        lmain.imgtk=imgtk
        lmain.configure(image=imgtk)
        root.update()
        lmain.after(10,show_subject)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        exit()



def show_emoji():
    frame2=cv2.imread(emoji_dist[show_text[0]])
    pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(pic2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('Times New Roman', 25 , 'bold'))
    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(10, show_emoji)




if __name__ == '__main__':
    frame_number=0
    root = tk.Tk()
    lmain=tk.Label(master=root,padx=0,bd=0)

    lmain2=tk.Label(master=root,bd=0)
    lmain3=tk.Label(master=root,bd=0,fg='#CDCDCD',bg='black')
    lmain.place(x=20,y=20)
    lmain2.place(x=630, y=20)
    lmain3.place(x=630,y=20)

    root.title("Turn your facial expression to Emoji")
    root.geometry("1150x575")
    root['bg'] = 'white'
    exitButton = Button(root, text='Quit', fg="red",bg="#40a832", command=root.destroy, font=('Times New Roman', 15, 'bold')).pack(side=BOTTOM)
    #threading.Thread(target=show_subject).start()
    #threading.Thread(target= show_emoji()).start()

    show_subject()
    show_emoji()

    root.mainloop()