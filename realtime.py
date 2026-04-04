import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import time
# from playsound import playsound  #it has issue when alerm start it will play full song that i don't wnat
import pygame

pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")
# pygame.mixer → for sound control 



model=load_model("mymodel.h5")

img_size=224

face_casced=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

# “cv.data.haarcascades gives the path to OpenCV’s built-in cascade files, and we append the XML filename to load the pretrained face detector.”

cap=cv.VideoCapture(0)

class_label={0:"close eyes",1:"open eyes"}

prev_time = 0

closed_start_time=None
alert_playing=False
threshold=5


closed_frames = 0
open_frames = 0
frame_threshold = 3
# agar tin frame mai agar open aaya then open agar tin frame mai close aaya the close


while True:
    ret,frame=cap.read()

    curr_time = time.time()
    time_taken=curr_time-prev_time
    fps = 1 / time_taken    #it show how many frame per sec
    prev_time = curr_time   #it storing curr_time for next_frame

    if not ret:
        print("camera is not working")
        break

    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face=face_casced.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        face_roi=frame[y:y+h,x:x+w]
        gray_face=gray[y:y+h,x:x+w]

        eyes = eye_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.1,   # smaller steps → more detection ->zoom
            minNeighbors=3   
        )

        
        eye_preds = []

        for (ex,ey,ew,eh) in eyes[:2]:
            eye = face_roi[ey:ey+eh, ex:ex+ew]

            try:
                eye = cv.cvtColor(eye, cv.COLOR_BGR2RGB)   #model expect RGB
                eye = cv.resize(eye, (img_size, img_size))
                eye = eye / 255.0
                eye = eye.reshape(1, img_size, img_size, 3)

                pred = model.predict(eye, verbose=0)  #TensorFlow runs predict(), it can talk to you (print messages) or stay silent.->by defult it talk means showing bar running bar ,0 means  slienty do task and it is fast
                eye_preds.append(pred[0][0])

                # draw box only
                cv.rectangle(face_roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

            except:
                pass

        if len(eye_preds) > 0:
            avg_pred = sum(eye_preds) / len(eye_preds)

            if avg_pred > 0.6:
                open_frames += 1
                closed_frames = max(0,closed_frames-1)
            elif avg_pred < 0.4:
                closed_frames += 1
                open_frames =max(0,open_frames-1)
            else:
                continue

            # Stable label
            if closed_frames >= frame_threshold:
                label = "close eyes"
            elif open_frames >= frame_threshold:
                label = "open eyes"
            else:
                continue


            if label=="close eyes":
                if closed_start_time is None:
                    closed_start_time=time.time()

                closetime=time.time()-closed_start_time

                if closetime>threshold:
                    cv.putText(frame, "DROWSY ALERT!", (x, y-40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    
                    if not alert_playing:
                        pygame.mixer.music.play(-1)  # loop forever
                        alert_playing = True
            
            else:  #open eye
                closed_start_time = None
                if alert_playing:
                    pygame.mixer.music.stop()
                    alert_playing = False

                        
            cv.putText(frame, label, (x, y-10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            
        
        else:
            # if eyes not detected->skip this fram
            continue



    cv.putText(frame, f"FPS: {int(fps)}", (10, 30),
           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    cv.imshow("Eye Detection", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    


cap.release()
cv.destroyAllWindows()





