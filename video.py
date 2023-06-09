# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:44:15 2022

@author: Admin
"""

import numpy as np
import cv2
import os

import faceRecognition as fr
print (fr)



path = "E:/python/btl/BTL/videotest/test8.mp4"
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'E:\python\btl\BTL\trainingData.yml')    #Give path of where trainingData.yml is saved

cap=cv2.VideoCapture(path)   #If you want to recognise face from a video then replace 0 with video path

name={0:"man" ,1:"woman"}    #Change names accordingly.  If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.
while True:
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)
    print("face Detected: ",faces_detected)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

    
    
    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)
        print ("Confidence :",confidence)
        print("label :",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        fr.put_text(test_img,predicted_name,x,y)

    resized_img=cv2.resize(test_img,(1000,1000))

    cv2.imshow("face detection ", resized_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()