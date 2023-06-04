import numpy as np
import cv2
import os
import glob

import faceRecognition as fr
print (fr)

test_img=cv2.imread(r'E:\python\btl\BTL\image_test\test3.jpg')  #Give path to the image which you want to test
#path = glob.glob("E:/python/btl/BTL/image_test/f18.jpg")
#for file in path:
    #test_img=cv2.imread(file) 
    


faces_detected,gray_img=fr.faceDetection(test_img)
print("face Detected: ",faces_detected)

#Training will begin from here

faces,faceID=fr.labels_for_training_data(r'E:\python\btl\BTL\training_image') #Give path to the train-images folder which has both labeled folder as 0 and 1
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.write(r'E:\python\btl\BTL\trainingData.yml') #It will save the trained model. Just give path to where you want to save



#Uncomment below line for subsequent runs
#face_recognizer=cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read('trainingData.yml')#use this to load training data for subsequent runs
name={0:"man",1:"woman"}    #Change names accordingly. If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.


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
cv2.waitKey(0) 
        
cv2.destroyAllWindows
