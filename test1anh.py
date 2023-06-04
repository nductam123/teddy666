# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:28:24 2023

@author: Admin
"""

import numpy as np
import cv2
import os
import glob
import faceRecognition as fr

print(fr)

test_img_path = "E:/python/btl/BTL/image_test/test1.jpg"
test_img = cv2.imread(test_img_path)
faces_detected, gray_img = fr.faceDetection(test_img)
print("Faces Detected:", faces_detected)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'E:\python\btl\BTL\trainingData.yml')

name = {0: "man", 1: "woman"}

for face in faces_detected:
    (x, y, w, h) = face
    roi_gray = gray_img[y:y + h, x:x + h]
    label, confidence = face_recognizer.predict(roi_gray)
    print("Confidence:", confidence)
    print("Label:", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    fr.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img, (700, 700))
cv2.imshow("Face Detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()