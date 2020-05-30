import dlib
import numpy as np
import cv2
import os
import sys
import imutils
from imutils import face_utils
import matplotlib.pyplot as plt
import pyautogui as pyat
# import tensorflow as tf

import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/akki/Downloads/model.dat')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# cap = cv2.VideoCapture('/content/drive/My Drive/eye gaze navigator/demo.mp4')
# Grab the input device, in this case the webcam
# You can also give path to the video file
vid = cv2.VideoCapture(0)
plt.ion()
# Put the code in try-except statements
# Catch the keyboard exception and 
# release the camera device and 
# continue with the rest of code.
try:
    while(True):
        # Capture frame-by-frame
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(frame)
        if not ret:
            # Release the Video Device if ret is false
            vid.release()
            # Message to be displayed after releasing the device
            print("Released Resource")
            break
        # Convert the image from OpenCV BGR format to matplotlib RGB format
        # to display the image
        # Display the frame
        rects = detector(gray,0)
        if rects ==0:
            print("no face detected")
        print(len(rects))
        for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            x_min = np.min(shape[:,0])
            y_min = np.min(shape[:,1])
            x_max = np.max(shape[:,0])
            y_max = np.max(shape[:,1])
            x_mid = int(x_max+x_min)
            y_mid = int(y_max+y_min)
            pyat.moveTo(x_mid+50, y_mid+50)
            for (sx, sy) in shape:
                cv2.circle(frame, (sx, sy), 5, (0, 0, 255), -1)
        
        frame = frame.astype('uint8')
        local_vars = list(locals().items())
        for var, obj in local_vars:
            print(var, sys.getsizeof(obj))
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        # show()
        # Display the frame until new frame is available
        # clear_output(wait=True)
    vid.release()
    cv2.destroyAllWindows()
except KeyboardInterrupt:
    # Release the Video Device
    vid.release()
    # Message to be displayed after releasing the device
    print("Released Video Resource")