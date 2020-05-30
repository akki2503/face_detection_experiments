import dlib
import numpy as np
import cv2
import os
import sys
import imutils
from imutils import face_utils
import matplotlib.pyplot as plt
import pyautogui as pyat
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vid = cv2.VideoCapture(0)
plt.ion()
count = 0
if os.path.exists('ann.txt'):
    os.remove('ann.txt')
annotation_file = open("ann.txt", "a")

crop = True
tlx, tly, rbx, rby = 0, 0, 0, 0
def mouse_click(event, x, y, flags, params):
    global tlx, tly, rbx, rby, crop
    print("sadas")
    cv2.rectangle(frame, (tlx, tly), (rbx, rby), (0, 255, 0), 2)
    cv2.imshow("cropped", frame)

   
try:
    while(True):
        # Capture frame-by-frame
        ret, frame = vid.read()
        if crop:
            frame_cp = frame.copy()
            if crop:
                # cv2.rectangle(frame_cp, (tlx, tly), (rbx, rby), (0, 255, 0), 2)
                image_path = '/home/akki/Documents/python_projects/face_data/image_'+str(int(count))+'.jpg'
                # annotation_file.write(str(roi) + ' ' + image_path + "\n")
                count+=1
                frame = frame.astype('uint8')
                if count>100:
                    break
                cv2.imshow('frame',frame_cp)
                cv2.imwrite(image_path, frame)
                cv2.waitKey(-1)
                # show()
                # Display the frame until new frame is available
                # clear_output(wait=True)
    annotation_file.close()
    vid.release()
    cv2.destroyAllWindows()
except KeyboardInterrupt:
    # Release the Video Device
    vid.release()
    # Message to be displayed after releasing the device
    print("Released Video Resource")