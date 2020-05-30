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

image_dir = '/home/akki/Documents/python_projects/face_data'
for root, dirs, files in os.walk(image_dir, topdown=False):
    for name in sorted(files):
        image_path = os.path.join(root,name)
        image = cv2.imread(image_path)
        r = cv2.selectROI(image, fromCenter=False, showCrosshair=False)
        annotation_file.write(str(r) + ' ' + image_path + "\n")
    annotation_file.close()