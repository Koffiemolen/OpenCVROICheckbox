from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
from pathlib import Path
from os import listdir
import os
import shutil
import functools
from PIL import Image

# pip install opencv-python numpy scipy Pillow imutils
# Source: https://stackoverflow.com/questions/48866205/detect-whether-the-checkbox-is-checked-using-opencv
# sourceDir = 'C:/Temp/Singer/Output/page-1.png'

path = "C:/Temp/Output/a/"
checkedP = "C:/Temp/checked/"
uncheckedP = "C:/Temp/unchecked/"
included_extensions = ['png', 'PNG']
whitePixelAverage = []
THRESHOLD = 117

allFiles = [f for f in listdir(path) if
            any(f.endswith(ext) for ext in included_extensions)]  # Get all files in current directory

length = len(allFiles)

for i in range(length):
    print(allFiles[i])
    img = cv2.imread(path + allFiles[i])
    # cv2.imshow('Original', img)
    imgCrop = img[1000:1300, 50:110]  # Select your ROI here, I allow for a bit of deviation between image scans
    # cv2.imshow('Cropped', imgCrop)
    # cv2.waitKey(0)

    # Converting image to gray scale
    grayImage = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)

    # Image thresholding
    _, binaryImage = cv2.threshold(grayImage, 180, 225, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binaryImage = 255 - binaryImage

    Image.fromarray(binaryImage).show()

    # Using morphological operations to identify edges
    # Set min width to detect horizontal lines
    line_min_width = 7

    # Kernel to detect horizontal lines
    kernel_h = np.ones((1, line_min_width), np.uint8)

    # Kernel to detect vertical lines
    kernel_v = np.ones((line_min_width, 1), np.uint8)

    # Horizontal kernel on image
    img_bin_h = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel_h)

    # Vertical kernel on the image
    img_bin_v = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel_v)

    # Show image with horizontal lines only and vertical lines only
    # Image.fromarray(img_bin_h).show()
    # Image.fromarray(img_bin_v).show()

    # Combining the image, horizontal + vertical
    img_bin_final = img_bin_h | img_bin_v

    # Show combined image
    # Image.fromarray(img_bin_final).show()

    # some hocus pocus
    final_kernel = np.ones((3, 3), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=2)
    # Image.fromarray(img_bin_final).show()

    # Contours Filtering
    _, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

    squares = 0 # Counter for numbering potential checkboxes
    for x, y, w, h, area in stats[2:]:
        print('rectangle : (x , y) = ', x, y)
        print('size: w, h', w, h)
        # Only draw a rectangle on image within a certain size range, width and height
        if 7 <= w <= 13 and 7 <= h <= 13:
            print('rectangle ', squares, ': (x , y) = ', x, y)
            print('size: w, h', w, h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, str(squares), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        squares += 1

    # Show image with found checkboxes highlighted in red
    Image.fromarray(img).show()
