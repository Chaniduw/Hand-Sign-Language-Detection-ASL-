# Importing necessary libraries
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Setting up the video capture from the default camera (0)
cap = cv2.VideoCapture(0)

# Initializing the hand detector with a maximum of 1 hand
detector = HandDetector(maxHands=1)

# Setting up parameters for image cropping and resizing
offset = 20  # Offset value for cropping around the detected hand
imgSize = 300  # Size of the final cropped and resized image

# Folder to save the captured images
folder = "Data/B"
counter = 0  # Counter for keeping track of the captured images

# Main loop for capturing and processing frames from the camera
while True:
    # Reading a frame from the camera
    success, img = cap.read()

    # Detecting hands in the frame using the HandDetector
    hands, img = detector.findHands(img)

    # Checking if hands are detected
    if hands:
        # Taking the first detected hand
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Creating a white canvas to place the cropped and resized hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Cropping the region around the detected hand with an offset
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Getting the shape of the cropped image
        imgCropShape = imgCrop.shape

        # Calculating aspect ratio of the hand's bounding box
        aspectRatio = h / w

        # Handling cases based on the aspect ratio
        if aspectRatio > 1:  # If the hand is taller than it is wide
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:  # If the hand is wider than it is tall
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Displaying the original cropped hand image
        cv2.imshow("ImageCrop", imgCrop)

        # Displaying the final image with white canvas
        cv2.imshow("ImageWhite", imgWhite)

    # Displaying the original frame with the detected hand
    cv2.imshow("Image", img)

    # Checking for the 's' key press to save the captured image
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        # Saving the captured image with a unique filename
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
