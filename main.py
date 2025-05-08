#importing different modules
import cv2
import mediapipe as mp
import numpy as np
from movement import isMovement


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) #capturing the video default webcam

landMarks = [] #landmarks -> that are used to store last 5 landmarks

movement_threshold = 0.02

movements = [0,0,0,0,0] #list to store current movements according to previous landmarks

trackMovement = 0

while cap.isOpened():
    ret, frame = cap.read()

    isMovement(frame,landMarks, movements, movementThreshold=movement_threshold)
          
    cv2.imshow('Filtered Movement Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


