#importing different modules
import cv2
import numpy as np
import mediapipe as mp
import time
import pygame
from movement import isMovement
from utils import draw_lines, get_landmark_y, get_pose_landmarks

cap = cv2.VideoCapture(0) #capturing the video default webcam
landMarks = [] #landmarks -> that are used to store last 5 landmarks
movement_threshold = 0.02
movements = [0,0,0,0,0] #list to store current movements according to previous landmarks

def preStartingLine(frame:np.array, movements:list, start_time, threshold:int = 0.02):
    """all the activities in pre starting line """
    goToTheStart = False
    currLandmark = get_pose_landmarks(frame)
    if currLandmark.any() is None:
        return None
    NOSE_Y = get_landmark_y(currLandmark, frame, mp.solutions.pose.PoseLandmark.NOSE)
    status = "Not Ready"

    if NOSE_Y is None:
        return None

    if NOSE_Y > 180 and NOSE_Y < 220:
        if start_time is None:
            start_time = time.time()
        

        elapsed = time.time() - start_time
        print(elapsed)
        if elapsed >= 3:
            status = "Ready"
            goToTheStart = True
            isMovement(frame, landMarks, movements, movementThreshold=threshold)
        
        if NOSE_Y < 200:
            start_time = time.time()
    else:
        status = "Not ready"
        start_time = None  # reset if user is out of position

    draw_lines(frame, 180, 220, status)
    return (start_time, goToTheStart)


def main():
    start_time = None
    while cap.isOpened():
        ret, frame = cap.read()
        
        """Pre starting line phase"""
        start_time, canGoToStartLine = preStartingLine(frame, movements, start_time)
        if canGoToStartLine:
            pygame.mixer.init()
            pygame.mixer.music.load("goSound.mp3")
            pygame.mixer.music.play()
            start_time = None
        cv2.imshow('Filtered Movement Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
