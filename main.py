#importing different modules
import cv2
import numpy as np
import mediapipe as mp
import time
import pygame
from movement import isMovement
from utils import draw_lines, get_landmark_y, get_pose_landmarks

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def preStartingLine(frame:np.array, movements:list, start_time, threshold:int = 0.02):
    """all the activities in pre starting line """
    goToTheStart = False
    currLandmark = get_pose_landmarks(frame)
    if currLandmark is None:
        return None, None
    NOSE_Y = get_landmark_y(currLandmark, frame, mp.solutions.pose.PoseLandmark.NOSE)
    status = "Not Ready"

    if NOSE_Y is None:
        return None, None

    if NOSE_Y > 180 and NOSE_Y < 220:
        if start_time is None:
            start_time = time.time()
        
        elapsed = time.time() - start_time
        print(elapsed)
        if elapsed >= 3:
            status = "Ready"
            goToTheStart = True

        if NOSE_Y < 180:
            start_time = time.time()
    else:
        status = "Not ready"
        start_time = None  # reset if user is out of position

    draw_lines(frame, 180, 220, status)
    return (start_time, goToTheStart)

def crossedLine(frame:np.array, lineLimit: int) -> bool:
    result = pose.process(frame)
    draw_lines(frame, lineLimit, lineLimit, "pos")
    if not result.pose_landmarks:
        return False
    #currYLandmarks = np.array([
     #   lm.y for lm in result.pose_landmarks.landmark
    #])
    currYLandmarks = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE].y

    image_height, image_width, _ = frame.shape
    nose_landmark = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
    currYLandmarks = int(nose_landmark.y * image_height)

    print("landmark: ", currYLandmarks)
    if(currYLandmarks > lineLimit):
        return True
    return False


def main():
    cap = cv2.VideoCapture(0) #capturing the video default webcam
    landMarks = [] #landmarks -> that are used to store last 5 landmarks
    movement_threshold = 0.02
    movements = [0,0,0,0,0] #list to store current movements according to previous landmarks
    start_time = None
    startingLineStartTime = time.time()
    goCount = True
    once = True
    preStartOkay = False
    while cap.isOpened():
        ret, frame = cap.read()
        isMovement(frame, landMarks, movements, movementThreshold=0.02)
        """Pre starting line phase"""
        start_time, canGoToStartLine = preStartingLine(frame, movements, start_time)
        
        if canGoToStartLine and goCount:
            pygame.mixer.init()
            pygame.mixer.music.load("goSound.mp3")
            pygame.mixer.music.play()
            goCount = False
            preStartOkay = True
            startingLineStartTime = time.time()

        """Starting line"""
        #check if the skater has crossed the line or not
        #if startingLineStartTime is None:
        elapsedStartingLine = time.time() - startingLineStartTime
        
        lineLimit = 400
        isCrossed = crossedLine(frame, lineLimit)
        if isCrossed and once and (elapsedStartingLine >= 5) and preStartOkay:
            pygame.mixer.init()
            pygame.mixer.music.load("falseStartBuzzer.mp3")
            pygame.mixer.music.play()
            once =  False
        
        cv2.imshow('Filtered Movement Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
