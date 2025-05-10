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

def getCurrentLandmarks(frame:np.array):
    #only  nose right now but will be implimented for the whole body afterwards
    result = pose.process(frame)
    if result.pose_landmarks is None:
        return None

    image_height, image_width, _ = frame.shape
    nose_landmark = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
    currYLandmarks = int(nose_landmark.y * image_height)

    return currYLandmarks

def preStartingLine(frame:np.array, movements:list, start_time, threshold:int = 0.02, timeThreshold = 3):
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
        if elapsed >= timeThreshold:
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
    currYLandmarks = getCurrentLandmarks(frame)
    if currYLandmarks is None:
        return False
    print("landmark: ", currYLandmarks)
    if(currYLandmarks > lineLimit):
        return True
    return False

def isReady(frame:np.array, lineLimit:int) ->bool:
    currYLandmarks = getCurrentLandmarks(frame)
    if currYLandmarks is None:
        return False
    imaginaryBackStartLine = lineLimit - 50
    
    draw_lines(frame, 350,400,"")
    if(currYLandmarks >= imaginaryBackStartLine and currYLandmarks <= lineLimit):
        if not crossedLine(frame, lineLimit):
            return True
        
    return False

def playsound(filename:str):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

def notInStartingPosition(frame:np.array, imaginaryStartLine, startLine):
    currYLandmarks = getCurrentLandmarks(frame)
    if currYLandmarks is None:
        return True
    
    if currYLandmarks < imaginaryStartLine or currYLandmarks > startLine:
        return True
        
    return False


def main():
    cap = cv2.VideoCapture(0) #capturing the video default webcam
    landMarks = [] #landmarks -> that are used to store last 5 landmarks
    movement_threshold = 0.02
    movements = [0,0,0,0,0] #list to store current movements according to previous landmarks

    #-----------------------#
    preStartLine = 200
    StartLine = 400

    playStarted = False

    TpreStart = None
    canLeavePreStartLine = True

    TSinceGoToStart = None
    corssedStartLineBeforeStart = False

    readySoundHeard = False
    TSinceReadySound = None

    startOK = False
    falseStartOnce = False
    gameStarted = False



    while cap.isOpened():
        ret, frame = cap.read()
        """Pre starting line phase"""
        TpreStart, canGoToStartLine = preStartingLine(frame, movements, TpreStart)
        
        #at the pre start line
        if canGoToStartLine and canLeavePreStartLine:
            playsound("goSound.mp3")
            canLeavePreStartLine = False
            TSinceGoToStart = time.time()
            playStarted = True

        """Starting line"""
        #check if the skater has crossed the line or not
        #if startingLineStartTime is None:
    
        #ready section
        if TSinceGoToStart is not None:
            TelapsedSinceGoToStart = time.time() - TSinceGoToStart
            isCrossed = crossedLine(frame, StartLine)
            if isCrossed and (not corssedStartLineBeforeStart) and (TelapsedSinceGoToStart >= 2) and playStarted:
                playsound("falseStartBuzzer.mp3")
                corssedStartLineBeforeStart = True

            if(TelapsedSinceGoToStart >= 5 and isReady(frame,400)) and (not readySoundHeard):
                playsound("readySound.mp3")
                readySoundHeard = True
                TSinceReadySound = time.time()

        #after ready section 
        if TSinceReadySound is not None:
            TelapsedSinceReadySound = time.time() - TSinceReadySound
            startOK = TelapsedSinceReadySound >=3 and TelapsedSinceReadySound <= 4.1
            if startOK and not falseStartOnce:
                if isMovement(frame, landMarks,movements,0.02):
                    playsound("falseStartBuzzer.mp3")
                    falseStartOnce = True
            if(not falseStartOnce and TelapsedSinceReadySound>=4.1) and not gameStarted:
                playsound("gunSound.mp3")
                gameStarted = True
                     
        cv2.imshow('Filtered Movement Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
