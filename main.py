import cv2
import numpy as np
import mediapipe as mp
import time
import pygame
from movement import isMovement
from utils import draw_lines, get_pose_landmarks, get_landmark_y

# Constants
PRE_START_Y_MIN = 180
PRE_START_Y_MAX = 220
START_LINE_Y = 400
IMAGINARY_START_LINE_OFFSET = 50
MOVEMENT_THRESHOLD = 0.02
READY_HOLD_TIME = 3
READY_TO_START_DELAY = 5
START_OK_WINDOW = (3, 4.1)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def get_nose_y(frame):
    landmarks = get_pose_landmarks(frame)
    if landmarks is not None:
        return get_landmark_y(landmarks, frame, mp_pose.PoseLandmark.NOSE)
    return None

def preStartingLine(frame, start_time):
    goToTheStart = False
    nose_y = get_nose_y(frame)
    status = "Not Ready"

    if nose_y is None:
        return None, None

    if PRE_START_Y_MIN < nose_y < PRE_START_Y_MAX:
        if start_time is None:
            start_time = time.time()

        elapsed = time.time() - start_time
        if elapsed >= READY_HOLD_TIME:
            status = "Ready"
            goToTheStart = True
    else:
        start_time = None

    draw_lines(frame, PRE_START_Y_MIN, PRE_START_Y_MAX, status)
    return start_time, goToTheStart

def crossedLine(frame):
    nose_y = get_nose_y(frame)
    return nose_y is not None and nose_y > START_LINE_Y

def isReady(frame):
    nose_y = get_nose_y(frame)
    if nose_y is None:
        return False
    imaginary_line = START_LINE_Y - IMAGINARY_START_LINE_OFFSET
    draw_lines(frame, imaginary_line, START_LINE_Y, "")
    return imaginary_line <= nose_y <= START_LINE_Y and not crossedLine(frame)

def playsound(filename):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound {filename}: {e}")

def main():
    cap = cv2.VideoCapture(0)
    landMarks = []
    movements = [0] * 5

    TpreStart = None
    canLeavePreStartLine = True
    TSinceGoToStart = None
    crossedTooEarly = False
    readySoundHeard = False
    TSinceReadySound = None
    falseStartOnce = False
    gameStarted = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        isMovement(frame, landMarks, movements, MOVEMENT_THRESHOLD)
        TpreStart, canGoToStartLine = preStartingLine(frame, TpreStart)

        if canGoToStartLine and canLeavePreStartLine:
            playsound("Sounds/goSound.mp3")
            canLeavePreStartLine = False
            TSinceGoToStart = time.time()

        if TSinceGoToStart:
            elapsed = time.time() - TSinceGoToStart

            if crossedLine(frame) and not crossedTooEarly and elapsed >= 2:
                playsound("Sounds/falseStartBuzzer.mp3")
                crossedTooEarly = True

            if elapsed >= READY_TO_START_DELAY and isReady(frame) and not readySoundHeard:
                playsound("Sounds/readySound.mp3")
                readySoundHeard = True
                TSinceReadySound = time.time()

        if TSinceReadySound:
            start_elapsed = time.time() - TSinceReadySound
            startOK = START_OK_WINDOW[0] <= start_elapsed <= START_OK_WINDOW[1]

            if startOK and not falseStartOnce:
                if isMovement(frame, landMarks, movements, MOVEMENT_THRESHOLD):
                    playsound("Sounds/falseStartBuzzer.mp3")
                    falseStartOnce = True

            if not falseStartOnce and start_elapsed > START_OK_WINDOW[1] and not gameStarted:
                playsound("gunSound.mp3")
                gameStarted = True

        cv2.imshow('Speed Skating False Start Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
