import cv2
import numpy as np
import mediapipe as mp
import time
import pygame
from movement import isMovement
from utils import draw_lines, get_pose_landmarks, get_landmark_y

# Constants
PRE_START_Y_MIN = 100
PRE_START_Y_MAX = 300
START_LINE_Y = 400
IMAGINARY_START_LINE_OFFSET = 50
MOVEMENT_THRESHOLD = 0.02
READY_HOLD_TIME = 3
READY_TO_START_DELAY = 5
START_OK_WINDOW = (3, 4.1)
GO_SOUND = "Sounds/goSound.mp3"
READY_SOUND = "Sounds/readySound.mp3"
FALSE_START_SOUND = "Sounds/falseStartBuzzer.mp3"
GUN_SOUND = "Sounds/gunSound.mp3"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def get_landmark_y_center(frame, landmark_ids):
    landmarks = get_pose_landmarks(frame)
    if landmarks is None:
        return None
    y_vals = [get_landmark_y(landmarks, frame, lid) for lid in landmark_ids]
    return int(np.mean(y_vals))

def preStartingLine(frame, start_time):
    goToTheStart = False
    hips_y = get_landmark_y_center(frame, [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP])
    status = "Not Ready"

    if hips_y is None:
        return None, None

    if PRE_START_Y_MIN < hips_y < PRE_START_Y_MAX:
        if start_time is None:
            start_time = time.time()
        elapsed = time.time() - start_time
        if elapsed >= 3:
            status = "Ready"
            goToTheStart = True
    else:
        start_time = None

    draw_lines(frame, PRE_START_Y_MIN, PRE_START_Y_MAX, status)
    print(hips_y)
    return start_time, goToTheStart

def crossedLine(frame):
    ankles_y = get_landmark_y_center(frame, [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE])
    return ankles_y is not None and ankles_y > START_LINE_Y

def isReady(frame):
    hips_y = get_landmark_y_center(frame, [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP])
    if hips_y is None:
        return False
    imaginary_line = START_LINE_Y - 50
    draw_lines(frame, imaginary_line, START_LINE_Y, "")
    return imaginary_line <= hips_y <= START_LINE_Y and not crossedLine(frame)

def playsound(filename):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound {filename}: {e}")

def main():
    cap = cv2.VideoCapture(1)
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

        isMovement(frame, landMarks, movements, MOVEMENT_THRESHOLD, fullBody=True)
        draw_lines(frame, PRE_START_Y_MIN, PRE_START_Y_MAX,state="done")
        imaginary_line = START_LINE_Y - 50
        draw_lines(frame, imaginary_line, START_LINE_Y, "yayy")
        TpreStart, canGoToStartLine = preStartingLine(frame, TpreStart)

        if canGoToStartLine and canLeavePreStartLine:
            playsound(GO_SOUND)
            canLeavePreStartLine = False
            TSinceGoToStart = time.time()

        if TSinceGoToStart:
            elapsed = time.time() - TSinceGoToStart

            if crossedLine(frame) and not crossedTooEarly and elapsed >= 2:
                playsound(FALSE_START_SOUND)
                crossedTooEarly = True

            if elapsed >= 5 and isReady(frame) and not readySoundHeard:
                playsound(READY_SOUND)
                readySoundHeard = True
                TSinceReadySound = time.time()

        if TSinceReadySound:
            start_elapsed = time.time() - TSinceReadySound
            startOK = 3 <= start_elapsed <= 4.1

            if startOK and not falseStartOnce:
                if isMovement(frame, landMarks, movements, MOVEMENT_THRESHOLD, fullBody=True):
                    playsound(FALSE_START_SOUND)
                    falseStartOnce = True

            if not falseStartOnce and start_elapsed > 4.1 and not gameStarted:
                playsound(GUN_SOUND)
                gameStarted = True

        cv2.imshow('Speed Skating Full-Body Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
