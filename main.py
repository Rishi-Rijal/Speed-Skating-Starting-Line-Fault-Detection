import cv2
import time
import pygame
from movement import isMovement
from utils import draw_lines
from detection import (
    isReady,
    preStartingLine,
    crossedLine,
)

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
        imaginary_line = START_LINE_Y - IMAGINARY_START_LINE_OFFSET
        draw_lines(frame, imaginary_line, START_LINE_Y, "yayy")
        TpreStart, canGoToStartLine = preStartingLine(
            frame,
            TpreStart,
            PRE_START_Y_MIN,
            PRE_START_Y_MAX,
            READY_HOLD_TIME,
        )

        if canGoToStartLine and canLeavePreStartLine:
            playsound(GO_SOUND)
            canLeavePreStartLine = False
            TSinceGoToStart = time.time()

        if TSinceGoToStart:
            elapsed = time.time() - TSinceGoToStart

            if crossedLine(frame, START_LINE_Y) and not crossedTooEarly and elapsed >= 2:
                playsound(FALSE_START_SOUND)
                crossedTooEarly = True

            if elapsed >= 5 and isReady(frame, START_LINE_Y, IMAGINARY_START_LINE_OFFSET) and not readySoundHeard:
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
