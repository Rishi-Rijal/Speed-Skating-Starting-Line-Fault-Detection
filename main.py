import argparse
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
from config_utils import load_config


def parse_args():
    """Return parsed command line arguments."""
    parser = argparse.ArgumentParser(
        description="Speed Skating starting line fault detection"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="OpenCV camera index to use"
    )
    parser.add_argument(
        "--config", default="config.json", help="Path to configuration file"
    )
    parser.add_argument("--threshold", type=float, help="Movement threshold")
    parser.add_argument("--pre-start-min", type=int, help="Pre-start line min")
    parser.add_argument("--pre-start-max", type=int, help="Pre-start line max")
    parser.add_argument("--start-line", type=int, help="Start line position")
    parser.add_argument(
        "--imaginary-offset", type=int, help="Offset for imaginary start line"
    )
    return parser.parse_args()



def playsound(filename):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound {filename}: {e}")

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply command line overrides
    if args.threshold is not None:
        cfg["threshold"] = args.threshold
    if args.pre_start_min is not None:
        cfg["preStartMin"] = args.pre_start_min
    if args.pre_start_max is not None:
        cfg["preStartMax"] = args.pre_start_max
    if args.start_line is not None:
        cfg["startLine"] = args.start_line
    if args.imaginary_offset is not None:
        cfg["imaginaryOffset"] = args.imaginary_offset

    cap = cv2.VideoCapture(args.camera)
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

        isMovement(
            frame, landMarks, movements, cfg["threshold"], fullBody=True
        )
        draw_lines(frame, cfg["preStartMin"], cfg["preStartMax"], state="done")
        imaginary_line = cfg["startLine"] - cfg["imaginaryOffset"]
        draw_lines(frame, imaginary_line, cfg["startLine"], "yayy")
        TpreStart, canGoToStartLine = preStartingLine(
            frame,
            TpreStart,
            cfg["preStartMin"],
            cfg["preStartMax"],
            cfg["readyHoldTime"],
        )

        if canGoToStartLine and canLeavePreStartLine:
            playsound(cfg["goSound"])
            canLeavePreStartLine = False
            TSinceGoToStart = time.time()

        if TSinceGoToStart:
            elapsed = time.time() - TSinceGoToStart

            if crossedLine(frame, cfg["startLine"]) and not crossedTooEarly and elapsed >= 2:
                playsound(cfg["falseStartSound"])
                crossedTooEarly = True

            if (
                elapsed >= cfg["readyToStartDelay"]
                and isReady(frame, cfg["startLine"], cfg["imaginaryOffset"])
                and not readySoundHeard
            ):
                playsound(cfg["readySound"])
                readySoundHeard = True
                TSinceReadySound = time.time()

        if TSinceReadySound:
            start_elapsed = time.time() - TSinceReadySound
            startOK = cfg["startOkMin"] <= start_elapsed <= cfg["startOkMax"]

            if startOK and not falseStartOnce:
                if isMovement(
                    frame, landMarks, movements, cfg["threshold"], fullBody=True
                ):
                    playsound(cfg["falseStartSound"])
                    falseStartOnce = True

            if not falseStartOnce and start_elapsed > cfg["startOkMax"] and not gameStarted:
                playsound(cfg["gunSound"])
                gameStarted = True

        cv2.imshow('Speed Skating Full-Body Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
