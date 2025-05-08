import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def isMovement(openCVFrame, landMarks: list, movements: list, movementThreshold: float = 0.02) -> bool:
    """Takes a Frame in RGB format , movement threshold(default 0.02), Landmarks list and previous movements calculated"""
    RGBFrame = cv2.cvtColor(openCVFrame, cv2.COLOR_BGR2RGB)
    result = pose.process(RGBFrame)

    if not result.pose_landmarks:
        return False

    mp_drawing.draw_landmarks(openCVFrame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Store current landmarks
    currLandmarks = np.array([
        (lm.x, lm.y, lm.z) for lm in result.pose_landmarks.landmark
    ])

    # Calculate movement only if previous data is available and valid
    for index, landmark in enumerate(landMarks):
        if landmark is not None :
            currMovement = np.linalg.norm(currLandmarks - landmark, axis=1).mean()
            movements[index] = currMovement

    # Update landmark buffer
    if len(landMarks) >= 5:
        landMarks.pop()
    landMarks.insert(0, currLandmarks)

    # Movement threshold check
    for movement in movements:
        if movement > movementThreshold:
            print(f"A movement here!")
            cv2.putText(openCVFrame, 'Significant Movement!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            return True

    return False
