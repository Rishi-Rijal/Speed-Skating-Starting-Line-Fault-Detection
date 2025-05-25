import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Key landmarks to track for full-body mode
FULL_BODY_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.NOSE
]

def isMovement(openCVFrame, landMarks: list, movements: list, movementThreshold: float = 0.02, fullBody: bool = False) -> bool:
    """Detects significant movement using pose landmarks across frames.

    Args:
        openCVFrame (np.array): Current frame in BGR format.
        landMarks (list): History of landmark arrays from previous frames.
        movements (list): Buffer to store average movement values.
        movementThreshold (float): Threshold to classify significant movement.
        fullBody (bool): Whether to use selected landmark comparison.

    Returns:
        bool: True if movement exceeds threshold, else False.
    """
    RGBFrame = cv2.cvtColor(openCVFrame, cv2.COLOR_BGR2RGB)
    result = pose.process(RGBFrame)

    if not result.pose_landmarks:
        return False

    mp_drawing.draw_landmarks(openCVFrame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    allLandmarks = np.array([
        (lm.x, lm.y, lm.z) for lm in result.pose_landmarks.landmark
    ])

    selectedLandmarks = np.array([allLandmarks[lm.value] for lm in FULL_BODY_LANDMARKS]) if fullBody else np.array([allLandmarks[mp_pose.PoseLandmark.NOSE.value]])

    for index, landmark in enumerate(landMarks):
        if landmark is not None:
            prev = np.array([landmark[lm.value] for lm in FULL_BODY_LANDMARKS]) if fullBody else np.array([landmark[mp_pose.PoseLandmark.NOSE.value]])
            currMovement = np.linalg.norm(selectedLandmarks - prev, axis=1).mean()
            movements[index] = currMovement

    if len(landMarks) >= 5:
        landMarks.pop()
    landMarks.insert(0, allLandmarks)

    for movement in movements:
        if movement > movementThreshold:
            cv2.putText(openCVFrame, 'Significant Movement!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            return True

    return False
