import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def get_pose_landmarks(frame):
    """Extracts pose landmarks from the frame.
    Args:
        frame (np.array): BGR image.

    Returns:
        np.array or None: Array of landmark (x, y, z) tuples or None.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if result.pose_landmarks:
        return np.array([(lm.x, lm.y, lm.z) for lm in result.pose_landmarks.landmark])
    return None

def get_landmark_y(landmarks, frame, landmark_id):
    """Converts landmark y-position to pixel space.
    Args:
        landmarks (np.array): Array of pose landmarks.
        frame (np.array): Image frame.
        landmark_id (PoseLandmark): ID of the landmark.

    Returns:
        int: Y coordinate in pixels.
    """
    return int(landmarks[landmark_id.value][1] * frame.shape[0])

def draw_lines(frame, pre_start_y, start_y, state):
    """Draws horizontal lines and status text.
    Args:
        frame (np.array): Image frame.
        pre_start_y (int): Y position for the pre-start line.
        start_y (int): Y position for the start line.
        state (str): State label to show on frame.
    """
    cv2.line(frame, (0, pre_start_y), (frame.shape[1], pre_start_y), (255, 0, 0), 2)
    cv2.line(frame, (0, start_y), (frame.shape[1], start_y), (0, 255, 0), 2)
    cv2.putText(frame, f'State: {state}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
