import time
from typing import List, Optional, Tuple

import mediapipe as mp
import numpy as np

from utils import draw_lines, get_pose_landmarks, get_landmark_y

mp_pose = mp.solutions.pose


def get_landmark_y_center(frame: np.ndarray, landmark_ids: List[mp_pose.PoseLandmark]) -> Optional[int]:
    """Return the average Y pixel position for the given landmarks."""
    landmarks = get_pose_landmarks(frame)
    if landmarks is None:
        return None
    y_vals = [get_landmark_y(landmarks, frame, lid) for lid in landmark_ids]
    return int(np.mean(y_vals))


def crossedLine(frame: np.ndarray, start_line_y: int) -> bool:
    """Check if both ankles have crossed the start line."""
    ankles_y = get_landmark_y_center(
        frame, [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]
    )
    return ankles_y is not None and ankles_y > start_line_y


def preStartingLine(
    frame: np.ndarray,
    start_time: Optional[float],
    y_min: int,
    y_max: int,
    hold_time: int = 3,
) -> Tuple[Optional[float], bool]:
    """Check if the skater holds position between y_min and y_max for hold_time."""
    go_to_start = False
    hips_y = get_landmark_y_center(
        frame, [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
    )
    status = "Not Ready"
    if hips_y is None:
        return None, False

    if y_min < hips_y < y_max:
        if start_time is None:
            start_time = time.time()
        elapsed = time.time() - start_time
        if elapsed >= hold_time:
            status = "Ready"
            go_to_start = True
    else:
        start_time = None

    draw_lines(frame, y_min, y_max, status)
    return start_time, go_to_start


def isReady(
    frame: np.ndarray,
    start_line_y: int,
    offset: int,
) -> bool:
    """Return True when hips are between the imaginary and start line."""
    hips_y = get_landmark_y_center(
        frame, [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
    )
    if hips_y is None:
        return False
    imaginary_line = start_line_y - offset
    draw_lines(frame, imaginary_line, start_line_y, "")
    return imaginary_line <= hips_y <= start_line_y and not crossedLine(frame, start_line_y)

