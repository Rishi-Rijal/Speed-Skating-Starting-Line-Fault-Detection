import cv2
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

def get_pose_landmarks(frame, pose_processor):
    """
    Extracts pose landmarks from the frame using a provided pose processor.
    Args:
        frame (np.array): BGR image.
        pose_processor: Initialized MediaPipe Pose object.
    Returns:
        MediaPipe pose landmarks result object or None.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_processor.process(rgb)
    return result.pose_landmarks

def get_landmark_center(landmarks, frame, landmark_ids):
    """
    Gets the center (x, y) pixel coordinate of a list of landmarks.
    Args:
        landmarks: MediaPipe pose landmarks object.
        frame (np.array): Image frame.
        landmark_ids (list): List of PoseLandmark IDs.
    Returns:
        tuple (int, int) or None: (x, y) coordinate in pixels or None.
    """
    if landmarks is None:
        return None
    
    points = []
    h, w, _ = frame.shape
    for lm_id in landmark_ids:
        lm = landmarks.landmark[lm_id]
        if lm.visibility > 0.5: # Only consider visible landmarks
            points.append((int(lm.x * w), int(lm.y * h)))

    if not points:
        return None
        
    center = np.mean(points, axis=0).astype(int)
    return tuple(center)

def transform_point(point, homography_matrix):
    """
    Transforms a single (x, y) point using the homography matrix.
    Args:
        point (tuple): (x, y) coordinate.
        homography_matrix (np.array): The 3x3 homography matrix.
    Returns:
        tuple or None: Transformed (x, y) coordinate or None.
    """
    if point is None:
        return None
    # cv2.perspectiveTransform expects a 3D array: (1, N, 2)
    point_to_transform = np.float32([[point]])
    transformed_point = cv2.perspectiveTransform(point_to_transform, homography_matrix)
    return transformed_point[0][0]