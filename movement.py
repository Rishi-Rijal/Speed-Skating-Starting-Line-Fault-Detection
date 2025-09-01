import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

FULL_BODY_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.NOSE
]

def is_movement(frame, pose_processor, landmark_history: list, movement_history: list, movement_threshold: float) -> bool:
    """Detects significant movement using pose landmarks across frames."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_processor.process(rgb_frame)

    if not result.pose_landmarks:
        return False

    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    current_landmarks = np.array([(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks.landmark])
    selected_landmarks = np.array([current_landmarks[lm.value][:2] for lm in FULL_BODY_LANDMARKS]) # Only use x, y for 2D movement

    movement_detected = False
    if landmark_history:
        prev_landmarks = landmark_history[0] # Compare with the most recent frame
        prev_selected = np.array([prev_landmarks[lm.value][:2] for lm in FULL_BODY_LANDMARKS])
        
        # Calculate Euclidean distance for each landmark
        curr_movement = np.linalg.norm(selected_landmarks - prev_selected, axis=1).mean()
        movement_history.append(curr_movement)
        if len(movement_history) > 5:
            movement_history.pop(0)

    # Update landmark history
    landmark_history.insert(0, current_landmarks)
    if len(landmark_history) > 5:
        landmark_history.pop()

    # Check for significant movement in recent history
    if any(m > movement_threshold for m in movement_history):
        cv2.putText(frame, 'MOVEMENT!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        movement_detected = True

    return movement_detected