#importing different modules
import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) #capturing the video default webcam

prev_landmarks1 = None
prev_landmarks2 = None
prev_landmarks3 = None
prev_landmarks4 = None
prev_landmarks5 = None

movement_threshold = 0.02
movement = 0
sec_movement = 0
thi_movement = 0
fou_movement = 0
fiv_movement = 0

prev_landmarks = None

trackMovement = 0

key_indices = [
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
]

while cap.isOpened():
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)

        landmarks = np.array([
            (result.pose_landmarks.landmark[i].x,
             result.pose_landmarks.landmark[i].y,
             result.pose_landmarks.landmark[i].z)
            for i in key_indices
        ])

        if prev_landmarks1 is not None:
            movement = np.linalg.norm(landmarks - prev_landmarks1, axis=1).mean()

        if prev_landmarks2 is not None:
             sec_movement = np.linalg.norm(landmarks - prev_landmarks2, axis=1).mean()
        
        if prev_landmarks3 is not None:
             thi_movement = np.linalg.norm(landmarks - prev_landmarks3, axis=1).mean()
        
        if prev_landmarks4 is not None:
             fou_movement = np.linalg.norm(landmarks - prev_landmarks4, axis=1).mean()

        if prev_landmarks5 is not None:
             fiv_movement = np.linalg.norm(landmarks - prev_landmarks5, axis=1).mean()
        
    
        if movement > movement_threshold:
                print(f"a movement here! {trackMovement}")
                trackMovement += 1
                cv2.putText(frame, 'Significant Movement!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if sec_movement > movement_threshold:
             cv2.putText(frame, 'Significant Movement!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        if thi_movement > movement_threshold:
             cv2.putText(frame, 'Significant Movement!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        if fou_movement > movement_threshold:
             cv2.putText(frame, 'Significant Movement!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        if fiv_movement > movement_threshold:
             cv2.putText(frame, 'Significant Movement!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        prev_landmarks5 = prev_landmarks4    
        prev_landmarks4 = prev_landmarks3
        prev_landmarks3 = prev_landmarks2
        prev_landmarks2 = prev_landmarks1
        prev_landmarks1 = landmarks
        #prev_landmarks = landmarks
    cv2.imshow('Filtered Movement Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


