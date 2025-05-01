#importing different modules
import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) #capturing the video default webcam


#landmarks that are used to store last 5 landmaarks
landMarks = []



movement_threshold = 0.02

#list to store current movements according to previous landmarks
movements = []


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
     
          #storing current landmarks
          currLandmarks = np.array([
               (result.pose_landmarks.landmark[i].x,
               result.pose_landmarks.landmark[i].y,
               result.pose_landmarks.landmark[i].z)
               for i in key_indices
        ])


          #caclulating movements with respect of last 5 landmarks and adding to movements
          for index, landmark in enumerate(landMarks, start=0):
               if landmark is not None:
                    currMovement = np.linalg.norm(currLandmarks - landmark, axis=1).mean()
                    movements.append(currMovement)

          for movement in movements:
               if movement > movement_threshold:
                    print(f"a movement here! {trackMovement}")
                    trackMovement += 1
                    cv2.putText(frame, 'Significant Movement!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2, cv2.LINE_AA)

          landMarks_size = len(landMarks)
          i = landMarks_size -1

          #shifting the landmarks data acc to new frame
          if len(landMarks) >= 5:
               landMarks.pop()  # Remove the oldest landmark set
          landMarks.insert(0, currLandmarks)  # Add the latest landmarks at the beginning

          
    cv2.imshow('Filtered Movement Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


