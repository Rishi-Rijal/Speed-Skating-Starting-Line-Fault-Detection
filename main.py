#importing different modules
import cv2
import numpy as np
import mediapipe as mp
import time
from movement import isMovement
from utils import draw_lines, get_landmark_y, get_pose_landmarks

cap = cv2.VideoCapture(0) #capturing the video default webcam
landMarks = [] #landmarks -> that are used to store last 5 landmarks
movement_threshold = 0.02
movements = [0,0,0,0,0] #list to store current movements according to previous landmarks

def main():
    start_time = None
    while cap.isOpened():
        ret, frame = cap.read()
        
        land_Marks = get_pose_landmarks(frame)
        NOSE_Y = get_landmark_y(land_Marks,frame,mp.solutions.pose.PoseLandmark.NOSE)
        status = "Not Ready"
        if NOSE_Y > 180 and NOSE_Y < 220:
            #print("correct position")

            if start_time is None:
                start_time = time.time()

            elapsed = time.time() - start_time
           # print(NOSE_Y > 200 and NOSE_Y < 300)
            if (NOSE_Y > 200 and NOSE_Y < 300) and elapsed >=3:
                status = "Ready"
                isMovement(frame,landMarks, movements, movementThreshold=movement_threshold)
            
            print(NOSE_Y < 200)
            if NOSE_Y < 200:
                print("rishi")
                start_time = time.time()
        else:
            status = "Not ready"
            
        draw_lines(frame, 200,400,status)
        cv2.imshow('Filtered Movement Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
