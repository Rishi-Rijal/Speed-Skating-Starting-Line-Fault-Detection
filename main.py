import cv2
import numpy as np
import mediapipe as mp
import time
import json
import pygame

from movement import is_movement
from utils import get_pose_landmarks, get_landmark_center, transform_point

# --- Configuration & Setup ---
def load_config():
    """Loads settings from config.json"""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found! Using default values.")
        return {
            "preStartMin": 180, "preStartMax": 220,
            "startLine": 400, "threshold": 0.02
        }

def load_homography_matrix(path='homography_matrix.npy'):
    """Loads the homography matrix."""
    try:
        return np.load(path)
    except FileNotFoundError:
        print(f"Error: Homography matrix '{path}' not found!")
        print("Please run setup_homography.py first to create it.")
        return None

def main():
    # Load configuration and homography
    config = load_config()
    H = load_homography_matrix()
    if H is None:
        return

    # Initialize Pygame Mixer for sound
    pygame.mixer.init()
    

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose_processor = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Video Capture
    cap = cv2.VideoCapture(0) 

    # State machine and timing variables
    state = "WAITING_FOR_SKATER"
    state_timer = None
    landmark_history = []
    movement_history = []
    
    # Constants for the map visualization
    MAP_WIDTH = 1000
    MAP_HEIGHT = 500

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get pose landmarks from the current frame
        landmarks = get_pose_landmarks(frame, pose_processor)
        
        # Get the center of the skater's ankles in the original frame
        ankles_center_orig = get_landmark_center(landmarks, frame, 
                                            [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE])

        # Transform the ankle center to the top-down map view
        ankles_center_map = transform_point(ankles_center_orig, H)
        
        movement = is_movement(frame, pose_processor, landmark_history, movement_history, config['threshold'])

        # --- State Machine Logic ---
        if ankles_center_map is not None:
            if state == "WAITING_FOR_SKATER":
                # Check if skater is in the pre-start zone
                if config["preStartMin"] < ankles_center_map[1] < config["preStartMax"]:
                    state = "IN_PRE_START"
                    state_timer = time.time()
            
            elif state == "IN_PRE_START":
                # Check if skater has held position for 3 seconds
                if time.time() - state_timer > 3:
                    # playsound(GO_SOUND) # Example sound
                    print("STATE CHANGE: Skater can now go to the start line.")
                    state = "APPROACHING_START"
                # If skater leaves the zone too early, reset
                elif not (config["preStartMin"] < ankles_center_map[1] < config["preStartMax"]):
                    state = "WAITING_FOR_SKATER"
            
            elif state == "APPROACHING_START":
                # Logic for when the skater moves to the start line
                # For this example, we'll just check if they are near the line
                ready_zone_y = config["startLine"]
                if ready_zone_y - 50 < ankles_center_map[1] < ready_zone_y:
                    print("STATE CHANGE: Skater is set at the start line.")
                    # playsound(READY_SOUND)
                    state = "SET"
                    state_timer = time.time()
                    movement_history.clear() # Clear movement history before checking for false start
            
            elif state == "SET":
                # Window to wait for the gun (e.g., between 3 and 4.1 seconds)
                elapsed = time.time() - state_timer
                if 1 < elapsed < 4: # Shortened window for easier testing
                    if movement:
                        print("STATE CHANGE: FALSE START!")
                        # playsound(FALSE_START_SOUND)
                        state = "FALSE_START"
                elif elapsed >= 4:
                    print("STATE CHANGE: GO!")
                    # playsound(GUN_SOUND)
                    state = "RACE_ACTIVE"

            # Add logic for FALSE_START and RACE_ACTIVE states if needed
            
        # --- Visualization ---
        # Create a blank image for our top-down map
        map_view = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
        # Draw the lines on the map
        cv2.line(map_view, (0, config["preStartMin"]), (MAP_WIDTH, config["preStartMin"]), (255, 0, 0), 2)
        cv2.line(map_view, (0, config["preStartMax"]), (MAP_WIDTH, config["preStartMax"]), (255, 0, 0), 2)
        cv2.line(map_view, (0, config["startLine"]), (MAP_WIDTH, config["startLine"]), (0, 255, 0), 2)
        
        # Draw the skater's position on the map
        if ankles_center_map is not None:
            cv2.circle(map_view, tuple(ankles_center_map.astype(int)), 10, (0, 255, 255), -1)

        # Display the state on the main video frame
        cv2.putText(frame, f'State: {state}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Live Camera View', frame)
        cv2.imshow('Top-Down Map View', map_view)

        if cv2.waitKey(1) & 0xFF == 27: # Press ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()