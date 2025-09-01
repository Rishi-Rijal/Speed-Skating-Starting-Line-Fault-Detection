import cv2
import numpy as np

# --- Configuration ---
CAMERA_INDEX = 0  
OUTPUT_MATRIX_FILE = 'homography_matrix.npy'
OUTPUT_WIDTH = 1000
OUTPUT_HEIGHT = 500

# Global variables
clicked_points = []
paused_frame = None

def click_event(event, x, y, flags, params):
    """Callback function to record mouse clicks on a paused frame."""
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and paused_frame is not None and len(clicked_points) < 4:
        print(f"Point {len(clicked_points) + 1}/4 selected: ({x}, {y})")
        clicked_points.append((x, y))
        # Draw a circle on the paused frame to show the click
        cv2.circle(paused_frame, (x, y), 5, (0, 255, 0), -1)

def main():
    """Main function to perform setup from a live feed."""
    global clicked_points, paused_frame
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {CAMERA_INDEX}")
        return

    window_name = 'Live Setup - [SPACE] to Freeze | [R] to Reset | [S] to Save | [Q] to Quit'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)

    while True:
        if paused_frame is None:
            # If not paused, read a new frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break
        else:
            # If paused, use the stored paused_frame
            frame = paused_frame.copy()

        # Display instructions on the frame
        cv2.putText(frame, "Press [SPACE] to freeze/unfreeze", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press [R] to reset points", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press [S] to save matrix", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar
            if paused_frame is None:
                paused_frame = frame.copy()
                print("Frame paused. You can now click the 4 corners.")
            else:
                paused_frame = None
                clicked_points = [] # Reset points when unfreezing
                print("Live feed resumed.")
        elif key == ord('r'):
            if paused_frame is not None:
                clicked_points = []
                # To show the circles are gone, we need to reset the paused_frame
                _, paused_frame = cap.read() # Read a fresh frame to pause on
                print("Points reset. Please select 4 new points.")
        elif key == ord('s'):
            if len(clicked_points) == 4:
                print("Calculating and saving homography matrix...")
                source_points = np.float32(clicked_points)
                destination_points = np.float32([
                    [0, 0], [OUTPUT_WIDTH - 1, 0],
                    [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1], [0, OUTPUT_HEIGHT - 1]
                ])

                H, _ = cv2.findHomography(source_points, destination_points)
                np.save(OUTPUT_MATRIX_FILE, H)
                print(f"Matrix saved successfully to '{OUTPUT_MATRIX_FILE}'")
                break # Exit after saving
            else:
                print("Error: You must select exactly 4 points on a paused frame before saving.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()