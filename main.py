import cv2
import numpy as np
import mediapipe as mp
import time
import json
import pygame

from movement import is_movement
from utils import get_pose_landmarks, get_landmark_center, transform_point
from mediapipe.python.solutions.pose import PoseLandmark as PL

# Sound files 
GO_SOUND = "Sounds/goSound.mp3"               # also used for "Go to the start" for now
READY_SOUND = "Sounds/readySound.mp3"
FALSE_START_SOUND = "Sounds/falseStartBuzzer.mp3"
SECOND_SHOT_SOUND = "Sounds/falseStartBuzzer.mp3"  # also reuse if no separate "second shot" file

# Mixer cache (module-level, used by AudioGate)
_mixer_ready = False
_sound_cache = {}

# Config / calibration helpers
def load_config():
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found! Using default values.")
        return {
            "preStartMin": 180, "preStartMax": 220,
            "startLine": 400, "threshold": 0.02,
            "microTremor": 0.008, "settleBreathSeconds": 1.0,
            "readyAssumeTimeout": 3.0, "holdPauseSeconds": 1.10,
            "innerOnLeft": True
        }

def load_homography_matrix(path='homography_matrix.npy'):
    try:
        return np.load(path)
    except FileNotFoundError:
        print(f"Error: Homography matrix '{path}' not found! Run setup_homography.py.")
        return None

# Non-blocking AudioGate
class AudioGate:
    """
    Non-blocking audio manager:
      - play(path): start sound asynchronously on a dedicated channel
      - is_done(): True once the sound finished (or if audio failed)
      - stop(): stop current sound immediately
    Use pattern: trigger sound, switch to a tiny wait-state, and advance only when is_done() returns True.
    """
    def __init__(self):
        self.channel = None
        self._ensure()

    def _ensure(self):
        global _mixer_ready
        if not _mixer_ready:
            # pre_init lowers latency and avoids first-play hiccup
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            _mixer_ready = True

    def play(self, path: str) -> bool:
        """Start playing asynchronously; remember the channel. Returns True if started (False if failed)."""
        try:
            self._ensure()
            snd = _sound_cache.get(path)
            if snd is None:
                snd = pygame.mixer.Sound(path)
                _sound_cache[path] = snd
            # Use/find a dedicated channel so we can poll it
            if self.channel is None:
                self.channel = pygame.mixer.find_channel(True)  # allocate if needed
            self.channel.play(snd)
            return True
        except Exception as e:
            print(f"[Sound error] {e}")
            self.channel = None
            return False

    def is_done(self) -> bool:
        """Non-blocking: returns True once the audio has finished (or if playback failed)."""
        return (self.channel is None) or (not self.channel.get_busy())

    def stop(self):
        """Optional: stop current audio immediately."""
        if self.channel is not None:
            self.channel.stop()

# Geometry / lane helpers
def get_map_point_for_landmarks(frame, pose, ids, H):
    pt = get_landmark_center(pose, frame, ids)
    return transform_point(pt, H) if pt is not None else None

def crosses_or_touches_start(start_y, point_map_y):
    # "Touches/crosses" means map y >= start line y (flip sign if your homography has inverted y)
    return point_map_y is not None and point_map_y >= start_y

def lane_for_x(x_map, map_width=1000, inner_on_left=True):
    if x_map is None:
        return None
    mid = map_width / 2
    left_side = x_map < mid
    return ("inner" if left_side else "outer") if inner_on_left else ("outer" if left_side else "inner")

# Main loop
def main():
    config = load_config()
    H = load_homography_matrix()
    if H is None:
        return

    mp_pose = mp.solutions.pose
    pose_processor = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(4)

    # FSM
    state = "WAITING_FOR_SKATER"  # Skaters behind blue line
    state_timer = None

    # Movement buffers
    landmark_history = []
    movement_history = []

    # False start tracking (per pair)
    false_start_count_by_pair = 0

    # Crossing memory: once the line is touched after READY (even if it goes back) → still a false start
    touched_line_after_ready = False

    # Constants for the map visualization
    MAP_WIDTH = 1000
    MAP_HEIGHT = 500

    # Lane memory for the offender
    offender_lane = None
    last_false_reason = ""

    # Audio (non-blocking)
    audio_gate = AudioGate()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Pose landmarks for the *current* visible skater(s)
        landmarks = get_pose_landmarks(frame, pose_processor)

        # Ankle center (map)
        ankles_center_orig = get_landmark_center(landmarks, frame, [PL.LEFT_ANKLE, PL.RIGHT_ANKLE])
        ankles_center_map = transform_point(ankles_center_orig, H) if ankles_center_orig else None

        # Hands (map) for line touch
        left_hand_map  = get_map_point_for_landmarks(frame, landmarks, [PL.LEFT_WRIST, PL.LEFT_INDEX], H) if landmarks else None
        right_hand_map = get_map_point_for_landmarks(frame, landmarks, [PL.RIGHT_WRIST, PL.RIGHT_INDEX], H) if landmarks else None

        # Movement signal (tremor-tolerant)
        significant_move, smoothed_mag = is_movement(
            frame, pose_processor, landmark_history, movement_history,
            config['threshold'], config['microTremor']
        )

        # Determine lane of the detected skater (if any)
        current_lane = None
        if ankles_center_map is not None:
            current_lane = lane_for_x(ankles_center_map[0], MAP_WIDTH, config.get("innerOnLeft", True))

        # --- STATE MACHINE ---
        now = time.time()
        if state == "WAITING_FOR_SKATER":
            # Expect skaters in pre-start zone (behind blue line)
            if ankles_center_map is not None and config["preStartMin"] < ankles_center_map[1] < config["preStartMax"]:
                state = "IN_PRE_START"
                state_timer = now
                touched_line_after_ready = False
                offender_lane = None
                last_false_reason = ""
                print("Skaters behind blue line. Waiting...")

        elif state == "IN_PRE_START":
            # Hold behind the blue line for ~3s
            if ankles_center_map is None or not (config["preStartMin"] < ankles_center_map[1] < config["preStartMax"]):
                state = "WAITING_FOR_SKATER"
            elif now - state_timer >= 3.0:
                print('Starter (to skaters): "Go to the start"')
                audio_gate.play(GO_SOUND)  # non-blocking
                state = "SAY_GO_TO_START"  # wait here until sound ends

        elif state == "SAY_GO_TO_START":
            if audio_gate.is_done():
                state = "APPROACHING_START"
                state_timer = now  # start approach timing when cue has finished

        elif state == "APPROACHING_START":
            # They approach the start line (green in your map). Once at line: settle/breath.
            ready_zone_y = config["startLine"]
            if ankles_center_map is not None and (ready_zone_y - 50) < ankles_center_map[1] < (ready_zone_y + 2):
                print("Skaters at start line. Wait for settle/breath...")
                state = "SETTLE_BEFORE_READY"
                state_timer = now

        elif state == "SETTLE_BEFORE_READY":
            if now - state_timer >= config["settleBreathSeconds"]:
                # Raise gun, say READY
                print('Starter: *raises gun*')
                print('Starter (clearly): "Ready."')
                audio_gate.play(READY_SOUND)  # non-blocking
                state = "AFTER_READY_AUDIO"   # start the timer only after audio completes
            # If they leave the line, back to approach
            elif ankles_center_map is None or ankles_center_map[1] < (config["startLine"] - 60):
                state = "APPROACHING_START"

        elif state == "AFTER_READY_AUDIO":
            if audio_gate.is_done():
                state = "READY_WAIT_POSITION"
                state_timer = now   # start the ready-assume timeout now
                touched_line_after_ready = False

        elif state == "READY_WAIT_POSITION":
            # They should assume the start within timeout
            elapsed = now - state_timer

            # Check illegal line touch after READY (hands or feet touching/crossing)
            if ankles_center_map is not None and crosses_or_touches_start(config["startLine"], ankles_center_map[1]):
                touched_line_after_ready = True
                offender_lane = offender_lane or current_lane

            if left_hand_map is not None and crosses_or_touches_start(config["startLine"], left_hand_map[1]):
                touched_line_after_ready = True
                offender_lane = offender_lane or current_lane
            if right_hand_map is not None and crosses_or_touches_start(config["startLine"], right_hand_map[1]):
                touched_line_after_ready = True
                offender_lane = offender_lane or current_lane

            if touched_line_after_ready:
                # Immediate false start
                last_false_reason = "Crossing the line"
                state = "FALSE_START"
                # do not 'continue' here; let next loop handle FALSE_START state
            else:
                # If they keep moving too much (not getting into position) until timeout → false start
                if elapsed > config["readyAssumeTimeout"]:
                    if significant_move or smoothed_mag > config["microTremor"]:
                        last_false_reason = "Going down too slow"
                        offender_lane = offender_lane or current_lane
                        state = "FALSE_START"
                # If they look stable enough (below tremor floor) → enter HOLD (1.1s)
                elif smoothed_mag <= config["microTremor"]:
                    state = "HOLD_BEFORE_GUN"
                    state_timer = now
                    print("Both skaters appear set. Holding...")

        elif state == "HOLD_BEFORE_GUN":
            hold_elapsed = now - state_timer

            # Any significant movement in the hold → Not stable
            if significant_move and smoothed_mag > (config["microTremor"] * 1.25):
                last_false_reason = "Not stable"
                offender_lane = offender_lane or current_lane
                state = "FALSE_START"

            # Any touch/cross during the hold?
            elif ankles_center_map is not None and crosses_or_touches_start(config["startLine"], ankles_center_map[1]):
                last_false_reason = "Crossing the line"
                offender_lane = offender_lane or current_lane
                state = "FALSE_START"

            elif left_hand_map is not None and crosses_or_touches_start(config["startLine"], left_hand_map[1]):
                last_false_reason = "Crossing the line"
                offender_lane = offender_lane or current_lane
                state = "FALSE_START"

            elif right_hand_map is not None and crosses_or_touches_start(config["startLine"], right_hand_map[1]):
                last_false_reason = "Crossing the line"
                offender_lane = offender_lane or current_lane
                state = "FALSE_START"

            # Hold complete → fire gun
            elif hold_elapsed >= config["holdPauseSeconds"]:
                print("** GUN FIRED **")
                audio_gate.play(GO_SOUND)   # non-blocking
                state = "AFTER_GUN_AUDIO"

        elif state == "AFTER_GUN_AUDIO":
            if audio_gate.is_done():
                state = "RACE_ACTIVE"

        elif state == "RACE_ACTIVE":
            # If skater visibly launched before gun, we would have already caught it in HOLD.
            # Show state, then wait for ESC or next reset (or future race logic).
            pass

        elif state == "FALSE_START":
            # Signal with second shot / whistle (non-blocking)
            print("** FALSE START **")
            audio_gate.play(FALSE_START_SOUND)
            state = "AFTER_FALSE_AUDIO"

        elif state == "AFTER_FALSE_AUDIO":
            if audio_gate.is_done():
                false_start_count_by_pair += 1
                lane_txt = offender_lane or (current_lane or "inner/outer")
                reason = last_false_reason or "Started before gun"

                # Announce reason
                print(f'Starter: "False start, {lane_txt} lane"')
                print(f'Starter: "{reason}"')

                if false_start_count_by_pair >= 2:
                    print(f'Starter: "False start, {lane_txt} lane, disqualified"')
                    print("→ Send the other skater again.")
                    # Reset pair count for next pairing
                    false_start_count_by_pair = 0

                # Restart from the beginning of the procedure
                state = "WAITING_FOR_SKATER"
                state_timer = None
                movement_history.clear()
                landmark_history.clear()
                touched_line_after_ready = False
                offender_lane = None
                last_false_reason = ""
                print('Starter: "Back to lanes — new start."')

        # --- Visualization ---
        map_view = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
        cv2.line(map_view, (0, config["preStartMin"]), (MAP_WIDTH, config["preStartMin"]), (255, 0, 0), 2)
        cv2.line(map_view, (0, config["preStartMax"]), (MAP_WIDTH, config["preStartMax"]), (255, 0, 0), 2)
        cv2.line(map_view, (0, config["startLine"]), (MAP_WIDTH, config["startLine"]), (0, 255, 0), 2)

        if ankles_center_map is not None:
            cv2.circle(map_view, tuple(np.array(ankles_center_map, dtype=int)), 10, (0, 255, 255), -1)

        cv2.putText(frame, f'State: {state}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live Camera View', frame)
        cv2.imshow('Top-Down Map View', map_view)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC quits
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()