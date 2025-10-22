import cv2
import numpy as np
import mediapipe as mp
import time
import json
import pygame
import math

from movement import is_movement
from utils import get_pose_landmarks, get_landmark_center, transform_point
from mediapipe.python.solutions.pose import PoseLandmark as PL

# =========================
# Sound files
# =========================
GO_TO_THE_START_SOUND = "Sounds/go_to_the_start.mp3"
CROSSING_THE_LINE_SOUND = "Sounds/crossing_the_line.mp3"
DISQUALIFIED_SOUND = "Sounds/disqualified.mp3"
DOWN_TOO_SLOW_SOUND = "Sounds/down_too_slow.mp3"
FALSE_START_SOUND = "Sounds/false_start.mp3"
BUZZER_SOUND = "Sounds/buzzer.mp3"
GUN_SHOT_SOUND = "Sounds/gun_shot.mp3"
INNER_LANE_SOUND = "Sounds/inner_lane.mp3"
OUTER_LANE_SOUND = "Sounds/outer_lane.mp3"
READY_SOUND = "Sounds/ready.mp3"
SECOND_SHOT_SOUND = "Sounds/falseStartBuzzer.mp3"

# =========================
# Mixer cache (module-level, used by AudioGate)
# =========================
_mixer_ready = False
_sound_cache = {}

# =========================
# Config / calibration helpers
# =========================
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

# =========================
# Audio: reason mapping
# =========================
def reason_to_sound(reason: str):
    if reason == "Crossing the line":
        return CROSSING_THE_LINE_SOUND
    if reason == "Going down too slow":
        return DOWN_TOO_SLOW_SOUND
    if reason == "Not stable":
        return BUZZER_SOUND
    return None

# =========================
# Non-blocking AudioGate
# =========================
class AudioGate:
    """
    Non-blocking audio manager:
      - play(path): start sound asynchronously on a dedicated channel
      - is_done(): True once the sound finished (or if audio failed)
      - stop(): stop current sound immediately
    """
    def __init__(self):
        self.channel = None
        self._ensure()

    def _ensure(self):
        global _mixer_ready
        if not _mixer_ready:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            _mixer_ready = True

    def play(self, path: str) -> bool:
        try:
            self._ensure()
            snd = _sound_cache.get(path)
            if snd is None:
                snd = pygame.mixer.Sound(path)
                _sound_cache[path] = snd
            if self.channel is None:
                self.channel = pygame.mixer.find_channel(True)
            # Don't stomp an ongoing clip (comment this out to allow interrupting)
            if self.channel.get_busy():
                return False
            self.channel.play(snd)
            return True
        except Exception as e:
            print(f"[Sound error] {e}")
            self.channel = None
            return False

    def is_done(self) -> bool:
        return (self.channel is None) or (not self.channel.get_busy())

    def stop(self):
        if self.channel is not None:
            self.channel.stop()

# =========================
# Geometry / lane helpers
# =========================
def get_map_point_for_landmarks(frame, landmarks, ids, H):
    pt = get_landmark_center(landmarks, frame, ids)
    return transform_point(pt, H) if pt is not None else None

def crosses_or_touches_start(start_y, point_map_y):
    return point_map_y is not None and point_map_y >= start_y

def lane_for_x(x_map, map_width=1000, inner_on_left=True):
    if x_map is None:
        return None
    mid = map_width / 2
    left_side = x_map < mid
    return ("inner" if left_side else "outer") if inner_on_left else ("outer" if left_side else "inner")

# =========================
# LineTouchFilter (hysteresis + debounce)
# =========================
class LineTouchFilter:
    """
    Schmitt-trigger + debounce for touching a horizontal line (y = line_y).
    """
    def __init__(self, line_y: float, band_px: float = 4.0, press_ms: int = 60, release_ms: int = 80):
        self.line_y = float(line_y)
        self.band = float(band_px)
        self.press_s = press_ms / 1000.0
        self.release_s = release_ms / 1000.0
        self.state = False
        self._pending = None  # (target_state, t_start)

    def _schmitt(self, y: float | None) -> bool:
        if y is None:
            return self.state
        if not self.state:
            return y >= (self.line_y + self.band)
        else:
            return y >= (self.line_y - self.band)

    def sample(self, y: float | None, now_s: float | None = None) -> bool:
        if now_s is None:
            now_s = time.perf_counter()
        raw = self._schmitt(y)

        if raw == self.state:
            self._pending = None
            return self.state

        if self._pending is None or self._pending[0] != raw:
            self._pending = (raw, now_s)

        need = self.press_s if raw else self.release_s
        if (now_s - self._pending[1]) >= need:
            self.state = raw
            self._pending = None

        return self.state

# =========================
# MovementGate (normalize + EMA + hysteresis + debounce)
# =========================
class MovementGate:
    """
    Filters a scalar motion magnitude with:
      - scale normalization (divide by body size)
      - EMA smoothing
      - deadband + hysteresis
      - debounce (sustain times for ON/OFF)
    """
    def __init__(
        self,
        deadband=0.010,      # ignore < 1.0% of body size
        on_thresh=0.020,     # enter "moving" above 2.0%
        off_thresh=0.012,    # exit "moving" below 1.2%
        ema_alpha=0.35,      # 0..1, higher = snappier
        sustain_on_ms=120,   # require ON for >= 120 ms
        sustain_off_ms=250   # require OFF for >= 250 ms
    ):
        self.deadband = deadband
        self.on = on_thresh
        self.off = off_thresh
        self.alpha = ema_alpha
        self.on_s = sustain_on_ms / 1000.0
        self.off_s = sustain_off_ms / 1000.0

        self.state = False
        self._ema = 0.0
        self._pending = None  # (target_state, t_start)

    def reset(self):
        self.state = False
        self._ema = 0.0
        self._pending = None

    def normalize_mag(self, mag, body_size):
        if body_size is None or body_size <= 1e-3:
            return 0.0 if mag is None else float(mag)
        return float(mag) / float(body_size)

    def sample(self, raw_mag, body_size, now_s):
        nm = self.normalize_mag(raw_mag or 0.0, body_size)
        # soft deadband
        if abs(nm) < self.deadband:
            nm = 0.0
        # EMA
        self._ema = self.alpha * nm + (1.0 - self.alpha) * self._ema

        # hysteresis
        want_on = self.state
        if not self.state:
            want_on = (self._ema >= self.on)
        else:
            want_on = (self._ema >= self.off)

        # debounce
        if want_on == self.state:
            self._pending = None
            return self.state, self._ema

        if self._pending is None or self._pending[0] != want_on:
            self._pending = (want_on, now_s)

        need = self.on_s if want_on else self.off_s
        if (now_s - self._pending[1]) >= need:
            self.state = want_on
            self._pending = None

        return self.state, self._ema

# =========================
# Utilities
# =========================
def estimate_body_size(landmarks, frame, fallback=300.0):
    """
    Returns a scale in pixels to normalize motion:
    average of shoulder width and hip width in the original image.
    """
    try:
        L = PL
        spans = []
        for a, b in [(L.LEFT_SHOULDER, L.RIGHT_SHOULDER),
                     (L.LEFT_HIP, L.RIGHT_HIP)]:
            p1 = get_landmark_center(landmarks, frame, [a])
            p2 = get_landmark_center(landmarks, frame, [b])
            if p1 is not None and p2 is not None:
                dx = p1[0] - p2[0]; dy = p1[1] - p2[1]
                spans.append(math.hypot(dx, dy))
        if spans:
            return sum(spans) / len(spans)
    except Exception:
        pass
    return fallback

# =========================
# Main loop
# =========================
def main():
    config = load_config()
    H = load_homography_matrix()
    if H is None:
        return

    mp_pose = mp.solutions.pose
    pose_processor = mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)

    # FSM
    state = "WAITING_FOR_SKATER"  # Skaters behind blue line
    state_timer = None

    # Movement buffers
    landmark_history = []
    movement_history = []

    # False start tracking (per pair)
    false_start_count_by_pair = 0

    # Crossing memory: once the line is touched after READY → still a false start
    touched_line_after_ready = False

    # Map constants (for drawing)
    MAP_WIDTH = 1000
    MAP_HEIGHT = 500

    # Lane memory for offender + reason
    offender_lane = None
    last_false_reason = ""

    # NEW: cues & warnings
    played_lane_cue = False
    warned_touch_after_ready = False

    # Audio
    audio_gate = AudioGate()

    # Filters: line touch (ankles + hands)
    ankle_touch_f = LineTouchFilter(config["startLine"], band_px=4, press_ms=60, release_ms=80)
    left_touch_f  = LineTouchFilter(config["startLine"], band_px=4, press_ms=60, release_ms=80)
    right_touch_f = LineTouchFilter(config["startLine"], band_px=4, press_ms=60, release_ms=80)

    # Movement gate
    movement_gate = MovementGate(
        deadband=0.010, on_thresh=0.020, off_thresh=0.012,
        ema_alpha=0.35, sustain_on_ms=120, sustain_off_ms=250
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()

            # Pose landmarks for the *current* visible skater(s)
            landmarks = get_pose_landmarks(frame, pose_processor)

            # Ankle center (map)
            ankles_center_orig = get_landmark_center(landmarks, frame, [PL.LEFT_ANKLE, PL.RIGHT_ANKLE])
            ankles_center_map = transform_point(ankles_center_orig, H) if ankles_center_orig else None

            # Hands (map) for line touch
            left_hand_map  = get_map_point_for_landmarks(frame, landmarks, [PL.LEFT_WRIST, PL.LEFT_INDEX], H) if landmarks else None
            right_hand_map = get_map_point_for_landmarks(frame, landmarks, [PL.RIGHT_WRIST, PL.RIGHT_INDEX], H) if landmarks else None

            # Movement signal (tremor-tolerant, from your existing function)
            _, smoothed_mag = is_movement(
                frame, pose_processor, landmark_history, movement_history,
                config['threshold'], config['microTremor']
            )

            # Normalize motion by body size and gate it
            body_size_px = estimate_body_size(landmarks, frame)
            moving, filtered_mag = movement_gate.sample(smoothed_mag, body_size_px, now)

            # Determine lane of the detected skater (if any)
            current_lane = None
            if ankles_center_map is not None:
                current_lane = lane_for_x(ankles_center_map[0], MAP_WIDTH, config.get("innerOnLeft", True))

            # Filtered line touches (use map y if available)
            ankle_y = ankles_center_map[1] if ankles_center_map is not None else None
            left_y  = left_hand_map[1] if left_hand_map is not None else None
            right_y = right_hand_map[1] if right_hand_map is not None else None

            ankle_touching = ankle_touch_f.sample(ankle_y, now)
            left_touching  = left_touch_f.sample(left_y, now)
            right_touching = right_touch_f.sample(right_y, now)

            # --- STATE MACHINE ---
            if state == "WAITING_FOR_SKATER":
                # Expect skaters in pre-start zone (behind blue line)
                if ankles_center_map is not None and config["preStartMin"] < ankles_center_map[1] < config["preStartMax"]:
                    state = "IN_PRE_START"
                    state_timer = now
                    touched_line_after_ready = False
                    offender_lane = None
                    last_false_reason = ""
                    played_lane_cue = False
                    warned_touch_after_ready = False
                    print("Skaters behind blue line. Waiting...")

            elif state == "IN_PRE_START":
                # Hold behind the blue line for ~3s
                in_band = (ankles_center_map is not None) and (config["preStartMin"] < ankles_center_map[1] < config["preStartMax"])
                if not in_band:
                    state = "WAITING_FOR_SKATER"
                else:
                    # NEW: announce lane once so athletes/spectators know who's where
                    if not played_lane_cue and current_lane:
                        lane_snd = INNER_LANE_SOUND if current_lane == "inner" else OUTER_LANE_SOUND
                        audio_gate.play(lane_snd)
                        played_lane_cue = True

                    if (now - state_timer) >= 3.0:
                        print('Starter (to skaters): "Go to the start"')
                        audio_gate.play(GO_TO_THE_START_SOUND)
                        state = "SAY_GO_TO_START"  # wait here until sound ends

            elif state == "SAY_GO_TO_START":
                if audio_gate.is_done():
                    state = "APPROACHING_START"
                    state_timer = now  # start approach timing when cue has finished

            elif state == "APPROACHING_START":
                # They approach the start line (green). Once at line: settle/breath.
                ready_zone_y = config["startLine"]
                if ankles_center_map is not None and (ready_zone_y - 50) < ankles_center_map[1] < (ready_zone_y + 2):
                    print("Skaters at start line. Wait for settle/breath...")
                    state = "SETTLE_BEFORE_READY"
                    state_timer = now

            elif state == "SETTLE_BEFORE_READY":
                if (now - state_timer) >= config["settleBreathSeconds"]:
                    # Raise gun, say READY
                    print('Starter: *raises gun*')
                    print('Starter (clearly): "Ready."')
                    audio_gate.play(READY_SOUND)
                    state = "AFTER_READY_AUDIO"   # start the timer only after audio completes
                # If they leave the line, back to approach
                elif ankles_center_map is None or ankles_center_map[1] < (config["startLine"] - 60):
                    state = "APPROACHING_START"

            elif state == "AFTER_READY_AUDIO":
                if audio_gate.is_done():
                    state = "READY_WAIT_POSITION"
                    state_timer = now   # start the ready-assume timeout now
                    touched_line_after_ready = False
                    warned_touch_after_ready = False

            elif state == "READY_WAIT_POSITION":
                # They should assume the start within timeout
                elapsed = now - state_timer

                # Check illegal line touch after READY (hands or feet touching/crossing)
                if (ankle_touching or left_touching or right_touching):
                    # NEW: quick warning beep the first time they touch after READY
                    if not warned_touch_after_ready:
                        audio_gate.play(BUZZER_SOUND)
                        warned_touch_after_ready = True

                    touched_line_after_ready = True
                    offender_lane = offender_lane or current_lane

                if touched_line_after_ready:
                    last_false_reason = "Crossing the line"
                    state = "FALSE_START"
                else:
                    # If they keep moving too much (not getting into position) until timeout → false start
                    if elapsed > config["readyAssumeTimeout"]:
                        if moving:
                            last_false_reason = "Going down too slow"
                            offender_lane = offender_lane or current_lane
                            state = "FALSE_START"
                    # If they look stable enough → enter HOLD
                    elif not moving:
                        state = "HOLD_BEFORE_GUN"
                        state_timer = now
                        print("Both skaters appear set. Holding...")

            elif state == "HOLD_BEFORE_GUN":
                hold_elapsed = now - state_timer

                # Any significant movement in the hold → Not stable
                if moving:
                    last_false_reason = "Not stable"
                    offender_lane = offender_lane or current_lane
                    state = "FALSE_START"

                # Any touch/cross during the hold?
                elif ankle_touching or left_touching or right_touching:
                    last_false_reason = "Crossing the line"
                    offender_lane = offender_lane or current_lane
                    state = "FALSE_START"

                # Hold complete → fire gun
                elif hold_elapsed >= config["holdPauseSeconds"]:
                    print("** GUN FIRED **")
                    audio_gate.play(GUN_SHOT_SOUND)
                    state = "AFTER_GUN_AUDIO"

            elif state == "AFTER_GUN_AUDIO":
                if audio_gate.is_done():
                    state = "RACE_ACTIVE"

            elif state == "RACE_ACTIVE":
                # future race logic here
                pass

            elif state == "FALSE_START":
                print("** FALSE START **")
                # Headline signal
                if not audio_gate.play(SECOND_SHOT_SOUND):
                    audio_gate.play(FALSE_START_SOUND)
                state = "AFTER_FALSE_AUDIO"

            elif state == "AFTER_FALSE_AUDIO":
                if audio_gate.is_done():
                    false_start_count_by_pair += 1
                    lane_txt = offender_lane or (current_lane or "inner/outer")
                    reason = last_false_reason or "Started before gun"

                    print(f'Starter: "False start, {lane_txt} lane"')
                    print(f'Starter: "{reason}"')

                    # Play lane call if we have a specific lane
                    lane_sound = None
                    if offender_lane == "inner":
                        lane_sound = INNER_LANE_SOUND
                    elif offender_lane == "outer":
                        lane_sound = OUTER_LANE_SOUND

                    if lane_sound and audio_gate.play(lane_sound):
                        state = "AFTER_LANE_AUDIO"
                    else:
                        # go straight to reason sound
                        rs = reason_to_sound(reason)
                        if rs and audio_gate.play(rs):
                            state = "AFTER_REASON_AUDIO"
                        else:
                            # No extra audio; check DQ next
                            if false_start_count_by_pair >= 2:
                                print(f'Starter: "False start, {lane_txt} lane, disqualified"')
                                print("→ Send the other skater again.")
                                audio_gate.play(DISQUALIFIED_SOUND)
                                state = "AFTER_DQ_AUDIO"
                            else:
                                # reset procedure (non-DQ) → cue restart
                                state = "WAITING_FOR_SKATER"
                                state_timer = None
                                movement_history.clear()
                                landmark_history.clear()
                                touched_line_after_ready = False
                                offender_lane = None
                                last_false_reason = ""
                                audio_gate.play(GO_TO_THE_START_SOUND)
                                print('Starter: "Back to lanes — new start."')

            elif state == "AFTER_LANE_AUDIO":
                if audio_gate.is_done():
                    # then the reason clip (if any)
                    rs = reason_to_sound(last_false_reason)
                    if rs and audio_gate.play(rs):
                        state = "AFTER_REASON_AUDIO"
                    else:
                        # No reason clip; move on to DQ check / reset
                        if false_start_count_by_pair >= 2:
                            print(f'Starter: "False start, {offender_lane or "inner/outer"} lane, disqualified"')
                            print("→ Send the other skater again.")
                            audio_gate.play(DISQUALIFIED_SOUND)
                            state = "AFTER_DQ_AUDIO"
                        else:
                            state = "WAITING_FOR_SKATER"
                            state_timer = None
                            movement_history.clear()
                            landmark_history.clear()
                            touched_line_after_ready = False
                            offender_lane = None
                            last_false_reason = ""
                            audio_gate.play(GO_TO_THE_START_SOUND)
                            print('Starter: "Back to lanes — new start."')

            elif state == "AFTER_REASON_AUDIO":
                if audio_gate.is_done():
                    if false_start_count_by_pair >= 2:
                        print(f'Starter: "False start, {offender_lane or "inner/outer"} lane, disqualified"')
                        print("→ Send the other skater again.")
                        audio_gate.play(DISQUALIFIED_SOUND)
                        state = "AFTER_DQ_AUDIO"
                    else:
                        state = "WAITING_FOR_SKATER"
                        state_timer = None
                        movement_history.clear()
                        landmark_history.clear()
                        touched_line_after_ready = False
                        offender_lane = None
                        last_false_reason = ""
                        audio_gate.play(GO_TO_THE_START_SOUND)
                        print('Starter: "Back to lanes — new start."')

            elif state == "AFTER_DQ_AUDIO":
                if audio_gate.is_done():
                    # Reset pair count for next pairing after a DQ
                    false_start_count_by_pair = 0
                    state = "WAITING_FOR_SKATER"
                    state_timer = None
                    movement_history.clear()
                    landmark_history.clear()
                    touched_line_after_ready = False
                    offender_lane = None
                    last_false_reason = ""
                    print('Starter: "Back to lanes — new start."')

            # --- Visualization ---
            MAP_HEIGHT = 500
            map_view = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
            cv2.line(map_view, (0, config["preStartMin"]), (MAP_WIDTH, config["preStartMin"]), (255, 0, 0), 2)
            cv2.line(map_view, (0, config["preStartMax"]), (MAP_WIDTH, config["preStartMax"]), (255, 0, 0), 2)
            cv2.line(map_view, (0, config["startLine"]), (MAP_WIDTH, config["startLine"]), (0, 255, 0), 2)

            if ankles_center_map is not None:
                cv2.circle(map_view, tuple(np.array(ankles_center_map, dtype=int)), 10, (0, 255, 255), -1)

            # Optional debug overlays
            cv2.putText(frame, f'State: {state}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Moving: {moving}  Mag(norm): {filtered_mag:.4f}', (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f'Touch A/L/R: {int(ankle_touching)}/{int(left_touching)}/{int(right_touching)}',
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow('Live Camera View', frame)
            cv2.imshow('Top-Down Map View', map_view)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC quits
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            pose_processor.close()
        except Exception:
            pass
        try:
            pygame.mixer.quit()
        except Exception:
            pass

if __name__ == "__main__":
    main()
