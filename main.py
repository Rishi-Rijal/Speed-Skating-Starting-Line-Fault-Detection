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
            cfg = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found! Using default values.")
        cfg = {}

    # Defaults (incl. new axis-aware settings)
    cfg.setdefault("preStartMin", 180)
    cfg.setdefault("preStartMax", 220)
    cfg.setdefault("startLine", 400)
    cfg.setdefault("threshold", 0.02)
    cfg.setdefault("microTremor", 0.008)
    cfg.setdefault("settleBreathSeconds", 1.0)
    cfg.setdefault("readyAssumeTimeout", 3.0)
    cfg.setdefault("holdPauseSeconds", 1.10)
    cfg.setdefault("innerOnLeft", True)

    # NEW: orientation controls
    # - startAxis: 'y' (top camera; horizontal start) or 'x' (side camera; vertical start)
    # - laneAxis:  'x' or 'y' — which axis splits inner/outer
    cfg.setdefault("startAxis", "y")
    cfg.setdefault("laneAxis", "x")
    return cfg

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
            # Don't stomp an ongoing clip (comment to allow interrupting)
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

def axis_value(pt, axis: str):
    """Return the scalar coordinate along specified axis ('x' or 'y')."""
    if pt is None:
        return None
    return pt[0] if axis == "x" else pt[1]

def lane_for_axis(val, axis_len=1000, inner_on_left=True):
    """Inner/outer based on position along the chosen axis."""
    if val is None:
        return None
    mid = axis_len / 2
    on_low_side = val < mid
    return ("inner" if on_low_side else "outer") if inner_on_left else ("outer" if on_low_side else "inner")

# =========================
# LineTouchFilter (hysteresis + debounce)
# =========================
class LineTouchFilter:
    """
    Schmitt-trigger + debounce for touching a line along one axis.
    """
    def __init__(self, line_pos: float, band_units: float = 4.0, press_ms: int = 60, release_ms: int = 80):
        self.line = float(line_pos)
        self.band = float(band_units)
        self.press_s = press_ms / 1000.0
        self.release_s = release_ms / 1000.0
        self.state = False
        self._pending = None  # (target_state, t_start)

    def _schmitt(self, v: float | None) -> bool:
        if v is None:
            return self.state
        if not self.state:
            return v >= (self.line + self.band)
        else:
            return v >= (self.line - self.band)

    def sample(self, v: float | None, now_s: float | None = None) -> bool:
        if now_s is None:
            now_s = time.perf_counter()
        raw = self._schmitt(v)

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

    # ========= Axis-aware setup =========
    start_axis = (config.get("startAxis") or "y").lower()
    lane_axis  = (config.get("laneAxis")  or "x").lower()
    assert start_axis in ("x", "y"), "startAxis must be 'x' or 'y'"
    assert lane_axis  in ("x", "y"), "laneAxis must be 'x' or 'y'"

    # Filters: line touch (ankles + hands) ON THE START AXIS
    ankle_touch_f = LineTouchFilter(config["startLine"], band_units=4, press_ms=60, release_ms=80)
    left_touch_f  = LineTouchFilter(config["startLine"], band_units=4, press_ms=60, release_ms=80)
    right_touch_f = LineTouchFilter(config["startLine"], band_units=4, press_ms=60, release_ms=80)

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

            # Determine lane along lane_axis
            current_lane = None
            if ankles_center_map is not None:
                lane_val = axis_value(ankles_center_map, lane_axis)
                axis_len = MAP_WIDTH if lane_axis == "x" else MAP_HEIGHT
                current_lane = lane_for_axis(lane_val, axis_len=axis_len, inner_on_left=config.get("innerOnLeft", True))

            # Filtered line touches on the START AXIS
            ankle_val = axis_value(ankles_center_map, start_axis)
            left_val  = axis_value(left_hand_map, start_axis)
            right_val = axis_value(right_hand_map, start_axis)

            ankle_touching = ankle_touch_f.sample(ankle_val, now)
            left_touching  = left_touch_f.sample(left_val, now)
            right_touching = right_touch_f.sample(right_val, now)

            # --- STATE MACHINE ---
            if state == "WAITING_FOR_SKATER":
                # Expect skaters in pre-start zone (behind blue band) — bands are along START AXIS
                # So the "preStartMin/Max" are compared along start_axis.
                in_band_val = axis_value(ankles_center_map, start_axis)
                if (in_band_val is not None) and (config["preStartMin"] < in_band_val < config["preStartMax"]):
                    state = "IN_PRE_START"
                    state_timer = now
                    touched_line_after_ready = False
                    offender_lane = None
                    last_false_reason = ""
                    played_lane_cue = False
                    warned_touch_after_ready = False
                    print("Skaters behind pre-start band. Waiting...")

            elif state == "IN_PRE_START":
                in_band_val = axis_value(ankles_center_map, start_axis)
                in_band = (in_band_val is not None) and (config["preStartMin"] < in_band_val < config["preStartMax"])
                if not in_band:
                    state = "WAITING_FOR_SKATER"
                else:
                    # Announce lane once so athletes/spectators know who's where
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
                # Close to start line along start_axis
                start_line_val = config["startLine"]
                val = axis_value(ankles_center_map, start_axis)
                if (val is not None) and ((start_line_val - 50) < val < (start_line_val + 2)):
                    print("Skaters at start line. Wait for settle/breath...")
                    state = "SETTLE_BEFORE_READY"
                    state_timer = now

            elif state == "SETTLE_BEFORE_READY":
                if (now - state_timer) >= config["settleBreathSeconds"]:
                    print('Starter: *raises gun*')
                    print('Starter (clearly): "Ready."')
                    audio_gate.play(READY_SOUND)
                    state = "AFTER_READY_AUDIO"   # start the timer only after audio completes
                else:
                    # If they leave the vicinity of the start line, back to approach
                    val = axis_value(ankles_center_map, start_axis)
                    if (val is None) or (val < (config["startLine"] - 60)):
                        state = "APPROACHING_START"

            elif state == "AFTER_READY_AUDIO":
                if audio_gate.is_done():
                    state = "READY_WAIT_POSITION"
                    state_timer = now   # start the ready-assume timeout now
                    touched_line_after_ready = False
                    warned_touch_after_ready = False

            elif state == "READY_WAIT_POSITION":
                elapsed = now - state_timer

                # Illegal touch after READY?
                if (ankle_touching or left_touching or right_touching):
                    if not warned_touch_after_ready:
                        audio_gate.play(BUZZER_SOUND)  # quick warning
                        warned_touch_after_ready = True
                    touched_line_after_ready = True
                    offender_lane = offender_lane or current_lane

                if touched_line_after_ready:
                    last_false_reason = "Crossing the line"
                    state = "FALSE_START"
                else:
                    # If they keep moving too much until timeout → false start
                    if elapsed > config["readyAssumeTimeout"]:
                        if moving:
                            last_false_reason = "Going down too slow"
                            offender_lane = offender_lane or current_lane
                            state = "FALSE_START"
                    # If stable → enter HOLD
                    elif not moving:
                        state = "HOLD_BEFORE_GUN"
                        state_timer = now
                        print("Both skaters appear set. Holding...")

            elif state == "HOLD_BEFORE_GUN":
                hold_elapsed = now - state_timer

                if moving:
                    last_false_reason = "Not stable"
                    offender_lane = offender_lane or current_lane
                    state = "FALSE_START"

                elif ankle_touching or left_touching or right_touching:
                    last_false_reason = "Crossing the line"
                    offender_lane = offender_lane or current_lane
                    state = "FALSE_START"

                elif hold_elapsed >= config["holdPauseSeconds"]:
                    print("** GUN FIRED **")
                    audio_gate.play(GUN_SHOT_SOUND)
                    state = "AFTER_GUN_AUDIO"

            elif state == "AFTER_GUN_AUDIO":
                if audio_gate.is_done():
                    state = "RACE_ACTIVE"

            elif state == "RACE_ACTIVE":
                pass

            elif state == "FALSE_START":
                print("** FALSE START **")
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

                    # Lane call
                    lane_sound = None
                    if offender_lane == "inner":
                        lane_sound = INNER_LANE_SOUND
                    elif offender_lane == "outer":
                        lane_sound = OUTER_LANE_SOUND

                    if lane_sound and audio_gate.play(lane_sound):
                        state = "AFTER_LANE_AUDIO"
                    else:
                        rs = reason_to_sound(reason)
                        if rs and audio_gate.play(rs):
                            state = "AFTER_REASON_AUDIO"
                        else:
                            if false_start_count_by_pair >= 2:
                                print(f'Starter: "False start, {lane_txt} lane, disqualified"')
                                print("→ Send the other skater again.")
                                audio_gate.play(DISQUALIFIED_SOUND)
                                state = "AFTER_DQ_AUDIO"
                            else:
                                # non-DQ reset
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
                    rs = reason_to_sound(last_false_reason)
                    if rs and audio_gate.play(rs):
                        state = "AFTER_REASON_AUDIO"
                    else:
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
            map_view = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)

            # Draw pre-start & start lines according to start_axis
            if start_axis == "y":
                # horizontal lines
                cv2.line(map_view, (0, config["preStartMin"]), (MAP_WIDTH, config["preStartMin"]), (255, 0, 0), 2)
                cv2.line(map_view, (0, config["preStartMax"]), (MAP_WIDTH, config["preStartMax"]), (255, 0, 0), 2)
                cv2.line(map_view, (0, config["startLine"]),   (MAP_WIDTH, config["startLine"]),   (0, 255, 0), 2)
            else:
                # vertical lines
                cv2.line(map_view, (config["preStartMin"], 0), (config["preStartMin"], MAP_HEIGHT), (255, 0, 0), 2)
                cv2.line(map_view, (config["preStartMax"], 0), (config["preStartMax"], MAP_HEIGHT), (255, 0, 0), 2)
                cv2.line(map_view, (config["startLine"],   0), (config["startLine"],   MAP_HEIGHT), (0, 255, 0), 2)

            if ankles_center_map is not None:
                p = tuple(np.array(ankles_center_map, dtype=int))
                cv2.circle(map_view, p, 10, (0, 255, 255), -1)

            # Optional debug overlays
            cv2.putText(frame, f'State: {state}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Moving: {moving}  Mag(norm): {filtered_mag:.4f}', (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f'Touch(startAxis {start_axis.upper()}) A/L/R: {int(ankle_touching)}/{int(left_touching)}/{int(right_touching)}',
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
