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
from itertools import cycle

import os, tempfile 
PUBLISH_DIR = "live"
os.makedirs(PUBLISH_DIR, exist_ok=True)


PUBLISH_DIR = "live"

# one toggler per stream
_toggles = {"left": cycle(("a", "b")), "right": cycle(("a", "b"))}

# =========================
# Sound files
# =========================
GO_TO_THE_START_SOUND = "Sounds/go_to_the_start.mp3"
CROSSING_THE_LINE_SOUND = "Sounds/crossing_the_line.mp3"
DISQUALIFIED_SOUND = "Sounds/disqualified.mp3"
DOWN_TOO_SLOW_SOUND = "Sounds/down_too_slow.mp3"
FALSE_START_SOUND = "Sounds/false_start.mp3"
BUZZER_SOUND = "Sounds/buzzer.mp3"
GUN_SHOT_SOUND = "Sounds/gun_shoot.mp3"
INNER_LANE_SOUND = "Sounds/inner_lane.mp3"
OUTER_LANE_SOUND = "Sounds/outer_lane.mp3"
READY_SOUND = "Sounds/ready.mp3"
WHISTLE_SOUND = "Sounds/whistle.mp3"
SECOND_SHOT_SOUND = "Sounds/falseStartBuzzer.mp3"
NOT_STABLE_SOUND = "Sounds/not_stable.mp3"

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

    # Defaults — thresholds are for movement.py
    cfg.setdefault("preStartMin", 180)
    cfg.setdefault("preStartMax", 220)
    cfg.setdefault("startLine", 400)
    cfg.setdefault("threshold", 0.015)    # movement.py "significant" threshold (strong)
    cfg.setdefault("microTremor", 0.008)  # movement.py tremor floor (baseline)
    cfg.setdefault("settleBreathSeconds", 1.0)
    cfg.setdefault("readyAssumeTimeout", 3.0)
    cfg.setdefault("holdPauseSeconds", 1.10)
    cfg.setdefault("cameraIndex", 0)
    cfg.setdefault("innerOnLeft", False)

    # Orientation
    cfg.setdefault("startAxis", "y")   # 'y' overhead, 'x' side camera
    cfg.setdefault("laneAxis", "x")    # lane split axis

    # Which side of startLine is LEGAL: "lt" => legal if value < startLine; "gt" => legal if value > startLine
    cfg.setdefault("legal_side", "lt")

    # Movement timing guards (around READY/HOLD)
    cfg.setdefault("readyGraceMs", 300)   # ignore movement for first 300ms after READY finishes
    cfg.setdefault("needMovingMs", 200)   # require continuous movement for >= 200ms to count

    # Two-level movement thresholds
    cfg.setdefault("weakFactor", 1.35)    # weak movement = EMA > microTremor * weakFactor
    cfg.setdefault("quietMs", 180)        # must be weak-still for this long to enter HOLD
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
        return NOT_STABLE_SOUND
    return None


def publish_frame(name: str, frame):
    """
    Ping-pong writer: write to left_a.jpg/left_b.jpg (or right_a/right_b)
    then update a tiny flag file so the UI knows which one is complete.
    """
    os.makedirs(PUBLISH_DIR, exist_ok=True)
    slot = next(_toggles[name])               # "a" or "b"
    base = os.path.join(PUBLISH_DIR, f"{name}_{slot}.jpg")
    ok = cv2.imwrite(base, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return
    # atomically tell the UI which one to read
    flag_tmp = os.path.join(PUBLISH_DIR, f".{name}.flag.tmp")
    flag     = os.path.join(PUBLISH_DIR, f"{name}.flag")
    with open(flag_tmp, "w", encoding="utf-8") as f:
        f.write(slot)                          # "a" or "b"
    os.replace(flag_tmp, flag)                 # atomic for Windows & POSIX



# =========================
# AudioGate (non-blocking, interrupt-capable)
# =========================
class AudioGate:
    """
    - play(path, interrupt=False): async play; if interrupt=True, stop current first
    - is_done(): robust finished check (channel idle + time guard)
    - stop(), set_volume()
    """
    def __init__(self):
        self.channel = None
        self.ends_at = 0.0
        self._ensure()

    def _ensure(self):
        global _mixer_ready
        if not _mixer_ready:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            _mixer_ready = True

    def play(self, path: str, interrupt: bool = False) -> bool:
        try:
            self._ensure()
            snd = _sound_cache.get(path)
            if snd is None:
                snd = pygame.mixer.Sound(path)
                _sound_cache[path] = snd

            if self.channel is None:
                self.channel = pygame.mixer.find_channel(True)

            if self.channel.get_busy():
                if not interrupt:
                    return False
                self.channel.stop()

            self.channel.play(snd)
            self.ends_at = time.perf_counter() + snd.get_length()
            return True
        except Exception as e:
            print(f"[Sound error] {e}")
            self.channel = None
            self.ends_at = 0.0
            return False

    def is_done(self) -> bool:
        chan_free = (self.channel is None) or (not self.channel.get_busy())
        time_passed = time.perf_counter() >= self.ends_at
        return chan_free and time_passed

    def stop(self):
        if self.channel is not None:
            self.channel.stop()
        self.ends_at = 0.0

    def set_volume(self, v: float):
        if self.channel is None:
            self.channel = pygame.mixer.find_channel(True)
        if self.channel is not None:
            self.channel.set_volume(max(0.0, min(1.0, float(v))))

# =========================
# Geometry / lane helpers
# =========================
def get_map_point_for_landmarks(frame, landmarks, ids, H):
    pt = get_landmark_center(landmarks, frame, ids)
    return transform_point(pt, H) if pt is not None else None

def axis_value(pt, axis: str):
    if pt is None:
        return None
    return pt[0] if axis == "x" else pt[1]

def lane_for_axis(val, axis_len=1000, inner_on_left=True):
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
    Schmitt-trigger + debounce for entering beyond a 1D line (along one axis).
    .sample() returns True when sustained beyond the line, False when sustained back.
    """
    def __init__(self, line_pos: float, band_units: float = 0.0, press_ms: int = 60, release_ms: int = 80):
        self.line = float(line_pos)
        self.band = float(band_units)
        self.press_s = press_ms / 1000.0
        self.release_s = release_ms / 1000.0
        self.state = False
        self._pending = None  # (target_state, t_start)

    def _schmitt(self, v: float | None) -> bool:
        if v is None:
            return False
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
# Utilities
# =========================
def estimate_body_size(landmarks, frame, fallback=300.0):
    """(Kept for possible future use; not required by movement.py path)"""
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
    pose_processor = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(config.get("cameraIndex", 0))

    # FSM
    state = "WAITING_FOR_SKATER"
    state_timer = None

    # Movement buffers for movement.py
    landmark_history = []
    movement_history = []

    # False start tracking
    false_start_count_by_pair = 0

    touched_line_after_ready = False

    MAP_WIDTH = 1000
    MAP_HEIGHT = 500

    offender_lane = None
    last_false_reason = ""

    played_lane_cue = False
    warned_touch_after_ready = False

    audio_gate = AudioGate()

    start_axis = (config.get("startAxis") or "y").lower()
    lane_axis  = (config.get("laneAxis")  or "x").lower()
    legal_side = (config.get("legal_side") or "lt").lower()
    assert start_axis in ("x", "y")
    assert lane_axis  in ("x", "y")
    assert legal_side in ("lt", "gt")

    band_px = 4  # hysteresis band

    # Filters for touching the start line (on the start axis)
    ankle_touch_f = LineTouchFilter(config["startLine"], band_units=band_px, press_ms=60, release_ms=80)
    left_touch_f  = LineTouchFilter(config["startLine"], band_units=band_px, press_ms=60, release_ms=80)
    right_touch_f = LineTouchFilter(config["startLine"], band_units=band_px, press_ms=60, release_ms=80)

    # READY/HOLD movement timing
    ready_grace_s = config.get("readyGraceMs", 300) / 1000.0
    need_moving_s = config.get("needMovingMs", 200) / 1000.0
    quiet_s       = config.get("quietMs", 180) / 1000.0
    weak_factor   = config.get("weakFactor", 1.35)

    # Movement sustain timers
    moving_on_since_strong = None  # when strong movement became True
    moving_on_since_weak   = None  # when weak movement became True
    still_since_weak       = None  # when weak movement became False

    def is_illegal(v):
        if v is None:
            return False
        if legal_side == "lt":     # legal is < startLine
            return v >= (config["startLine"] + band_px)
        else:                      # legal is > startLine
            return v <= (config["startLine"] - band_px)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()

            # Pose landmarks for the *current* visible skater(s)
            landmarks = get_pose_landmarks(frame, pose_processor)

            # Feet & hands in map coords
            ankles_center_orig = get_landmark_center(landmarks, frame, [PL.LEFT_ANKLE, PL.RIGHT_ANKLE])
            ankles_center_map = transform_point(ankles_center_orig, H) if ankles_center_orig else None

            left_hand_map  = get_map_point_for_landmarks(frame, landmarks, [PL.LEFT_WRIST, PL.LEFT_INDEX], H) if landmarks else None
            right_hand_map = get_map_point_for_landmarks(frame, landmarks, [PL.RIGHT_WRIST, PL.RIGHT_INDEX], H) if landmarks else None

            # Movement detection (from movement.py)
            significant_move, smoothed_mag = is_movement(
                frame, pose_processor, landmark_history, movement_history,
                config['threshold'], config['microTremor']
            )

            # Two-level movement
            move_strong = bool(significant_move)  # EMA >= threshold
            move_weak   = bool(smoothed_mag > (config['microTremor'] * weak_factor))  # creeping above tremor

            # Strong sustain
            if move_strong:
                if moving_on_since_strong is None:
                    moving_on_since_strong = now
            else:
                moving_on_since_strong = None

            # Weak sustain + stillness
            if move_weak:
                if moving_on_since_weak is None:
                    moving_on_since_weak = now
                still_since_weak = None
            else:
                moving_on_since_weak = None
                if still_since_weak is None:
                    still_since_weak = now

            # Lane by lane_axis
            current_lane = None
            if ankles_center_map is not None:
                lane_val = axis_value(ankles_center_map, lane_axis)
                axis_len = MAP_WIDTH if lane_axis == "x" else MAP_HEIGHT
                current_lane = lane_for_axis(lane_val, axis_len=axis_len, inner_on_left=config.get("innerOnLeft", False))

            # Filtered line touches on start axis
            ankle_val = axis_value(ankles_center_map, start_axis)
            left_val  = axis_value(left_hand_map,  start_axis)
            right_val = axis_value(right_hand_map, start_axis)

            ankle_touching = ankle_touch_f.sample(ankle_val, now)
            left_touching  = left_touch_f.sample(left_val,  now)
            right_touching = right_touch_f.sample(right_val, now)

            # Direction-aware illegal touch
            ankle_illegal = ankle_touching and is_illegal(ankle_val)
            left_illegal  = left_touching  and is_illegal(left_val)
            right_illegal = right_touching and is_illegal(right_val)
            any_illegal_touch = ankle_illegal or left_illegal or right_illegal

            # --- STATE MACHINE ---
            if state == "WAITING_FOR_SKATER":
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
                    if not played_lane_cue and current_lane:
                        played_lane_cue = True
                    if (now - state_timer) >= 3.0:
                        print('Starter (to skaters): "Go to the start"')
                        audio_gate.play(GO_TO_THE_START_SOUND)
                        state = "SAY_GO_TO_START"

            elif state == "SAY_GO_TO_START":
                if audio_gate.is_done():
                    state = "APPROACHING_START"
                    state_timer = now

            elif state == "APPROACHING_START":
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
                    state = "AFTER_READY_AUDIO"
                else:
                    val = axis_value(ankles_center_map, start_axis)
                    if (val is None) or (val < (config["startLine"] - 60)):
                        state = "APPROACHING_START"

            elif state == "AFTER_READY_AUDIO":
                if audio_gate.is_done():
                    state = "READY_WAIT_POSITION"
                    state_timer = now
                    touched_line_after_ready = False
                    warned_touch_after_ready = False
                    # Reset movement timers exactly when READY completes
                    moving_on_since_strong = None
                    moving_on_since_weak   = None
                    still_since_weak       = None

            elif state == "READY_WAIT_POSITION":
                elapsed = now - state_timer

                # Illegal touch after READY?
                if any_illegal_touch:
                    if not warned_touch_after_ready:
                        # audio_gate.play(BUZZER_SOUND)  # warning beep
                        warned_touch_after_ready = True
                    touched_line_after_ready = True
                    offender_lane = offender_lane or current_lane

                if touched_line_after_ready:
                    last_false_reason = "Crossing the line"
                    state = "FALSE_START"
                else:
                    # After the READY grace:
                    if elapsed >= ready_grace_s:
                        # 1) Strong, sustained movement → "Not stable" if within timeout; else "Going down too slow"
                        if moving_on_since_strong and (now - moving_on_since_strong) >= need_moving_s:
                            if elapsed <= config["readyAssumeTimeout"]:
                                last_false_reason = "Not stable"
                                offender_lane = offender_lane or current_lane
                                state = "FALSE_START"
                            else:
                                last_false_reason = "Going down too slow"
                                offender_lane = offender_lane or current_lane
                                state = "FALSE_START"
                        # 2) After timeout, even weak sustained movement counts → "Going down too slow"
                        elif elapsed > config["readyAssumeTimeout"]:
                            if moving_on_since_weak and (now - moving_on_since_weak) >= need_moving_s:
                                last_false_reason = "Going down too slow"
                                offender_lane = offender_lane or current_lane
                                state = "FALSE_START"

                    # Enter HOLD only if weak-still long enough and still within timeout
                    if state == "READY_WAIT_POSITION" and elapsed < config["readyAssumeTimeout"]:
                        if (still_since_weak is not None) and ((now - still_since_weak) >= quiet_s):
                            state = "HOLD_BEFORE_GUN"
                            state_timer = now
                            print("Both skaters appear set. Holding...")

            elif state == "HOLD_BEFORE_GUN":
                hold_elapsed = now - state_timer

                # Sustained strong or weak movement during hold → Not stable
                if (moving_on_since_strong and (now - moving_on_since_strong) >= need_moving_s) or \
                   (moving_on_since_weak   and (now - moving_on_since_weak)   >= need_moving_s):
                    last_false_reason = "Not stable"
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
                # Nothing after the race starts
                # I'll add it later if needed
                pass

            elif state == "FALSE_START":
                print("** FALSE START **")
                # whistle then second shot (or fallback) in sequence
                if audio_gate.play(WHISTLE_SOUND, interrupt=True):
                    state = "AFTER_WHISTLE_AUDIO"
                else:
                    state = "AFTER_WHISTLE_AUDIO"

            elif state == "AFTER_WHISTLE_AUDIO":
                if audio_gate.is_done():
                    if not audio_gate.play(SECOND_SHOT_SOUND, interrupt=True):
                        audio_gate.play(FALSE_START_SOUND, interrupt=True)
                    state = "AFTER_FALSE_AUDIO"

            elif state == "AFTER_FALSE_AUDIO":
                if audio_gate.is_done():
                    false_start_count_by_pair += 1
                    lane_txt = offender_lane or (current_lane or "inner/outer")
                    reason = last_false_reason or "Started before gun"

                    print(f'Starter: "False start, {lane_txt} lane"')
                    print(f'Starter: "{reason}"')

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

            if start_axis == "y":
                cv2.line(map_view, (0, config["preStartMin"]), (MAP_WIDTH, config["preStartMin"]), (255, 0, 0), 2)
                cv2.line(map_view, (0, config["preStartMax"]), (MAP_WIDTH, config["preStartMax"]), (255, 0, 0), 2)
                cv2.line(map_view, (0, config["startLine"]),   (MAP_WIDTH, config["startLine"]),   (0, 255, 0), 2)
            else:
                cv2.line(map_view, (config["preStartMin"], 0), (config["preStartMin"], MAP_HEIGHT), (255, 0, 0), 2)
                cv2.line(map_view, (config["preStartMax"], 0), (config["preStartMax"], MAP_HEIGHT), (255, 0, 0), 2)
                cv2.line(map_view, (config["startLine"],   0), (config["startLine"],   MAP_HEIGHT), (0, 255, 0), 2)

            if ankles_center_map is not None:
                p = tuple(np.array(ankles_center_map, dtype=int))
                cv2.circle(map_view, p, 10, (0, 255, 255), -1)

            # Debug overlays (movement.py signal)
            cv2.putText(frame, f'State: {state}', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f'EMA(movement.py): {float(smoothed_mag):.4f}', (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f'strong:{int(move_strong)} weak:{int(move_weak)}', (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f'Illegal touch A/L/R: {int(ankle_illegal)}/{int(left_illegal)}/{int(right_illegal)}',
                        (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            publish_frame("left", frame)       # will appear in the left panel in the UI
            publish_frame("right", map_view)   # will appear in the right panel in the UI

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
