import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Landmarks to track 
FULL_BODY_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,      mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,     mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.NOSE
]

def _body_size_px(landmarks_xy, frame_w, frame_h):
    """
    Estimate body size in from shoulder/hip spans.
    landmarks_xy: Nx2 normalized coords (0..1)
    """
    def span(a, b):
        ax, ay = landmarks_xy[a]; bx, by = landmarks_xy[b]
        # normalized -> pixels
        ax *= frame_w; ay *= frame_h
        bx *= frame_w; by *= frame_h
        return np.hypot(ax - bx, ay - by)

    L = mp_pose.PoseLandmark
    spans = []
    try:
        spans.append(span(L.LEFT_SHOULDER.value, L.RIGHT_SHOULDER.value))
        spans.append(span(L.LEFT_HIP.value,      L.RIGHT_HIP.value))
    except Exception:
        pass

    # Fallback to a safe value if we can't compute spans
    if not spans:
        return max(frame_w, frame_h) * 0.25
    return float(np.mean(spans))

def is_movement(frame, pose_processor, landmark_history: list, movement_history: list,
                movement_threshold: float, tremor_floor: float):
    """
    Detects movement using pose landmarks across frames.
    Returns (significant_move: bool, smoothed_mag_norm: float)
    Approach:
      - Uses pixel distances but NORMALIZES by body size in pixels -> distance-invariant
      - Weights landmark movement by visibility to suppress far/occluded noise
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_processor.process(rgb_frame)

    if not result.pose_landmarks:
        return False, 0.0

    # skeleton for debugging
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Gather normalized landmark coords and visibility
    all_lms = result.pose_landmarks.landmark
    # Nx4: (x, y, z, visibility)
    current = np.array([(lm.x, lm.y, lm.z, lm.visibility) for lm in all_lms], dtype=np.float32)

    # Build arrays for the selected landmarks
    idxs = [lm.value for lm in FULL_BODY_LANDMARKS]
    curr_xy = current[idxs, :2]            # normalized (0..1)
    curr_vis = current[idxs, 3]            # visibility 0..1

    # Push current frame to history (head = most recent)
    landmark_history.insert(0, current)
    if len(landmark_history) > 5:
        landmark_history.pop()

    if len(landmark_history) < 2:
        return False, 0.0

    prev = landmark_history[1]
    prev_xy = prev[idxs, :2]
    prev_vis = prev[idxs, 3]

    # Frame size (for px conversion)
    frame_h, frame_w = frame.shape[:2]

    # Estimate body size in pixels (average shoulder/hip span)
    body_px = _body_size_px(current[:, :2], frame_w, frame_h)
    if body_px < 1.0:  # guard
        body_px = 1.0

    # Compute per-landmark pixel displacement
    # normalized -> pixels
    dxy_px = np.empty_like(curr_xy)
    dxy_px[:, 0] = (curr_xy[:, 0] - prev_xy[:, 0]) * frame_w
    dxy_px[:, 1] = (curr_xy[:, 1] - prev_xy[:, 1]) * frame_h
    dist_px = np.linalg.norm(dxy_px, axis=1)  # per-landmark movement in pixels

    # Visibility weighting (use min of current/previous vis to be conservative)
    vis_w = np.minimum(curr_vis, prev_vis)
    # Clip and non-linear mapping (so very low vis contributes almost nothing)
    vis_w = np.clip(vis_w, 0.0, 1.0) ** 1.5

    # Robust mean movement normalized by body size
    # (divide pixel motion by body size to get a dimensionless ratio)
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_norm = dist_px / body_px

    # Weighted mean (ignore NaNs)
    valid = np.isfinite(dist_norm)
    if not np.any(valid):
        return False, 0.0
    w = vis_w[valid]
    if np.sum(w) < 1e-6:
        # if everything is near-invisible, treat as no reliable movement
        return False, 0.0
    mag_norm = float(np.sum(dist_norm[valid] * w) / np.sum(w))

    # Smooth with EMA over last ~10 frames
    movement_history.append(mag_norm)
    if len(movement_history) > 10:
        movement_history.pop(0)

    alpha = 0.3
    ema = movement_history[0]
    for v in movement_history[1:]:
        ema = alpha * v + (1 - alpha) * ema

    # Decide significance using normalized units
    significant = ema > movement_threshold
    # HUD if really above tremor
    if significant and ema > (tremor_floor * 1.25):
        cv2.putText(frame, 'MOVEMENT!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    return significant, float(ema)
