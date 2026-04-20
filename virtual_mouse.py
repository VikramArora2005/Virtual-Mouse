"""
Virtual Mouse Using Hand Gesture Recognition
============================================
Uses the NEW MediaPipe Tasks API (HandLandmarker) for hand landmark detection
and PyAutoGUI to control the mouse.

Requires:
    pip install mediapipe opencv-python pyautogui numpy requests

The script auto-downloads the hand_landmarker.task model on first run.

Controls:
  Index finger up              → Move cursor
  Index + Middle up            → Click-ready mode
  Index + Middle pinch         → Left click
  Thumb + Index pinch          → Right click
  All 5 fingers up             → Scroll (move hand up/down)
  Fist                         → Drag
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
import urllib.request

# ─── New Tasks API imports ────────────────────────────────────────────────────
BaseOptions            = mp.tasks.BaseOptions
HandLandmarker         = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions  = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult   = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode      = mp.tasks.vision.RunningMode

# ─── Landmark indices (MediaPipe Hand) ───────────────────────────────────────
WRIST           = 0
THUMB_TIP       = 4
THUMB_IP        = 3
INDEX_TIP       = 8
INDEX_PIP       = 6
MIDDLE_TIP      = 12
MIDDLE_PIP      = 10
RING_TIP        = 16
RING_PIP        = 14
PINKY_TIP       = 20
PINKY_PIP       = 18

# ─── Configuration ────────────────────────────────────────────────────────────
CAMERA_INDEX       = 0
FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480
SMOOTHING          = 6          # Moving-average window (higher = smoother, more lag)
CLICK_THRESHOLD    = 38         # Pixel distance between fingertips to trigger click
SCROLL_SENSITIVITY = 18         # Hand-pixel movement per scroll tick
COOLDOWN_FRAMES    = 18         # Frames to wait between clicks
MODEL_PATH         = "hand_landmarker.task"
MODEL_URL          = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
# ──────────────────────────────────────────────────────────────────────────────

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

SCREEN_W, SCREEN_H = pyautogui.size()


# ─── Auto-download model if missing ──────────────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading hand_landmarker.task model …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Model saved to: {MODEL_PATH}")
    else:
        print(f"[INFO] Model found: {MODEL_PATH}")


# ─── Smoothing buffer ─────────────────────────────────────────────────────────
class Smoother:
    def __init__(self, size=SMOOTHING):
        self.buf  = []
        self.size = size

    def smooth(self, x, y):
        self.buf.append((x, y))
        if len(self.buf) > self.size:
            self.buf.pop(0)
        return (
            int(np.mean([p[0] for p in self.buf])),
            int(np.mean([p[1] for p in self.buf])),
        )

    def reset(self):
        self.buf.clear()


# ─── Landmark utilities ───────────────────────────────────────────────────────
def lm_px(landmark, idx):
    """Convert normalised landmark to pixel coords."""
    p = landmark[idx]
    return int(p.x * FRAME_WIDTH), int(p.y * FRAME_HEIGHT)

def dist_px(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ─── Finger-state detection ───────────────────────────────────────────────────
def fingers_up(landmark):
    """
    Returns [thumb, index, middle, ring, pinky]  (1 = up, 0 = down)
    Uses the NEW Tasks API NormalizedLandmark objects.
    """
    # Thumb: compare x (horizontal movement)
    thumb  = 1 if landmark[THUMB_TIP].x < landmark[THUMB_IP].x else 0

    # Four fingers: tip Y < pip Y  →  finger is raised
    index  = 1 if landmark[INDEX_TIP ].y < landmark[INDEX_PIP ].y else 0
    middle = 1 if landmark[MIDDLE_TIP].y < landmark[MIDDLE_PIP].y else 0
    ring   = 1 if landmark[RING_TIP  ].y < landmark[RING_PIP  ].y else 0
    pinky  = 1 if landmark[PINKY_TIP ].y < landmark[PINKY_PIP ].y else 0

    return thumb, index, middle, ring, pinky


# ─── Gesture classifier ───────────────────────────────────────────────────────
def classify_gesture(landmark):
    """Return a gesture string and the pixel position of the index fingertip."""
    thumb, index, middle, ring, pinky = fingers_up(landmark)

    index_tip  = lm_px(landmark, INDEX_TIP)
    thumb_tip  = lm_px(landmark, THUMB_TIP)
    middle_tip = lm_px(landmark, MIDDLE_TIP)

    pinch_idx_thumb   = dist_px(index_tip,  thumb_tip)
    pinch_idx_middle  = dist_px(index_tip,  middle_tip)

    # ── Rules (order matters — most specific first) ──────────────────────────
    # Scroll: all five fingers up
    if thumb and index and middle and ring and pinky:
        return "SCROLL", index_tip

    # Drag: closed fist
    if not index and not middle and not ring and not pinky:
        return "DRAG", index_tip

    # Right click: thumb + index only, pinched
    if thumb and index and not middle and not ring and not pinky:
        if pinch_idx_thumb < CLICK_THRESHOLD:
            return "RIGHT_CLICK", index_tip
        return "MOVE", index_tip          # thumb + index not pinched → still move

    # Left click: index + middle up, pinched together
    if index and middle and not ring and not pinky:
        if pinch_idx_middle < CLICK_THRESHOLD:
            return "LEFT_CLICK", index_tip
        return "MOVE_READY", index_tip   # two fingers up but not pinched

    # Move: only index finger up
    if index and not middle and not ring and not pinky:
        return "MOVE", index_tip

    return "IDLE", index_tip


# ─── Draw hand skeleton manually ─────────────────────────────────────────────
CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

def draw_skeleton(frame, landmark):
    """Draw landmarks and connections using the Tasks API landmark list."""
    # Connections
    for conn in CONNECTIONS:
        p1 = lm_px(landmark, conn.start)
        p2 = lm_px(landmark, conn.end)
        cv2.line(frame, p1, p2, (0, 200, 255), 2, cv2.LINE_AA)
    # Landmarks
    for idx in range(21):
        px = lm_px(landmark, idx)
        cv2.circle(frame, px, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, px, 5, (0, 150, 255),  1,  cv2.LINE_AA)
    # Highlight fingertips
    for tip_idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        px = lm_px(landmark, tip_idx)
        cv2.circle(frame, px, 8, (0, 255, 120), -1, cv2.LINE_AA)


# ─── HUD overlay ─────────────────────────────────────────────────────────────
LEGEND = [
    ("Index up",          "Move cursor"),
    ("2 fingers up",      "Click-ready"),
    ("2 fingers pinch",   "Left click"),
    ("Thumb+Idx pinch",   "Right click"),
    ("All fingers up",    "Scroll"),
    ("Fist",              "Drag"),
]

GESTURE_COLOR = {
    "MOVE":        (0, 255, 120),
    "MOVE_READY":  (0, 220, 255),
    "LEFT_CLICK":  (0, 100, 255),
    "RIGHT_CLICK": (200, 0, 255),
    "SCROLL":      (255, 180, 0),
    "DRAG":        (0, 80, 255),
    "IDLE":        (180, 180, 180),
}

def draw_hud(frame, gesture):
    color = GESTURE_COLOR.get(gesture, (200, 200, 200))
    cv2.rectangle(frame, (0, 0), (280, 38), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesture: {gesture}",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

    for i, (key, val) in enumerate(LEGEND):
        y = FRAME_HEIGHT - 10 - i * 20
        cv2.putText(frame, f"{key}: {val}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ensure_model()

    smoother       = Smoother()
    prev_scroll_y  = None
    click_cooldown = 0
    dragging       = False
    gesture_label  = "IDLE"
    latest_result  = None          # shared between callback and main loop

    # ── LIVE_STREAM callback ────────────────────────────────────────────────
    def result_callback(result: HandLandmarkerResult, output_image, timestamp_ms: int):
        nonlocal latest_result
        latest_result = result

    # ── HandLandmarker setup ────────────────────────────────────────────────
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        result_callback=result_callback,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Virtual Mouse started  |  Press 'q' to quit")

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_ts = 0  # monotonically increasing timestamp in ms

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera read failed.")
                break

            frame     = cv2.flip(frame, 1)          # mirror
            frame_ts += 33                           # ~30 fps tick

            # Convert to MediaPipe Image and send asynchronously
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            landmarker.detect_async(mp_image, frame_ts)

            # ── Process latest result ──────────────────────────────────────
            if latest_result and latest_result.hand_landmarks:
                landmark = latest_result.hand_landmarks[0]   # first hand

                draw_skeleton(frame, landmark)

                gesture, index_tip = classify_gesture(landmark)
                gesture_label = gesture

                # Map webcam coords → screen coords
                sx = int(np.interp(index_tip[0], [40, FRAME_WIDTH  - 40], [0, SCREEN_W]))
                sy = int(np.interp(index_tip[1], [40, FRAME_HEIGHT - 40], [0, SCREEN_H]))
                sx, sy = smoother.smooth(sx, sy)

                # ── Execute mouse action ──────────────────────────────────
                if gesture == "MOVE":
                    if dragging:
                        pyautogui.mouseUp(); dragging = False
                    pyautogui.moveTo(sx, sy)

                elif gesture == "MOVE_READY":
                    if dragging:
                        pyautogui.mouseUp(); dragging = False
                    pyautogui.moveTo(sx, sy)

                elif gesture == "LEFT_CLICK":
                    pyautogui.moveTo(sx, sy)
                    if click_cooldown == 0:
                        pyautogui.click()
                        click_cooldown = COOLDOWN_FRAMES

                elif gesture == "RIGHT_CLICK":
                    pyautogui.moveTo(sx, sy)
                    if click_cooldown == 0:
                        pyautogui.rightClick()
                        click_cooldown = COOLDOWN_FRAMES

                elif gesture == "SCROLL":
                    iy = index_tip[1]
                    if prev_scroll_y is not None:
                        delta = (prev_scroll_y - iy) / SCROLL_SENSITIVITY
                        if abs(delta) >= 0.5:
                            pyautogui.scroll(int(delta))
                    prev_scroll_y = iy

                elif gesture == "DRAG":
                    if not dragging:
                        pyautogui.mouseDown(); dragging = True
                    pyautogui.moveTo(sx, sy)

                else:   # IDLE
                    if dragging:
                        pyautogui.mouseUp(); dragging = False

                if gesture != "SCROLL":
                    prev_scroll_y = None

            else:
                # No hand detected
                gesture_label = "No hand"
                smoother.reset()
                if dragging:
                    pyautogui.mouseUp(); dragging = False
                prev_scroll_y = None

            if click_cooldown > 0:
                click_cooldown -= 1

            draw_hud(frame, gesture_label)
            cv2.imshow("Virtual Mouse  |  Hand Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    if dragging:
        pyautogui.mouseUp()
    print("Virtual Mouse stopped.")


if __name__ == "__main__":
    main()
