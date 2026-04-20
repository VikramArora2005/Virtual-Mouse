"""
Virtual Mouse Using Hand Gesture Recognition
============================================
Uses MediaPipe for hand landmark detection and PyAutoGUI to control the mouse.

Controls:
  - Index finger up        → Move cursor
  - Index + Middle up      → Left click (pinch fingers together)
  - Thumb + Index pinch    → Right click
  - All fingers up         → Scroll (move hand up/down)
  - Fist (all down)        → Drag (hold)
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ─── Configuration ────────────────────────────────────────────────────────────
CAMERA_INDEX       = 0          # Webcam index (0 = default)
FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480
SMOOTHING          = 5          # Higher = smoother but more lag (1-10)
CLICK_THRESHOLD    = 35         # Pixel distance to trigger click
SCROLL_SENSITIVITY = 20         # Pixels of hand movement per scroll unit
# ──────────────────────────────────────────────────────────────────────────────

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0          # Remove artificial delay

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

SCREEN_W, SCREEN_H = pyautogui.size()


# ─── Helper: smoothing buffer ─────────────────────────────────────────────────
class Smoother:
    def __init__(self, size=SMOOTHING):
        self.buf  = []
        self.size = size

    def smooth(self, x, y):
        self.buf.append((x, y))
        if len(self.buf) > self.size:
            self.buf.pop(0)
        avg_x = int(np.mean([p[0] for p in self.buf]))
        avg_y = int(np.mean([p[1] for p in self.buf]))
        return avg_x, avg_y


# ─── Helper: landmark accessors ───────────────────────────────────────────────
def lm(hand_landmarks, idx):
    """Return (x, y) pixel coords for a landmark index."""
    lk = hand_landmarks.landmark[idx]
    return int(lk.x * FRAME_WIDTH), int(lk.y * FRAME_HEIGHT)

def dist(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ─── Finger-up detection ──────────────────────────────────────────────────────
def fingers_up(hand_landmarks):
    """
    Returns list [thumb, index, middle, ring, pinky]
    1 = finger is raised, 0 = finger is folded
    """
    tips = [4, 8, 12, 16, 20]
    pip  = [2, 6, 10, 14, 18]   # second knuckle

    status = []
    # Thumb: compare x instead of y (horizontal finger)
    wrist = hand_landmarks.landmark[0]
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    status.append(1 if thumb_tip.x < thumb_mcp.x else 0)

    # Other four fingers
    for tip, pip_idx in zip(tips[1:], pip[1:]):
        status.append(
            1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip_idx].y else 0
        )
    return status


# ─── Gesture → action mapping ─────────────────────────────────────────────────
def detect_gesture(fingers, hand_landmarks):
    """
    Returns a gesture string based on finger states.
    """
    thumb, index, middle, ring, pinky = fingers

    # Pinch: index and thumb close together → left click
    index_tip = lm(hand_landmarks, 8)
    thumb_tip  = lm(hand_landmarks, 4)
    pinch_dist = dist(index_tip, thumb_tip)

    # Index + middle up, rest down → move
    if index and not middle and not ring and not pinky:
        return "MOVE"

    # Index + middle both up → click/double-click mode
    if index and middle and not ring and not pinky:
        if pinch_dist < CLICK_THRESHOLD:
            return "LEFT_CLICK"
        return "MOVE_CLICK_READY"

    # Thumb + index pinch (both up, close) → right click
    if thumb and index and not middle and not ring and not pinky:
        if pinch_dist < CLICK_THRESHOLD:
            return "RIGHT_CLICK"

    # All fingers up → scroll
    if thumb and index and middle and ring and pinky:
        return "SCROLL"

    # Fist → drag
    if not index and not middle and not ring and not pinky:
        return "DRAG"

    return "IDLE"


# ─── Main loop ────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    smoother       = Smoother()
    prev_scroll_y  = None
    click_cooldown = 0
    dragging       = False
    gesture_label  = "IDLE"

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
    ) as hands:

        print("Virtual Mouse started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)   # Mirror
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                fingers = fingers_up(hand_landmarks)
                gesture = detect_gesture(fingers, hand_landmarks)
                gesture_label = gesture

                # Index fingertip → mouse coords (map webcam to screen)
                ix, iy = lm(hand_landmarks, 8)
                sx = int(np.interp(ix, [50, FRAME_WIDTH  - 50], [0, SCREEN_W]))
                sy = int(np.interp(iy, [50, FRAME_HEIGHT - 50], [0, SCREEN_H]))
                sx, sy = smoother.smooth(sx, sy)

                # ── Gesture actions ──────────────────────────────────────────
                if gesture == "MOVE":
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    pyautogui.moveTo(sx, sy)

                elif gesture == "MOVE_CLICK_READY":
                    pyautogui.moveTo(sx, sy)

                elif gesture == "LEFT_CLICK" and click_cooldown == 0:
                    pyautogui.click()
                    click_cooldown = 20   # frames

                elif gesture == "RIGHT_CLICK" and click_cooldown == 0:
                    pyautogui.rightClick()
                    click_cooldown = 20

                elif gesture == "SCROLL":
                    if prev_scroll_y is None:
                        prev_scroll_y = iy
                    delta = (prev_scroll_y - iy) / SCROLL_SENSITIVITY
                    if abs(delta) > 0.3:
                        pyautogui.scroll(int(delta))
                    prev_scroll_y = iy

                elif gesture == "DRAG":
                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                    pyautogui.moveTo(sx, sy)

                else:   # IDLE
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    prev_scroll_y = None

                if gesture != "SCROLL":
                    prev_scroll_y = None

            else:
                gesture_label = "No hand detected"
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # Cool-down counter
            if click_cooldown > 0:
                click_cooldown -= 1

            # ── HUD overlay ──────────────────────────────────────────────────
            cv2.rectangle(frame, (0, 0), (260, 35), (0, 0, 0), -1)
            cv2.putText(frame, f"Gesture: {gesture_label}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)

            # Legend
            legend = [
                ("1 finger up",        "Move cursor"),
                ("2 fingers up",       "Click mode"),
                ("2 fingers pinch",    "Left click"),
                ("Thumb+Index pinch",  "Right click"),
                ("All fingers up",     "Scroll"),
                ("Fist",               "Drag"),
            ]
            for i, (key, val) in enumerate(legend):
                cv2.putText(frame, f"{key}: {val}",
                            (8, FRAME_HEIGHT - 10 - i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Virtual Mouse – Hand Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    if dragging:
        pyautogui.mouseUp()
    print("Virtual Mouse stopped.")


if __name__ == "__main__":
    main()
