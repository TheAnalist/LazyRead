"""
Eye-Tracking Scroll Controller
--------------------------------
Scrolls the screen up/down based on vertical eye movement, detected via
MediaPipe FaceMesh landmarks. See README.md for setup and usage.
"""

import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ---- Config ---------------------------------------------------------------
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 153, 145, 246]
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 380, 374, 466]

CALIBRATION_TIME = 5          # seconds
SCROLL_UP_THRESHOLD = 5       # pixels
SCROLL_DOWN_THRESHOLD = 5     # pixels
SCROLL_AMOUNT = 10
COOLDOWN = 0.3                # seconds between scroll actions (prevents spamming)

pyautogui.FAILSAFE = True     # move mouse to a screen corner to abort pyautogui


def get_eye_points(landmarks, indices, w, h):
    return np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    )


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check the camera index/permissions.")

    calibration_done = False
    initial_left_centroid = None
    initial_right_centroid = None
    start_time = time.time()
    last_scroll_time = 0.0

    cv2.namedWindow("Eye Tracking and Scrolling", cv2.WINDOW_NORMAL)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror view, feels more natural
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_eye = get_eye_points(landmarks, LEFT_EYE_INDICES, w, h)
                right_eye = get_eye_points(landmarks, RIGHT_EYE_INDICES, w, h)

                for point in left_eye:
                    cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
                for point in right_eye:
                    cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

                if not calibration_done:
                    initial_left_centroid = left_eye.mean(axis=0).astype(int)
                    initial_right_centroid = right_eye.mean(axis=0).astype(int)

                    elapsed = time.time() - start_time
                    remaining = max(0, int(CALIBRATION_TIME - elapsed))
                    cv2.putText(frame, "Look straight ahead - calibrating...",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Time remaining: {remaining}s",
                                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if elapsed >= CALIBRATION_TIME:
                        calibration_done = True

                else:
                    current_left_centroid = left_eye.mean(axis=0).astype(int)
                    current_right_centroid = right_eye.mean(axis=0).astype(int)

                    left_dy = current_left_centroid[1] - initial_left_centroid[1]
                    right_dy = current_right_centroid[1] - initial_right_centroid[1]

                    now = time.time()
                    if now - last_scroll_time > COOLDOWN:
                        if left_dy > SCROLL_DOWN_THRESHOLD or right_dy > SCROLL_DOWN_THRESHOLD:
                            pyautogui.scroll(-SCROLL_AMOUNT)
                            last_scroll_time = now
                            cv2.putText(frame, "Scrolling down", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        elif left_dy < -SCROLL_UP_THRESHOLD or right_dy < -SCROLL_UP_THRESHOLD:
                            pyautogui.scroll(SCROLL_AMOUNT)
                            last_scroll_time = now
                            cv2.putText(frame, "Scrolling up", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, "Press 'q' to quit, 'r' to recalibrate", (50, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Eye Tracking and Scrolling", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                calibration_done = False
                start_time = time.time()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    main()
