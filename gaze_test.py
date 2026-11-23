"""
LazyRead gaze test
------------------
Live eyes-on-screen metrics from the MediaPipe face blendshapes, plus a
guided recording that suggests thresholds for the gaze gate.

Run:
    python gaze_test.py

Keys:
    1 = record "looking at the screen" (3 s)
    2 = record "looking away" (3 s, left/right/up/down)
    3 = record "eyes closed" (2 s)
    q = quit

The values shown are the smoothed ones used by LazyRead; the recorded
statistics use the raw values of each frame.
"""

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from LazyRead import (
    GAZE_BLINK_MAX,
    GAZE_DOWN_MAX,
    GAZE_LATERAL_MAX,
    GAZE_UP_MAX,
    GazeGate,
    IRIS_INDICES,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    ensure_model,
    gaze_metrics,
    window_is_open,
)

CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
WINDOW_NAME = "LazyRead gaze test"
RECORD_SECONDS = 3.0
CLOSED_SECONDS = 2.0


def stats(samples):
    lat = np.array([s[0] for s in samples])
    up = np.array([s[1] for s in samples])
    down = np.array([s[2] for s in samples])
    blink = np.array([s[3] for s in samples])
    return lat, up, down, blink


def print_recording(name, samples):
    if not samples:
        return
    lat, up, down, blink = stats(samples)
    print(
        f"[{name}] n={len(samples)} | "
        f"lateral min/max/mean={lat.min():.2f}/{lat.max():.2f}/{lat.mean():.2f} | "
        f"up min/max/mean={up.min():.2f}/{up.max():.2f}/{up.mean():.2f} | "
        f"down min/max/mean={down.min():.2f}/{down.max():.2f}/{down.mean():.2f} | "
        f"blink min/max/mean={blink.min():.2f}/{blink.max():.2f}/{blink.mean():.2f}"
    )


def suggest_thresholds(records):
    on, away, closed = records["on"], records["away"], records["closed"]
    if not on or not away:
        return
    lat_on, _, down_on, blink_on = stats(on)
    lat_away, _, down_away, _ = stats(away)

    print("\n--- suggested thresholds ---")
    if lat_away.min() > lat_on.max():
        print(f"GAZE_LATERAL_MAX = {(lat_on.max() + lat_away.min()) / 2:.2f}")
    else:
        print("lateral: on-screen and away-overlap; record again or tune manually")

    if down_away.min() > down_on.max():
        print(f"GAZE_DOWN_MAX = {(down_on.max() + down_away.min()) / 2:.2f}")
    else:
        print("down: on-screen and away-overlap; record again or tune manually")

    print("GAZE_UP_MAX: looking up is the scroll-up direction; keep it high (default 0.80) or tune from the app debug.")
    print("note: keep GAZE_LATERAL_MAX above the lateral value seen during scroll-up (head up, e.g. 0.65).")

    if closed:
        _, _, _, blink_closed = stats(closed)
        if blink_closed.min() > blink_on.max():
            print(f"GAZE_BLINK_MAX = {(blink_on.max() + blink_closed.min()) / 2:.2f}")
        else:
            print("blink: closed-eyes and open-eyes values overlap")
    print("----------------------------\n")


def main():
    model_path = ensure_model()
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_face_blendshapes=True,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        landmarker.close()
        raise RuntimeError("Could not open the webcam.")

    gate = GazeGate()
    last_timestamp_ms = 0
    records = {"on": [], "away": [], "closed": []}
    recording = None
    record_start = 0.0
    window_ready = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            results = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
            )
            now = time.monotonic()

            if results.face_landmarks and results.face_blendshapes:
                landmarks = results.face_landmarks[0]
                lateral, up, down, blink = gaze_metrics(results.face_blendshapes[0])
                on_screen = gate.update(results.face_blendshapes[0])
                color = (0, 255, 0) if on_screen else (0, 165, 255)

                for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                    point = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                    cv2.circle(frame, point, 1, (0, 255, 0), -1)
                if len(landmarks) > IRIS_INDICES[-1]:
                    for idx in IRIS_INDICES:
                        point = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                        cv2.circle(frame, point, 2, (0, 255, 255), -1)

                cv2.putText(
                    frame,
                    f"lateral {gate.lateral:.2f}  (max {GAZE_LATERAL_MAX:.2f})",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    f"up {gate.up:.2f}  (max {GAZE_UP_MAX:.2f})",
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    f"down {gate.down:.2f}  (max {GAZE_DOWN_MAX:.2f})",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    f"blink {gate.blink:.2f}  (max {GAZE_BLINK_MAX:.2f})",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    "EYES: ON SCREEN" if on_screen else "EYES: AWAY",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

                if recording is not None:
                    records[recording].append((lateral, up, down, blink))
                    total = CLOSED_SECONDS if recording == "closed" else RECORD_SECONDS
                    remaining = max(0.0, total - (now - record_start))
                    cv2.putText(
                        frame,
                        f"recording '{recording}': {remaining:.1f}s",
                        (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    if now - record_start >= total:
                        finished = recording
                        print_recording(recording, records[recording])
                        recording = None
                        suggest_thresholds(records)
                        if finished == "on":
                            print("Next: press 2 and look away (left/right/up/down).")
                        elif finished == "away":
                            print("Next: press 3 and close your eyes.")
            else:
                gate.reset()
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                "1: look at screen  2: look away  3: close eyes  q: quit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            if window_ready and not window_is_open(WINDOW_NAME):
                break
            cv2.imshow(WINDOW_NAME, frame)
            window_ready = True

            key = cv2.waitKey(1) & 0xFF
            if not window_is_open(WINDOW_NAME):
                break
            if key == ord("q"):
                break
            if key in (ord("1"), ord("2"), ord("3")):
                recording = {ord("1"): "on", ord("2"): "away", ord("3"): "closed"}[key]
                records[recording] = []
                record_start = time.monotonic()
                print(f"Recording '{recording}' for a few seconds...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
