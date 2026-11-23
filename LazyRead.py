"""
LazyRead - Face-Position Scroll Controller
------------------------------------------
Scrolls the screen up/down based on vertical movement of the face, detected
via MediaPipe FaceLandmarker (Tasks API). See README.md for setup and usage.

LazyRead is a fork of the original AutoScroller "eye tracking" demo. That
demo averaged a few eye-contour landmarks, but those points move with the
head, so the real control signal has always been the face position. This
version uses the face-box centre directly and does not read the eye landmarks.
"""

import ctypes
import os
import sys
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import random
from math import sin
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---- Config ---------------------------------------------------------------
# Face-oval landmarks: enough to measure the face area with 36 points instead
# of iterating over all 478 landmarks every frame.
FACE_OVAL_ANCHORS = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]

CALIBRATION_TIME = 2.0  # seconds for the initial baseline
RECALIBRATION_TIME = 1.0  # seconds when re-locking after a face move/loss
CALIBRATION_MIN_SAMPLES = 10  # never lock a baseline with less data than this
CALIBRATION_MAX_SAMPLES = 120  # ~4 s at 30 fps, keeps memory bounded

# Recalibration container around the calibrated face centre, in pixels. Small
# lateral movements are tolerated; leaving the container for FACE_MOVE_FRAMES
# consecutive frames re-locks the reference, so the next up/down gesture starts
# from a fresh position. The vertical margin is deliberately wider than the
# scroll range, so normal scrolling does not interrupt tracking.
FACE_CONTAINER_X_PX = 80  # lateral half-width, in pixels
FACE_CONTAINER_Y_PX = 140  # vertical half-height, in pixels
FACE_SIZE_TOLERANCE = 0.30  # relative change of the box width/height
FACE_MOVE_FRAMES = 6  # consecutive frames outside the container before re-locking
FACE_LOST_FRAMES = 10  # consecutive frames without a face before resetting

# Privacy mode for the preview: the detected face is replaced by a solid dark
# box and the rest of the frame is pixelated, so a demo recording never shows
# the user or the room. All tracking still runs on the untouched frame.
PRIVACY_MODE = True
PRIVACY_BOX_COLOR = (35, 35, 35)  # BGR dark gray
PRIVACY_FACE_PADDING = 0.30  # expand the face box to cover hair/ears/chin
PRIVACY_PIXEL_BLOCK = 16  # pixelation block size, in pixels

SCROLL_UP_THRESHOLD = 7.8  # pixels
SCROLL_DOWN_THRESHOLD = 7.5  # pixels
COOLDOWN = 0.4  # seconds between scroll actions (prevents spamming)

SCROLL_AMOUNT = 10
SCROLL_MAX_EXPONENT = 5
SCROLL_RAMP_STEP_MS = 460  # 250
SCROLL_RELEASE_FRAMES = 4

# Hand gesture that asks for a recalibration: open hand with all five fingers
# extended and the palm plane facing the camera. Anything else (fist, pointing,
# victory, thumb up, hand edge-on, ...) is ignored.
HAND_FINGER_JOINTS = (
    (2, 3, 4),  # thumb: MCP, IP, TIP
    (5, 6, 8),  # index: MCP, PIP, TIP
    (9, 10, 12),  # middle
    (13, 14, 16),  # ring
    (17, 18, 20),  # pinky
)
HAND_FINGER_MIN_ANGLE = 145.0  # degrees at the joint (180 = perfectly straight)
HAND_PALM_MIN_FACING = 0.6  # |normal.z| / |normal| (1.0 = palm parallel to image)
HAND_MIN_CONFIDENCE = 0.7  # handedness score reported by the model
HAND_CHECK_EVERY = 2  # run the hand model every N frames (saves CPU)
HAND_CONFIRM_CHECKS = 3  # consecutive positive checks needed to trigger
HAND_RELEASE_CHECKS = 3  # consecutive negative checks needed to re-arm

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task"
)
MODEL_HAND_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task"
)
WINDOW_NAME = "LazyRead"  # OpenCV preview window
VK_CONTROL = 0x11  # Win32 virtual-key code for the Ctrl key
pyautogui.FAILSAFE = True  # move mouse to a screen corner to abort pyautogui


def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    """Download the FaceLandmarker model on first run (needs internet once)."""
    if os.path.isfile(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading FaceLandmarker model to {path} ...")
    tmp_path = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except OSError as exc:
        raise RuntimeError(
            "Could not download the MediaPipe face_landmarker model.\n"
            f"Download it manually from:\n  {url}\n"
            f"and save it as:\n  {path}"
        ) from exc
    os.replace(tmp_path, path)
    print("Model ready.")
    return path


def ensure_model_hand(path=MODEL_HAND_PATH, url=MODEL_HAND_URL):
    """Download the HandLandmarker model on first run (needs internet once)."""
    if os.path.isfile(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading HandLandmarker model to {path} ...")
    tmp_path = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except OSError as exc:
        raise RuntimeError(
            "Could not download the MediaPipe hand_landmarker model.\n"
            f"Download it manually from:\n  {url}\n"
            f"and save it as:\n  {path}"
        ) from exc
    os.replace(tmp_path, path)
    print("Model ready.")
    return path


def get_face_box(landmarks, w, h):
    """Bounding box of the face oval in pixels."""
    xs = [landmarks[i].x for i in FACE_OVAL_ANCHORS]
    ys = [landmarks[i].y for i in FACE_OVAL_ANCHORS]
    return min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h


def box_center_size(box):
    x0, y0, x1, y1 = box
    return (
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
        max(1.0, x1 - x0),
        max(1.0, y1 - y0),
    )


def draw_box(frame, box, color, thickness=1):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)


def draw_progress(frame, progress, y=95):
    x0, width, height = 50, 200, 8
    cv2.rectangle(frame, (x0, y), (x0 + width, y + height), (70, 70, 70), -1)
    cv2.rectangle(
        frame, (x0, y), (x0 + int(width * progress), y + height), (0, 255, 255), -1
    )


def pixelate(image, block=PRIVACY_PIXEL_BLOCK):
    """Downscale and upscale with nearest-neighbour to hide image details."""
    h, w = image.shape[:2]
    small = cv2.resize(
        image, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def privacy_box(box, w, h, padding=PRIVACY_FACE_PADDING):
    """Face box expanded by a margin and clamped to the frame."""
    x0, y0, x1, y1 = box
    pad_x = (x1 - x0) * padding
    pad_y = (y1 - y0) * padding
    return (
        max(0, int(x0 - pad_x)),
        max(0, int(y0 - pad_y)),
        min(w - 1, int(x1 + pad_x)),
        min(h - 1, int(y1 + pad_y)),
    )


def window_is_open(name=WINDOW_NAME):
    """False once the user closes the preview window with the X button."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        # The window handle is already gone.
        return False


def _windows_ctrl_checker():
    user32 = ctypes.windll.user32
    return lambda: bool(user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)


def _macos_ctrl_checker():
    framework = (
        "/System/Library/Frameworks/ApplicationServices.framework/"
        "ApplicationServices"
    )
    try:
        lib = ctypes.CDLL(framework)
    except OSError:
        return None
    lib.CGEventSourceKeyState.argtypes = [ctypes.c_int32, ctypes.c_uint16]
    lib.CGEventSourceKeyState.restype = ctypes.c_bool
    # 1 = kCGEventSourceStateHIDSystemState; 0x3B/0x3E = left/right Control.
    return lambda: bool(
        lib.CGEventSourceKeyState(1, 0x3B) or lib.CGEventSourceKeyState(1, 0x3E)
    )


def _x11_ctrl_checker():
    import ctypes.util

    candidates = [ctypes.util.find_library("X11"), "libX11.so.6", "libX11.so"]
    x11 = None
    for name in candidates:
        if not name:
            continue
        try:
            x11 = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if x11 is None:
        return None

    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    display = x11.XOpenDisplay(None)
    if not display:
        return None

    x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    x11.XKeysymToKeycode.restype = ctypes.c_ubyte
    keycodes = [
        x11.XKeysymToKeycode(display, 0xFFE3),  # Control_L
        x11.XKeysymToKeycode(display, 0xFFE4),  # Control_R
    ]
    keycodes = [keycode for keycode in keycodes if keycode]
    if not keycodes:
        return None

    x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
    x11.XQueryKeymap.restype = ctypes.c_int
    keys = (ctypes.c_ubyte * 32)()

    def check():
        if not x11.XQueryKeymap(display, keys):
            return False
        return any(keys[keycode // 8] & (1 << (keycode % 8)) for keycode in keycodes)

    return check


_ctrl_checker = None


def _build_ctrl_checker():
    if os.name == "nt":
        return _windows_ctrl_checker()
    if sys.platform == "darwin":
        return _macos_ctrl_checker()
    if os.name == "posix":
        # Linux and other Unix systems with an X server / XWayland.
        return _x11_ctrl_checker()
    return None


def ctrl_is_down():
    """True while the physical Ctrl key is held.

    Uses the global key state: GetAsyncKeyState on Windows, CGEventSource on
    macOS and XQueryKeymap on Unix/X11. Wheel events sent while Ctrl is down
    are interpreted as zoom by browsers, PDF viewers and similar apps, so
    scrolling is paused until Ctrl is released. On platforms where the global
    key state is not available (e.g. some Wayland sessions) this returns False.
    """
    global _ctrl_checker
    if _ctrl_checker is None:
        _ctrl_checker = _build_ctrl_checker() or (lambda: False)
    try:
        return bool(_ctrl_checker())
    except (AttributeError, OSError, ValueError):
        return False


def angle_deg(a, b, c):
    """Angle at vertex b formed by the 3D points a-b-c, in degrees."""
    v1 = a - b
    v2 = c - b
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    cos = float(np.dot(v1, v2)) / (norm1 * norm2)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def is_open_hand_toward_camera(landmarks, handedness, w, h):
    """True for an open hand (five extended fingers) facing the camera.

    Landmarks are converted to pixel space (x/z use the width scale, y the
    height) so the joint angles and the palm normal are not distorted by the
    frame aspect ratio. Palm vs. back of the hand is treated the same way:
    both are a flat "stop" hand shown to the camera.
    """
    if len(landmarks) < 21:
        return False
    if handedness is not None and handedness.score < HAND_MIN_CONFIDENCE:
        return False

    pts = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks], dtype=float)

    for mcp, pip, tip in HAND_FINGER_JOINTS:
        if angle_deg(pts[mcp], pts[pip], pts[tip]) < HAND_FINGER_MIN_ANGLE:
            return False

    # Normal of the palm plane; a large |z| means the palm is parallel to the
    # image plane, i.e. shown to the camera instead of seen edge-on.
    normal = np.cross(pts[5] - pts[0], pts[17] - pts[0])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return False
    return abs(float(normal[2])) / norm >= HAND_PALM_MIN_FACING


class ScrollRamp:
    def __init__(self):
        self.reset()

    def reset(self):
        self.direction = 0
        self.started_ms = 0
        self.idle_frames = 0

    def step(self, direction, now_ms):
        """Returns (direction, exponent) for current frame."""
        if direction == 0:
            self.idle_frames += 1
            if self.idle_frames >= SCROLL_RELEASE_FRAMES:
                self.reset()
            return 0, 0

        self.idle_frames = 0
        if direction != self.direction:
            self.direction = direction
            self.started_ms = now_ms

        exponent = (now_ms - self.started_ms) // SCROLL_RAMP_STEP_MS
        return direction, min(exponent, SCROLL_MAX_EXPONENT)


class Calibration:
    """Neutral face baseline, plus detection of head drift.

    Box centre and size are averaged with a median, so a couple of blinks or
    bad frames do not corrupt the reference and a short window is enough.
    """

    def __init__(self):
        self.reset(CALIBRATION_TIME)

    def reset(self, duration=CALIBRATION_TIME):
        self.done = False
        self.duration = duration
        self.start_time = time.time()
        self.box_samples = deque(maxlen=CALIBRATION_MAX_SAMPLES)
        self.moved_frames = 0
        self.displaced = False  # last frame was outside the tolerance container
        self.center = None  # baseline box center (normalized)
        self.size = None  # baseline box size (normalized)

    def elapsed(self):
        return time.time() - self.start_time

    def progress(self):
        return min(1.0, self.elapsed() / self.duration)

    def add_sample(self, center, size):
        self.box_samples.append((*center, *size))

    def try_lock(self):
        """Freeze the baseline once enough time and samples were collected."""
        if self.done or len(self.box_samples) < CALIBRATION_MIN_SAMPLES:
            return False
        if self.elapsed() < self.duration:
            return False

        cx, cy, bw, bh = np.median(np.asarray(self.box_samples), axis=0)
        self.center = (cx, cy)
        self.size = (bw, bh)
        self.done = True
        self.moved_frames = 0
        return True

    def face_left_container(self, center, size, w, h):
        """True if the face stayed outside the tolerance container for a while.

        The container is a box of +/- FACE_CONTAINER_X_PX and
        +/- FACE_CONTAINER_Y_PX around the baseline centre. Small lateral
        shifts are tolerated; a real sideways movement re-locks the reference
        so the vertical baseline is fresh for the next scroll gesture. The
        vertical margin is larger than the scroll range, so scrolling does not
        trigger a re-lock.
        """
        if not self.done:
            return False

        dx = abs(center[0] - self.center[0]) * w
        dy = abs(center[1] - self.center[1]) * h
        dw = abs(size[0] - self.size[0]) / max(self.size[0], 1e-6)
        dh = abs(size[1] - self.size[1]) / max(self.size[1], 1e-6)

        outside = (
            dx > FACE_CONTAINER_X_PX
            or dy > FACE_CONTAINER_Y_PX
            or dw > FACE_SIZE_TOLERANCE
            or dh > FACE_SIZE_TOLERANCE
        )
        self.displaced = outside
        self.moved_frames = self.moved_frames + 1 if outside else 0
        return self.moved_frames >= FACE_MOVE_FRAMES

    def container_box(self, w, h):
        """Pixel box of the tolerated face-centre movement."""
        cx, cy = self.center
        return (
            cx * w - FACE_CONTAINER_X_PX,
            cy * h - FACE_CONTAINER_Y_PX,
            cx * w + FACE_CONTAINER_X_PX,
            cy * h + FACE_CONTAINER_Y_PX,
        )

    def baseline_box(self, w, h):
        """Baseline face area in pixels (for drawing the reference box)."""
        cx, cy = self.center
        bw, bh = self.size
        return (
            (cx - bw / 2) * w,
            (cy - bh / 2) * h,
            (cx + bw / 2) * w,
            (cy + bh / 2) * h,
        )


def main():
    model_path = ensure_model()
    model_hand_path = ensure_model_hand()
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_hand_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        face_landmarker.close()
        hand_landmarker.close()
        raise RuntimeError("Could not open webcam. Check the camera index/permissions.")

    calibration = Calibration()
    scroll_ramp = ScrollRamp()
    last_scroll_time = 0.0
    last_timestamp_ms = 0
    missing_frames = 0

    hand_check_frame = 0
    hand_gesture_checks = 0
    hand_release_checks = HAND_RELEASE_CHECKS
    hand_latched = False
    hand_request_recalibration = False
    last_hand_landmarks = []
    last_hand_ok = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    window_ready = False
    privacy_mode = PRIVACY_MODE

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror view, feels more natural
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Privacy mode only affects the preview: detection keeps using the
            # untouched rgb_frame. Pixelation happens before the overlays so
            # boxes, labels and hand dots stay sharp.
            if privacy_mode:
                frame = pixelate(frame, PRIVACY_PIXEL_BLOCK)

            # VIDEO mode needs strictly increasing timestamps.
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Hand gesture, throttled (the hand model is only needed a few
            # times per second). The trigger is edge-based: it fires once when
            # the open hand has been held for a few checks and only re-arms
            # after the hand is lowered or closed.
            hand_check_frame += 1
            hand_request_recalibration = False
            if hand_check_frame % HAND_CHECK_EVERY == 0:
                hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                last_hand_landmarks = hand_results.hand_landmarks
                last_hand_ok = [
                    is_open_hand_toward_camera(
                        hand_landmarks, hand_results.handedness[i][0], w, h
                    )
                    for i, hand_landmarks in enumerate(hand_results.hand_landmarks)
                ]

                if any(last_hand_ok):
                    hand_gesture_checks += 1
                    hand_release_checks = 0
                else:
                    hand_gesture_checks = 0
                    hand_release_checks += 1
                    if hand_release_checks >= HAND_RELEASE_CHECKS:
                        hand_latched = False

                if hand_gesture_checks >= HAND_CONFIRM_CHECKS and not hand_latched:
                    hand_latched = True
                    hand_request_recalibration = True

            # Green dots = recognised open hand, orange = hand doing something
            # else (so it is clear why a recalibration did not start).
            for hand_landmarks, ok in zip(last_hand_landmarks, last_hand_ok):
                color = (0, 255, 0) if ok else (0, 165, 255)
                for lm in hand_landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, color, -1)

            if results.face_landmarks:
                # Face found again after a loss: the old baseline is meaningless.
                if missing_frames >= FACE_LOST_FRAMES:
                    calibration.reset(RECALIBRATION_TIME)
                    last_scroll_time = time.time()
                missing_frames = 0

                landmarks = results.face_landmarks[0]

                box = get_face_box(landmarks, w, h)
                cx, cy, bw, bh = box_center_size(box)
                center = (cx / w, cy / h)
                size = (bw / w, bh / h)

                # Solid gray box over the face (padded to cover hair/ears);
                # the tracking boxes are drawn on top of it afterwards.
                if privacy_mode:
                    px0, py0, px1, py1 = privacy_box(box, w, h)
                    cv2.rectangle(
                        frame, (px0, py0), (px1, py1), PRIVACY_BOX_COLOR, -1
                    )

                if not calibration.done:
                    scroll_ramp.reset()
                    calibration.add_sample(center, size)

                    if calibration.try_lock():
                        last_scroll_time = time.time()
                        cv2.putText(
                            frame,
                            "Calibrated - tracking ready",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        draw_box(frame, box, (0, 255, 255), 2)
                    else:
                        remaining = max(
                            0.0, calibration.duration - calibration.elapsed()
                        )
                        cv2.putText(
                            frame,
                            "Look straight ahead - calibrating...",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Time remaining: {remaining:.1f}s",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame,
                            "Keep your face inside the yellow box",
                            (50, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (200, 200, 200),
                            1,
                        )
                        draw_progress(frame, calibration.progress())
                        draw_box(frame, box, (0, 255, 255), 1)

                elif calibration.face_left_container(center, size, w, h):
                    # Face left the tolerance container: re-lock the reference
                    # in this new position, ready for the next scroll gesture.
                    calibration.reset(RECALIBRATION_TIME)
                    last_scroll_time = time.time()
                    cv2.putText(
                        frame,
                        "Face moved - re-calibrating...",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    draw_box(frame, box, (0, 0, 255), 2)

                elif hand_request_recalibration:
                    # Open hand with five extended fingers shown to the
                    # camera: re-lock the face reference once.
                    calibration.reset(RECALIBRATION_TIME)
                    scroll_ramp.reset()
                    last_scroll_time = time.time()
                    cv2.putText(
                        frame,
                        "Open hand - re-calibrating...",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    draw_box(frame, box, (0, 0, 255), 2)

                else:
                    draw_box(frame, calibration.container_box(w, h), (110, 110, 110), 1)
                    draw_box(frame, calibration.baseline_box(w, h), (0, 150, 150), 1)
                    draw_box(frame, box, (0, 255, 0), 1)

                    # Vertical movement of the face centre, in pixels.
                    dy = (center[1] - calibration.center[1]) * h

                    now = time.time()
                    if calibration.displaced:
                        # Head is already drifting out of the area: wait for the
                        # re-lock counter instead of scrolling on a stale reference.
                        scroll_ramp.reset()
                        cv2.putText(
                            frame,
                            "Hold still...",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 165, 255),
                            2,
                        )
                    else:
                        direction = (
                            -1
                            if dy > SCROLL_DOWN_THRESHOLD
                            else 1
                            if dy < -SCROLL_UP_THRESHOLD
                            else 0
                        )

                        if ctrl_is_down():
                            # Ctrl+wheel zooms the active app instead of
                            # scrolling: pause until the key is released.
                            scroll_ramp.reset()
                            if direction:
                                cv2.putText(
                                    frame,
                                    "Ctrl held - scrolling paused",
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 165, 255),
                                    2,
                                )
                        else:
                            direction, exponent = scroll_ramp.step(
                                direction, int(now * 1000)
                            )

                            if direction and now - last_scroll_time > COOLDOWN:
                                pyautogui.scroll(
                                    direction * (SCROLL_AMOUNT << exponent)
                                )
                                last_scroll_time = now
                                cv2.putText(
                                    frame,
                                    (
                                        "Scrolling down"
                                        if direction < 0
                                        else "Scrolling up"
                                    ),
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0) if direction < 0 else (0, 0, 255),
                                    2,
                                )
            else:
                missing_frames += 1
                cv2.putText(
                    frame,
                    "No face detected - look at the camera",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                "Press 'q' to quit, 'r' to recalibrate, 'k' to pause, 'p' privacy",
                (50, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            if privacy_mode:
                cv2.putText(
                    frame,
                    "Privacy: ON",
                    (max(10, w - 150), 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180),
                    1,
                )

            # Once the X button destroys the window, the next imshow would
            # recreate it (with a fresh size): stop before that happens.
            if window_ready and not window_is_open():
                break
            cv2.imshow(WINDOW_NAME, frame)
            window_ready = True

            key = cv2.waitKey(1) & 0xFF
            if not window_is_open():
                break
            if key == ord("q"):
                break
            elif key == ord("r"):
                calibration.reset(CALIBRATION_TIME)
            elif key == ord("p"):
                privacy_mode = not privacy_mode
            # stop tracking for a bit
            # waits for key "k" to be re-typed
            elif key == ord("k"):
                scroll_ramp.reset()
                last_scroll_time = time.time()
                resumed = False
                while not resumed and window_is_open():
                    stop_key = cv2.waitKey(1) & 0xFF
                    if stop_key == ord("k"):
                        resumed = True
                    elif stop_key == ord("q"):
                        break
                if not resumed:
                    break

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        face_landmarker.close()
        hand_landmarker.close()


if __name__ == "__main__":
    main()
