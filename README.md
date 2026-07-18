# Eye-Tracking Scroll Controller
 
Scroll your screen up/down by moving your eyes, using your webcam and MediaPipe FaceMesh.
 
## What it does
1. Opens your webcam and detects your face.
2. For the first 5 seconds, it **calibrates** to your neutral eye position (look straight ahead).
3. After that, if your eyes move up or down past a small threshold, it scrolls the active window accordingly.
4. Press **`r`** any time to recalibrate, **`q`** to quit.
## Requirements
- Python 3.9–3.11 recommended (MediaPipe support varies by version)
- A working webcam
## Setup
```bash
pip install opencv-python mediapipe numpy pyautogui
```
 
On Linux you may also need:
```bash
sudo apt-get install python3-tk python3-dev scrot
```
(needed by `pyautogui` for screen control on some distros)
 
## Run
```bash
python eye_scroll.py
```
 
A window titled **"Eye Tracking and Scrolling"** will open showing your webcam feed with eye landmarks marked in green.
 
## Usage tips
- During calibration, keep your head still and look straight at the screen.
- Small, deliberate up/down eye movements trigger scrolling — normal blinking won't.
- If scrolling feels too sensitive or too sluggish, tweak these constants near the top of `eye_scroll.py`:
  - `SCROLL_UP_THRESHOLD` / `SCROLL_DOWN_THRESHOLD` — how far your eyes must move (in pixels) to trigger a scroll
  - `SCROLL_AMOUNT` — how much each scroll event moves the page
  - `COOLDOWN` — minimum time between scroll actions
- Move your mouse to a screen corner at any time to trigger pyautogui's fail-safe and stop automated input immediately.
## What changed from the original script
- Removed dead code (`locked_left_eye` / `locked_right_eye` were computed but never used).
- Fixed structure: previously the capture loop ran at import time regardless of `__main__`, and cleanup only ran inside the `__main__` guard — this could leak the camera handle. Now everything is wrapped in `main()` and released in a `finally` block, so the webcam is released even if an error occurs.
- Added a webcam-open check (`cap.isOpened()`), since the original assumed the camera always opens successfully.
- Added a scroll **cooldown** so one sustained eye movement doesn't fire dozens of scroll events per second.
- Added a mirror flip (`cv2.flip`) so the feed matches natural hand/eye mirror expectations.
- Added a recalibration key (`r`) and an on-screen "no face detected" message.
- Replaced the manual `boundingRect` centroid math with a simpler landmark-mean centroid (fewer lines, same effect).
- Properly closes the MediaPipe `FaceMesh` instance on exit.

## Known limitations
- Eye-position-based scrolling is sensitive to head movement — moving your whole head will look like eye movement too, since it's based on pixel position, not gaze direction relative to your face.
- Works with one face only (`max_num_faces=1`).
- Lighting conditions affect landmark detection accuracy.