# LazyRead
 
LazyRead scrolls your screen up/down by moving your head (the **face position**, not your gaze), using your webcam and MediaPipe FaceLandmarker. The name comes from the laziness of scrolling long text by hand. Show an open hand to the camera to force a re-calibration.

> **Background:** LazyRead is a fork of the original [AutoScroller](https://github.com/DissanayakeLYB/AutoScroller) demo. Despite the old "eye tracking" name, the real control signal is the face position. See [Naming](#naming) below.
 
## What it does
1. Opens your webcam and detects your face.
2. For the first 2 seconds it **calibrates**: it locks the neutral face position and the area (bounding box) where your face currently is.
3. After that, if your **face** moves up or down past a small threshold, it scrolls the active window accordingly. Moving only your eyes does not scroll, because the face box barely moves without the head.
4. If your face leaves the tolerance container (a real sideways shift, or leaning in/out), it finds it again and **re-calibrates automatically** (about 1 second), so the old reference cannot cause false scrolling.
5. You can also force a re-calibration by showing an **open hand with all five fingers extended** to the camera. Any other hand pose (fist, pointing, victory, thumbs-up, hand seen edge-on) is ignored.
6. Press **`r`** any time to recalibrate, **`k`** to pause/resume, **`q`** to quit.

## Naming
The original AutoScroller project started as an eye-tracking demo, but its implementation averaged the pixel position of a few **eye-contour landmarks** as the scroll signal. Those landmarks are attached to the head: when you raise or lower your face they all move together, while a change of gaze moves them only slightly (the iris/pupil was not tracked). The real control signal was therefore the **face position**, not where you are looking.

LazyRead is named after the laziness of scrolling long text by hand. It is a fork of that original demo, but it no longer reads the eye landmarks: the scroll signal comes from the **face-box centre directly**, so the old "eye tracking" misnomer is documented here but no longer present in the code.

## Requirements
- Python 3.9+ (tested on 3.13)
- MediaPipe **1.x** (the old `mp.solutions` API was removed)
- A working webcam
- Internet access on first run (to download the ~3.7 MB `face_landmarker.task` and the ~7.8 MB `hand_landmarker.task` models)

## Setup
```bash
pip install -r requirements.txt
```

On first run the script downloads the FaceLandmarker and HandLandmarker models to `models/`.
 
On Linux you may also need:
```bash
sudo apt-get install python3-tk python3-dev scrot
```
(needed by `pyautogui` for screen control on some distros)
 
## Run
```bash
python LazyRead.py
```
 
A window titled **LazyRead** will open showing your webcam feed with the face area drawn as a box.
Hand landmarks are drawn too: **green dots** mean the open-hand gesture was recognised, **orange dots** mean the hand is doing something else and will not trigger a re-calibration.
 
## Usage tips
- During calibration, keep your head still inside the **yellow box** and look straight at the screen. The baseline is the median of ~2 s of frames, so a blink or two does not spoil it.
- While tracking, the **gray box is the tolerance container**: small sideways shifts inside it are ignored. The thin yellow box is the calibrated face reference and the green box is where the face is now.
- Small, deliberate up/down **head** movements trigger scrolling; moving only your eyes (gaze) has little effect. Normal blinking won't trigger anything.
- **Hand re-calibration:** raise your hand with all five fingers extended and the palm (or back of the hand) facing the camera, like a "stop" gesture, and hold it for a moment. Green landmark dots confirm the pose; orange dots mean it was not recognised.
- If scrolling feels too sensitive or too sluggish, tweak these constants near the top of `LazyRead.py`:
  - `SCROLL_UP_THRESHOLD` / `SCROLL_DOWN_THRESHOLD` — how far your **face** must move (in pixels, measured from the face box) to trigger a scroll
  - `SCROLL_AMOUNT` — base scroll step; `SCROLL_RAMP_STEP_MS` / `SCROLL_MAX_EXPONENT` control the acceleration (up to 16x)
  - `COOLDOWN` — minimum time between scroll actions
  - `CALIBRATION_TIME` / `RECALIBRATION_TIME` — initial and automatic re-calibration duration
  - `FACE_CONTAINER_X_PX` — lateral tolerance in pixels (default 80): sideways movement inside ±this value is ignored, so normal head jitter does not cost a re-calibration; leaving it for a few frames re-locks the reference
  - `FACE_CONTAINER_Y_PX` — vertical tolerance in pixels (default 140), deliberately wider than the scroll range so a normal up/down gesture does not interrupt tracking
  - `FACE_MOVE_FRAMES` / `FACE_LOST_FRAMES` — consecutive frames required before declaring the face outside the container or lost (higher = more tolerant)
  - `HAND_FINGER_MIN_ANGLE` — joint angle (degrees) above which a finger counts as extended; lower it if your thumb is not recognised
  - `HAND_PALM_MIN_FACING` — how front-facing the palm must be (0–1, higher = stricter)
  - `HAND_CONFIRM_CHECKS` / `HAND_RELEASE_CHECKS` — consecutive checks needed to trigger and to re-arm the gesture
  - `HAND_CHECK_EVERY` — runs the hand model every N frames (1 = every frame, higher = lighter CPU)
- Move your mouse to a screen corner at any time to trigger pyautogui's fail-safe and stop automated input immediately.

## What changed from AutoScroller
- Rebranded as **LazyRead**: the entry point is now `LazyRead.py` and the preview window is titled `LazyRead`.
- Added a lateral tolerance container: small sideways head movements are ignored, while a real lateral shift re-locks the vertical baseline automatically.
- Removed dead code (`locked_left_eye` / `locked_right_eye` were computed but never used).
- Fixed structure: previously the capture loop ran at import time regardless of `__main__`, and cleanup only ran inside the `__main__` guard — this could leak the camera handle. Now everything is wrapped in `main()` and released in a `finally` block, so the webcam is released even if an error occurs.
- Added a webcam-open check (`cap.isOpened()`), since the original assumed the camera always opens successfully.
- Added a scroll **cooldown** so one sustained head movement doesn't fire dozens of scroll events per second.
- Added a mirror flip (`cv2.flip`) so the feed matches natural mirror expectations.
- Added a recalibration key (`r`) and an on-screen "no face detected" message.
- Replaced the manual `boundingRect` eye-centroid math with the face-box centre (fewer moving parts, same vertical signal).
- Properly closes the MediaPipe FaceLandmarker instance on exit.
- Documented and fixed a long-standing misnomer: the script tracks the **face position**, not the eyes. The averaged eye landmarks shifted 1:1 with the head, while a pure gaze change barely moved them, so raising/lowering the face is the real input. The scroll signal now comes from the face-box centre, and the eye landmarks are no longer read.

## Origin and license
LazyRead is a fork of [DissanayakeLYB/AutoScroller](https://github.com/DissanayakeLYB/AutoScroller), released under the MIT License. The original copyright notice is kept in [LICENSE](LICENSE).

## Face-area calibration
- During calibration the script records the face's bounding box (from 36 face-oval landmarks, not all 478, so it stays cheap). Its centre and size are averaged with a **median**, which is why 2 s are enough: blinks and single bad frames are ignored.
- The vertical scroll signal is the **face-box centre** relative to that baseline: head translation moves it directly, while moving just your eyes does not. The eye landmarks are no longer read.
- While tracking, the face-box centre is compared with the baseline every frame. A **tolerance container** of ±`FACE_CONTAINER_X_PX` × ±`FACE_CONTAINER_Y_PX` pixels around the baseline is drawn in gray: small sideways shifts inside it are ignored, but leaving it for `FACE_MOVE_FRAMES` consecutive frames discards the baseline and re-locks in ~1 s, so the next vertical gesture starts from a fresh position. Box size changes beyond `FACE_SIZE_TOLERANCE` also trigger a re-lock (leaning in/out).
- The lateral tolerance is the point of the container: the head never stays perfectly still, so lateral jitter should not cost a re-calibration. When the drift is real, the re-lock is actually useful: it re-centres the vertical reference before the next up/down movement.
- If the face is not detected for `FACE_LOST_FRAMES` frames, the old baseline is discarded too; when the face comes back, calibration restarts from the new position.
- MediaPipe's VIDEO mode already re-detects the face when tracking is lost, so "finding it again" needs no extra inference pass.

## Hand-gesture recalibration
- The HandLandmarker model returns 21 landmarks. The gesture counts as "open hand toward the camera" only when **all five fingers are extended** (angle at the MCP/IP joint ≥ `HAND_FINGER_MIN_ANGLE`, 145°) **and** the palm plane is roughly parallel to the image plane (`|normal.z| / |normal| ≥ HAND_PALM_MIN_FACING`, 0.6). A minimum model confidence of 0.7 rejects uncertain detections.
- The model is configured for **up to two hands** (`num_hands=2`, detection confidence 0.4), so the gesture works with either hand and the open right hand is still found when the other hand is also in frame. The trigger fires if **any** detected hand matches the pose.
- Palm and back of the hand are treated the same: both are a flat "stop" hand shown to the camera.
- Fist, pointing, victory, thumbs-up and hands seen edge-on fail the angle or palm-plane test, so ordinary hand movements do not reset the calibration.
- The trigger is **edge-based**: it fires after `HAND_CONFIRM_CHECKS` consecutive positive checks and stays latched until the hand disappears for `HAND_RELEASE_CHECKS` checks. This is why holding the open hand does not cause an endless recalibration loop.
- A gesture shown during the initial calibration is ignored (the calibration is already running); lower the hand and show it again afterwards.
- The hand model is the expensive one (~12 ms/frame with one hand, ~23 ms with two, vs ~3 ms for the face), so it runs every `HAND_CHECK_EVERY` frames (default 2). With the face model also running, the whole loop costs ~12–13 ms/frame on CPU.

## Known limitations
- This is **face-position tracking, not gaze tracking**: nodding, leaning or shifting your posture scrolls the page, and moving your eyes alone has little effect. That is the intended behaviour of this project; automatic re-calibration resets the reference after large movements.
- Works with one face only (`max_num_faces=1`).
- Lighting conditions affect landmark detection accuracy.