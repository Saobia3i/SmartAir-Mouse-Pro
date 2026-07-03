# Project Development Context: SmartAir Mouse Pro

This document serves as the developer and maintainer context guide for **SmartAir Mouse Pro**. It outlines the application’s design decisions, threading models, mathematics formulas, and components interactions.

---

## 🏗️ Architecture Design

The application follows a **Modular Clean Architecture** pattern where input frame capture, coordinate filtering, gesture state processing, and GUI drawing are completely decoupled.

```text
+-------------------+
|  Camera Source    | (Webcam Capture via OpenCV)
+---------+---------+
          |
          v [raw image frame]
+---------+---------+
|   HandTracker     | (MediaPipe Tasks API - extracts joint landmarks)
+---------+---------+
          |
          v [21 landmarks (x, y, z)]
+---------+---------+
| GestureRecognizer | (Classification via spatial rules & hysteresis)
+---------+---------+
          |
          v [gesture state & raw pointer coordinate]
+---------+---------+
|  MouseController  | (2D Kalman Filtering, EMA, and pointer injection)
+---------+---------+
          |
          +-------------------------+-------------------------+
          | (Cursor coordinates)    | (Click actions)         | (HUD stats)
          v                         v                         v
+---------+---------+     +---------+---------+     +---------+---------+
|  CursorEffects    |     |  OS Mouse Driver  |     |   Desktop GUI     |
| (Sparks, Ripples) |     |  (pynput injection)     | (CustomTkinter)   |
+---------+---------+     +-------------------+     +-------------------+
          |
          v [trail lines & particles]
+---------+---------+
|  OverlayWindow    | (Transparent topmost click-through Win32 canvas)
+-------------------+
```

---

## 🧵 Threading Model & Synchronization

To maintain a consistent **60 FPS** target for both cursor responsiveness and GUI animations, the application employs a multithreaded architecture:

1. **Main UI Thread**:
   - Manages the CustomTkinter desktop interface.
   - Spawns the fullscreen borderless `OverlayWindow` as a Toplevel widget.
   - Runs a polling tick loop at 60 Hz (`self.after(16, self._poll_ui_queues)`) that pulls processed frames and metadata from thread-safe queues, rendering trails, HUD text, and camera previews.
   - Handles the settings inputs and calibration modals.

2. **Background Tracking Worker (Daemon Thread)**:
   - Continuously captures frames from OpenCV Video Capture.
   - Processes landmark extraction via MediaPipe `HandLandmarker` task.
   - Runs gesture calculations, updates coordinates using the Kalman filter, and triggers virtual OS mouse events (clicks, movement, drag-releases) asynchronously using pynput.
   - Feeds the display queues.

3. **Throw-Away Action Threads**:
   - Screenshot saving triggers: Writing screenshots to disk is offloaded to a daemon thread to prevent dropping frames on target captures.

---

## 📈 Mathematics & Signal Processing

### 1. 2D Kalman Filter
To remove micro-tremors from the user's hand without introducing noticeable lag, a 2D Kalman Filter is used. 
* **State Vector**: $x_t = [x, y, v_x, v_y]^T$ where $(x, y)$ are screen coordinates and $(v_x, v_y)$ are velocities.
* **Transition Matrix $F$**:
  $$F = \begin{bmatrix} 1 & 0 & dt & 0 \\ 0 & 1 & 0 & dt \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
* **Measurement Matrix $H$**:
  $$H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$
* **Process Noise $Q$** and **Measurement Noise $R$** are calibrated to balance lag vs. smoothness.

### 2. Exponential Moving Average (EMA)
The output of the Kalman filter is interpolated with the previous cursor position:
$$S_t = \alpha \cdot K_t + (1 - \alpha) \cdot S_{t-1}$$
Where $S_t$ is the smoothed output coordinate, $K_t$ is the Kalman filtered estimate, and $\alpha \in [0.05, 0.8]$ is the user-configured smoothing factor.

### 3. Screen Mapping with Edge Compensation
The hand coordinates $(hx, hy)$ are mapped from an inner calibrated bounding region $[xmin, xmax] \times [ymin, ymax]$ to the screen coordinates:
$$screen_x = \text{clamp}\left( \frac{hx - xmin}{xmax - xmin}, 0, 1 \right) \cdot ScreenWidth$$
$$screen_y = \text{clamp}\left( \frac{hy - ymin}{ymax - ymin}, 0, 1 \right) \cdot ScreenHeight$$
This allows comfortable movements within a small physical zone while allowing the cursor to reach the screen boundaries easily (edge compensation).

---

## 📂 Configuration Storage

All configurations are saved in [settings.json](file:///e:/vs%20code%20projects/smartMourse/SmartAir-Mouse-Pro/config/settings.json) and thread-safely managed by [settings.py](file:///e:/vs%20code%20projects/smartMourse/SmartAir-Mouse-Pro/settings.py). 

During startup, the application auto-creates the directory structure:
- `config/` (Settings parameters)
- `screenshots/` (Saves captured screenshots)
- `assets/` (Stores `hand_landmarker.task` and programmatically generates synthesized wav audios)
