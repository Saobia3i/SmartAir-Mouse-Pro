"""Constants for SmartAir Mouse Pro.

Defines default settings, gesture configurations, UI styling parameters, and MediaPipe landmarks.
"""

from typing import Dict, Any

# Screen settings
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
TARGET_FPS = 60

# MediaPipe Landmark Indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Gestures
GESTURE_MOVE = "MOVE"
GESTURE_LEFT_CLICK = "LEFT_CLICK"
GESTURE_RIGHT_CLICK = "RIGHT_CLICK"
GESTURE_DRAG = "DRAG"
GESTURE_SCROLL = "SCROLL"
GESTURE_SCREENSHOT = "SCREENSHOT"
GESTURE_PAUSE = "PAUSE"
GESTURE_LOCK = "LOCK"
GESTURE_NONE = "NONE"

GESTURE_DESCRIPTIONS = {
    GESTURE_MOVE: "Index finger: Move cursor",
    GESTURE_LEFT_CLICK: "Thumb + Index pinch: Left Click",
    GESTURE_RIGHT_CLICK: "Thumb + Middle pinch: Right Click",
    GESTURE_DRAG: "Index + Middle pinch: Drag",
    GESTURE_SCROLL: "Two fingers vertical: Scroll",
    GESTURE_SCREENSHOT: "Three fingers: Screenshot",
    GESTURE_PAUSE: "Open Palm: Pause Tracking",
    GESTURE_LOCK: "Closed Fist: Lock Cursor",
    GESTURE_NONE: "No Hand Detected"
}

# Settings Defaults
DEFAULT_SETTINGS: Dict[str, Any] = {
    "camera_index": 0,
    "sensitivity_x": 1.5,
    "sensitivity_y": 1.5,
    "smoothing_factor": 0.20,  # Kalman + EMA
    "trail_length": 15,
    "trail_color": "#00FFCC",  # Neon cyan
    "cursor_size": 12,
    "click_threshold": 0.04,  # Normalized distance for pinch click
    "scroll_speed": 1.0,
    "particles_enabled": True,
    "sound_enabled": True,
    "theme_mode": "dark",
    # Calibration results
    "reach_xmin": 0.2,
    "reach_xmax": 0.8,
    "reach_ymin": 0.2,
    "reach_ymax": 0.8,
    "pinch_threshold_left": 0.04,
    "pinch_threshold_right": 0.04,
    "hand_size_baseline": 0.3
}

# UI/UX & Overlay Colors
COLOR_NEON_BLUE = "#0066FF"
COLOR_NEON_GREEN = "#39FF14"
COLOR_NEON_RED = "#FF3131"
COLOR_NEON_YELLOW = "#FFF01F"
COLOR_NEON_CYAN = "#00FFCC"
COLOR_BG_DARK = "#121212"
COLOR_SIDEBAR_DARK = "#1E1E1E"

# Timing
DEBOUNCE_FRAMES = 3
COOLDOWN_SCREENSHOT = 3.0  # seconds
COOLDOWN_CLICK = 0.3  # seconds
