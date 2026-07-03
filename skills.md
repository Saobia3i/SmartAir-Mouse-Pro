# Developer Skills & Coding Guidelines: SmartAir Mouse Pro

This document outlines the development standards, GUI patterns, and packaging techniques utilized in **SmartAir Mouse Pro**. It serves as a reference for extending or modifying the codebase.

---

## 💻 Codebase Coding Standards

To maintain clean architecture, all code contributions must follow these guidelines:

1. **Strict Type Hinting**: All function signatures must include full type annotations (e.g. `def process_frame(self) -> Tuple[Optional[Any], Dict[str, Any]]:`).
2. **No print() Statements**: Use Python's built-in `logging` module to log events. Logging config is set up in `utils.py` and streams to standard output and `app.log`.
3. **Google-Style Docstrings**: Document all classes and functions:
   ```python
   def calculate_distance(pt1: Tuple[float, float, float], pt2: Tuple[float, float, float]) -> float:
       """Calculates 3D Euclidean distance between two landmarks.

       Args:
           pt1: Landmark 1 (x, y, z).
           pt2: Landmark 2 (x, y, z).

       Returns:
           Euclidean distance.
       """
   ```
4. **DRY & SOLID Principles**: Ensure logic (mouse math, tracking) is completely separate from representation (Tkinter HUDs, browser interfaces).

---

## 🎨 Tkinter Fullscreen Overlays Tips

Creating fullscreen overlays that are transparent, stay on top, and allow click-through requires specific Windows attributes:

1. **Borderless Geometry**:
   ```python
   window.overrideredirect(True)
   window.geometry(f"{screen_w}x{screen_h}+0+0")
   ```
2. **Topmost & Transparency Key Color**:
   Set `-topmost` to stay on top, and set `-transparentcolor` to match the canvas background color (which makes that background fully transparent and click-through on Windows):
   ```python
   window.attributes("-topmost", True)
   window.attributes("-transparentcolor", "#010101")
   ```
3. **Click-Through Windows Styles (WS_EX_TRANSPARENT)**:
   To ensure clicks fall through the canvas to elements below, call pywin32 functions to modify window extended styles:
   ```python
   import win32gui
   import win32con
   hwnd = window.winfo_id()
   styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
   new_styles = styles | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED
   win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, new_styles)
   ```

---

## 📷 MediaPipe Tasks & OpenCV Guidelines

1. **MediaPipe Image Formats**: MediaPipe Tasks accepts frames wrapped in `mp.Image` objects rather than raw numpy arrays:
   ```python
   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_frame_rgb)
   ```
2. **Headless OpenCV**: When deploying online (like on Hugging Face Spaces), always use `opencv-python-headless` instead of `opencv-python` to avoid crashing due to missing X11 libraries on Linux containers.
3. **Manual Skeleton Drawing**: To bypass import discrepancies across different MediaPipe versions, draw the skeleton lines directly on the frame using OpenCV's `cv2.line` and joint indexes, rather than importing drawing solutions.

---

## 📦 PyInstaller Packaging Guide

To package SmartAir Mouse Pro into a standalone executable (`.exe`) on Windows:

1. **Install PyInstaller**:
   ```bash
   py -m pip install pyinstaller
   ```
2. **Execute PyInstaller Script**:
   Configure PyInstaller to bundle dependencies, collect metadata, and include the task file model:
   ```bash
   pyinstaller --noconsole --name="SmartAirMousePro" --add-data "assets;assets" app.py
   ```
   * `--noconsole`: Hides the secondary command-prompt console window when launching the desktop app.
   * `--add-data "assets;assets"`: Ensures your synthesized WAV assets and downloaded `hand_landmarker.task` file are correctly bundled into the distribution directories.
