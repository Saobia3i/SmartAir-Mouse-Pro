"""Gradio Web Application for SmartAir Mouse Pro.

Runs a browser-based demo showcasing real-time camera capture, MediaPipe Tasks tracking,
gesture classification overlays, and confidence metrics directly in the web browser.
"""

import os
import sys
import threading
import cv2
import urllib.request
import numpy as np
import gradio as gr
from pathlib import Path
from typing import Tuple, Dict, Any

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ensure workspace paths resolved
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from constants import GESTURE_DESCRIPTIONS
from gesture_recognizer import GestureRecognizer

# Ensure model downloaded
assets_dir = Path("./assets").resolve()
assets_dir.mkdir(parents=True, exist_ok=True)
model_path = assets_dir / "hand_landmarker.task"

if not model_path.exists():
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    print(f"Downloading hand_landmarker.task from {url} to {model_path}...")
    try:
        import ssl
        context = ssl._create_unverified_context()
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, context=context, timeout=30) as response, open(model_path, "wb") as out_file:
            out_file.write(response.read())
        print("Download completed successfully.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        raise RuntimeError(f"Could not download hand_landmarker.task: {e}")

# Initialize MediaPipe Tasks Hand Landmarker
base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# Initialize Gesture Recognizer (using a standard default click threshold of 0.045)
recognizer = GestureRecognizer(click_threshold=0.045)

# Thread safety lock for MediaPipe Tasks Hand Landmarker (inference is not thread-safe)
mp_lock = threading.Lock()



def process_webcam_frame(frame: Any, pinch_threshold: float = 0.045) -> Tuple[Any, str, float]:
    """Processes a frame from the browser webcam, draws tracking data, and predicts gestures.

    Args:
        frame: Input RGB image from webcam (numpy array).

    Returns:
        A tuple of (annotated_rgb_frame, gesture_name, confidence).
    """
    try:
        recognizer.update_thresholds(float(pinch_threshold))

        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8), "No Frame", 0.0

        # Extract image if frame is passed as a dictionary (Gradio 5 Image/Editor data structure)
        if isinstance(frame, dict):
            if "composite" in frame and frame["composite"] is not None:
                frame = frame["composite"]
            elif "background" in frame and frame["background"] is not None:
                frame = frame["background"]
            elif "image" in frame and frame["image"] is not None:
                frame = frame["image"]
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8), "Invalid Dict Frame", 0.0

        # Double check frame is indeed a valid numpy array
        if not isinstance(frame, np.ndarray):
            return np.zeros((480, 640, 3), dtype=np.uint8), f"Invalid Type: {type(frame).__name__}", 0.0

        # Flip horizontally for mirrored view
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # MediaPipe process (Gradio input frames are already in RGB format)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        with mp_lock:
            results = detector.detect(mp_image)

        gesture_name = "NONE"
        confidence = 0.0

        if results.hand_landmarks and results.handedness:
            landmarks = results.hand_landmarks[0]
            handedness = results.handedness[0][0]

            label = handedness.category_name
            score = handedness.score

            # Extract normalized coordinates
            landmark_list = [(lm.x, lm.y, lm.z) for lm in landmarks]

            # Recognize gesture
            gesture_name, confidence = recognizer.recognize(landmark_list)

            # Draw skeleton connections manually using OpenCV
            connections = [
                # Thumb
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Index
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Middle
                (9, 10), (10, 11), (11, 12),
                # Ring
                (13, 14), (14, 15), (15, 16),
                # Pinky
                (0, 17), (17, 18), (18, 19), (19, 20),
                # Palm Knuckles
                (5, 9), (9, 13), (13, 17)
            ]

            # Draw skeleton lines
            for start, end in connections:
                p1 = (int(landmark_list[start][0] * width), int(landmark_list[start][1] * height))
                p2 = (int(landmark_list[end][0] * width), int(landmark_list[end][1] * height))
                cv2.line(frame, p1, p2, (57, 255, 20), 2)  # Neon Green

            # Draw joints
            for pt in landmark_list:
                p = (int(pt[0] * width), int(pt[1] * height))
                cv2.circle(frame, p, 4, (0, 240, 255), -1)  # Cyan

            # Compute bounding box
            x_coords = [lm[0] for lm in landmark_list]
            y_coords = [lm[1] for lm in landmark_list]
            xmin = int(min(x_coords) * width)
            xmax = int(max(x_coords) * width)
            ymin = int(min(y_coords) * height)
            ymax = int(max(y_coords) * height)

            # Add padding and draw bounding box
            padding = 10
            xmin = max(0, xmin - padding)
            ymin = max(0, ymin - padding)
            xmax = min(width, xmax + padding)
            ymax = min(height, ymax + padding)

            # Draw Bounding Box (Red for right hand, Blue for left hand)
            box_color = (0, 102, 255) if label == "Left" else (57, 255, 20)  # BGR
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)

            # Draw Gesture Label on frame
            cv2.putText(
                frame,
                f"{gesture_name} ({int(confidence*100)}%)",
                (xmin, max(30, ymin - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Print Hand Info
            cv2.putText(
                frame,
                f"Hand: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 204),
                2,
                cv2.LINE_AA
            )

        # Return frame, active gesture label and details
        return frame, gesture_name, float(confidence)

    except Exception as e:
        import traceback
        print(f"Error in process_webcam_frame: {e}")
        traceback.print_exc()
        return np.zeros((480, 640, 3), dtype=np.uint8), f"Error: {str(e)}", 0.0


# ==========================================
# GRADIO UI BUILDER
# ==========================================
theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    font=["Arial", "sans-serif"],
    font_mono=["Consolas", "monospace"],
).set(
    body_background_fill="#121212",
    body_text_color="#FFFFFF",
    block_background_fill="#1E1E1E",
    block_border_width="1px",
    block_label_text_color="#AAAAAA"
)

APP_CSS = """
.gradio-container {
    background: #121212 !important;
    color: #ffffff !important;
    max-width: 1180px !important;
}
.app-shell {
    gap: 0 !important;
}
.app-sidebar {
    background: #1E1E1E;
    border-right: 1px solid #2D2D2D;
    min-height: 720px;
    padding: 16px 10px;
}
.app-main {
    padding: 16px 20px;
}
.app-title {
    color: #ffffff;
    font: 700 22px/1.2 Segoe UI, sans-serif;
    letter-spacing: 0;
    margin: 0 0 14px;
}
.sidebar-title {
    color: #00FFCC;
    font: 700 16px/1.2 Segoe UI, sans-serif;
    text-align: center;
    margin: 2px 0 18px;
}
.stat-card {
    background: #1E1E1E;
    border: 1px solid #2B2B2B;
    border-radius: 6px;
    padding: 12px;
}
.stat-label {
    color: #888888;
    font: 700 10px/1.2 Segoe UI, sans-serif;
    text-transform: uppercase;
}
.stat-value {
    color: #00FFCC;
    font: 700 18px/1.3 Segoe UI, sans-serif;
    margin-top: 4px;
}
.camera-panel {
    background: #000000;
    border: 1px solid #222222;
    border-radius: 6px;
    padding: 10px;
}
.guide-table {
    background: #121212;
    border-radius: 6px;
}
.compact-note {
    color: #AAAAAA;
    font-size: 12px;
    margin-top: 12px;
}
"""

with gr.Blocks(title="SmartAir Mouse Pro - Web Sandbox") as demo:
    gr.Markdown(
        """
        # 🖐️ SmartAir Mouse Pro - Web Demo
        This sandbox demonstrates the core computer vision algorithms of the **SmartAir Mouse Pro** system.
        It runs MediaPipe Tasks hand tracking and matches posture joints against our spatial gesture classifier in real-time.
        """
    )

    with gr.Row(elem_classes=["app-shell"]):
        # ==========================================
        # LEFT PANEL: SIDEBAR SETTINGS (matches app.py)
        # ==========================================
        with gr.Column(scale=2, elem_classes=["app-sidebar"]):
            gr.HTML("<div class='sidebar-title'>SETTINGS PANEL</div>")

            camera_sel = gr.Dropdown(
                choices=["Camera 0", "Camera 1", "Camera 2"],
                value="Camera 0",
                label="Camera Source"
            )

            sensitivity_slider = gr.Slider(
                minimum=0.5, maximum=4.0, value=1.5, step=0.1,
                label="Sensitivity"
            )

            smoothing_slider = gr.Slider(
                minimum=0.05, maximum=0.8, value=0.2, step=0.01,
                label="Smoothing Factor"
            )

            trail_len_slider = gr.Slider(
                minimum=0, maximum=50, value=20, step=1,
                label="Trail Length"
            )

            trail_color_sel = gr.Dropdown(
                choices=["Cyan", "Green", "Red", "Yellow", "Blue"],
                value="Cyan",
                label="Trail Color"
            )

            glow_size_slider = gr.Slider(
                minimum=5, maximum=30, value=15, step=1,
                label="Cursor Glow Size"
            )

            pinch_sens_slider = gr.Slider(
                minimum=0.02, maximum=0.08, value=0.045, step=0.001,
                label="Pinch Sensitivity"
            )

            scroll_speed_slider = gr.Slider(
                minimum=0.2, maximum=3.0, value=1.0, step=0.1,
                label="Scroll Speed"
            )

            sound_effects = gr.Checkbox(label="Sound Effects", value=True)
            visual_particles = gr.Checkbox(label="Visual Particles", value=True)
            show_skeleton = gr.Checkbox(label="Show Screen Skeleton", value=True)
            mouse_control = gr.Checkbox(label="Mouse Control (Desktop Only)", value=False, interactive=False)

        # ==========================================
        # RIGHT PANEL: MAIN WORKSPACE (matches app.py)
        # ==========================================
        with gr.Column(scale=3, elem_classes=["app-main"]):
            gr.HTML("<h1 class='app-title'>SMARTAIR MOUSE PRO</h1>")

            # Statistics / Metrics Row
            with gr.Row():
                gesture_output = gr.Textbox(
                    label="Gesture",
                    value="NONE",
                    interactive=False
                )
                confidence_output = gr.Number(
                    label="Confidence",
                    value=0.0,
                    interactive=False
                )

            # Video feeds side-by-side panel
            with gr.Row(elem_classes=["camera-panel"]):
                webcam_input = gr.Image(
                    sources=["webcam"],
                    type="numpy",
                    label="Camera Source",
                    streaming=True
                )
                annotated_output = gr.Image(
                    type="numpy",
                    label="Tracked Hand Preview",
                    interactive=False
                )

            # Gesture guide legend below
            gr.Markdown("### 📜 Gesture Control Cheat Sheet")
            guide_data = [[name.replace("_", " "), desc] for name, desc in GESTURE_DESCRIPTIONS.items()]
            gr.DataFrame(
                headers=["Gesture", "Action"],
                value=guide_data,
                interactive=False,
                wrap=True,
                elem_classes=["guide-table"]
            )

    # Register frame process event hook. The webcam component emits change events
    # continuously when streaming=True, and this path keeps Gradio's API map stable
    # across Space runtime versions.
    webcam_input.change(
        fn=process_webcam_frame,
        inputs=[webcam_input, pinch_sens_slider],
        outputs=[annotated_output, gesture_output, confidence_output],
        queue=True,
        api_name="process_frame"
    )

    gr.Markdown(
        """
        *Note: This browser version performs hand skeleton mapping only and does not control the physical mouse.
        Launch the local Python desktop GUI client (`app.py`) to activate operating system cursor controls.*
        """
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", theme=theme, css=APP_CSS)
