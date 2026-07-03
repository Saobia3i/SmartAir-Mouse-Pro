"""Gradio Web Application for SmartAir Mouse Pro.

Runs a browser-based demo showcasing real-time camera capture, MediaPipe Tasks tracking,
gesture classification overlays, and confidence metrics directly in the web browser.
"""

import os
import sys
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
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(model_path, "wb") as out_file:
            out_file.write(response.read())
    except Exception as e:
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


def process_webcam_frame(frame: np.ndarray) -> Tuple[np.ndarray, str, float]:
    """Processes a frame from the browser webcam, draws tracking data, and predicts gestures.

    Args:
        frame: Input RGB image from webcam (numpy array).

    Returns:
        A tuple of (annotated_rgb_frame, gesture_name, confidence).
    """
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8), "No Frame", 0.0

    # Flip horizontally for mirrored view
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # MediaPipe process (Gradio input frames are already in RGB format)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
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


# ==========================================
# GRADIO UI BUILDER
# ==========================================
theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#121212",
    body_text_color="#FFFFFF",
    block_background_fill="#1E1E1E",
    block_border_width="1px",
    block_label_text_color="#AAAAAA"
)

with gr.Blocks(title="SmartAir Mouse Pro - Web Sandbox") as demo:
    gr.Markdown(
        """
        # 🖐️ SmartAir Mouse Pro - Web Demo
        This sandbox demonstrates the core computer vision algorithms of the **SmartAir Mouse Pro** system.
        It runs MediaPipe Tasks hand tracking and matches posture joints against our spatial gesture classifier in real-time.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            # Input webcam node
            webcam_input = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="Active Web Camera Stream",
                streaming=True
            )
            annotated_output = gr.Image(
                type="numpy",
                label="Tracked Hand Preview",
                interactive=False
            )
            
        with gr.Column(scale=2):
            # Text metrics
            gesture_output = gr.Textbox(
                label="Recognized Gesture State",
                value="NONE",
                interactive=False
            )
            confidence_output = gr.Number(
                label="Classification Confidence",
                value=0.0,
                interactive=False
            )
            
            # Instructions table
            gr.Markdown("### 📜 Gesture Control Cheat Sheet")
            
            guide_data = [[name, desc] for name, desc in GESTURE_DESCRIPTIONS.items()]
            gr.DataFrame(
                headers=["Gesture", "Action Triggered"],
                value=guide_data,
                interactive=False,
                wrap=True
            )

    # Register frame process event hook
    webcam_input.stream(
        fn=process_webcam_frame,
        inputs=[webcam_input],
        outputs=[annotated_output, gesture_output, confidence_output],
        queue=True
    )
    
    gr.Markdown(
        """
        *Note: This browser version performs hand skeleton mapping only and does not control the physical mouse.
        Launch the local Python desktop GUI client (`app.py`) to activate operating system cursor controls.*
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", theme=theme)
