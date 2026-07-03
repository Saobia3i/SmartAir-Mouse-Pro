"""Gradio Web Application for SmartAir Mouse Pro.

Runs a browser-based demo showcasing real-time camera capture, MediaPipe tracking,
gesture classification overlays, and confidence metrics directly in the web browser.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import gradio as gr
from typing import Tuple, Dict, Any

# Ensure workspace paths resolved
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from constants import GESTURE_DESCRIPTIONS
from gesture_recognizer import GestureRecognizer

# Initialize MediaPipe Hands processor
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

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
    results = hands.process(frame)

    gesture_name = "NONE"
    confidence = 0.0

    if results.multi_hand_landmarks and results.multi_handedness:
        landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0]
        
        # Extract normalized coordinates
        landmark_list = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
        
        # Recognize gesture
        gesture_name, confidence = recognizer.recognize(landmark_list)
        
        # Draw skeleton connections
        mp_draw.draw_landmarks(
            frame,
            landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_draw_styles.get_default_hand_landmarks_style(),
            mp_draw_styles.get_default_hand_connections_style()
        )
        
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
        box_color = (0, 102, 255) if handedness.label == "Left" else (57, 255, 20)  # BGR
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
            f"Hand: {handedness.label}",
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

with gr.Blocks(theme=theme, title="SmartAir Mouse Pro - Web Sandbox") as demo:
    gr.Markdown(
        """
        # 🖐️ SmartAir Mouse Pro - Web Demo
        This sandbox demonstrates the core computer vision algorithms of the **SmartAir Mouse Pro** system.
        It runs MediaPipe hand tracking and matches posture joints against our spatial gesture classifier in real-time.
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
        outputs=[webcam_input, gesture_output, confidence_output],
        queue=True
    )
    
    gr.Markdown(
        """
        *Note: This browser version performs hand skeleton mapping only and does not control the physical mouse.
        Launch the local Python desktop GUI client (`app.py`) to activate operating system cursor controls.*
        """
    )

if __name__ == "__main__":
    logger_init = gr.logging.get_logger(__name__)
    demo.launch(server_name="127.0.0.1", server_port=7860)
