"""Hand tracking module using MediaPipe.

Acquires camera frames, automatically detects active video inputs, runs hand landmarks detection,
and processes statistics like tracker FPS, confidence, and handedness.
"""

import cv2
import mediapipe as mp
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

# Try standard solutions imports
try:
    from mediapipe.solutions import hands as mp_hands
    from mediapipe.solutions import drawing_utils as mp_draw
    from mediapipe.solutions import drawing_styles as mp_draw_styles
except ImportError:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_draw
    import mediapipe.python.solutions.drawing_styles as mp_draw_styles

logger = logging.getLogger(__name__)

class HandTracker:
    """Manages OpenCV Video Capture and MediaPipe Hand Tracking execution."""

    def __init__(self,
                 camera_index: int = 0,
                 max_num_hands: int = 1,
                 detection_confidence: float = 0.75,
                 tracking_confidence: float = 0.75) -> None:
        """Initializes the hand tracker and media resources.

        Args:
            camera_index: Index of camera to try first.
            max_num_hands: Maximum number of hands to track simultaneously.
            detection_confidence: Minimum detection confidence threshold.
            tracking_confidence: Minimum tracking confidence threshold.
        """
        self.camera_index = camera_index
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.prev_time = time.time()
        self.fps = 0.0
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp_draw
        self.mp_draw_styles = mp_draw_styles

    def auto_select_camera(self) -> int:
        """Iterates over common camera indexes to find an active video input.

        Returns:
            The index of the selected working camera, or -1 if none found.
        """
        # Try index 0 up to 3
        test_indices = [self.camera_index, 0, 1, 2]
        # Remove duplicates while keeping order
        test_indices = list(dict.fromkeys(test_indices))
        
        for index in test_indices:
            logger.info("Testing camera interface at index %d...", index)
            # On Windows, cv2.CAP_DSHOW is generally faster to initialize and avoids logs
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    logger.info("Found working camera at index %d", index)
                    cap.release()
                    return index
                cap.release()
                
        logger.warning("No camera index successfully opened. Defaulting to index 0.")
        return 0

    def start_camera(self) -> bool:
        """Attempts to open the video capture stream.

        Returns:
            True if camera started successfully, False otherwise.
        """
        self.stop_camera()
        
        # Verify and select working index
        active_index = self.auto_select_camera()
        self.camera_index = active_index
        
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logger.error("Failed to open camera index %d", self.camera_index)
            return False
            
        # Optimize camera settings for 720p 60FPS if supported
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Enable buffer optimization
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info("Camera started at index %d. Resolution: %dx%d", self.camera_index, int(actual_w), int(actual_h))
        return True

    def stop_camera(self) -> None:
        """Releases the camera video stream resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera capture released.")

    def update_fps(self) -> None:
        """Calculates tracking frames-per-second."""
        current_time = time.time()
        elapsed = current_time - self.prev_time
        if elapsed > 0:
            current_fps = 1.0 / elapsed
            # Simple running average to smooth the FPS display
            self.fps = self.fps * 0.9 + current_fps * 0.1
        self.prev_time = current_time

    def process_frame(self) -> Tuple[Optional[np_image_raw := Any], Dict[str, Any]]:
        """Reads a frame, flips it, runs MediaPipe, and extracts tracking data.

        Returns:
            A tuple of (processed_cv2_frame, tracking_metadata_dict).
        """
        if self.cap is None or not self.cap.isOpened():
            return None, {"active": False, "reason": "Camera not started"}

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, {"active": False, "reason": "No frame read"}

        # Calculate tracker execution FPS
        self.update_fps()

        # Flip the frame horizontally for mirror-view cursor navigation
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # MediaPipe requires RGB format, avoid unnecessary copies where possible
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Performance optimization: pass by reference
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        hand_data: Optional[Dict[str, Any]] = None

        if results.multi_hand_landmarks and results.multi_handedness:
            # We process the first detected hand for cursor control
            landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0]
            
            label = handedness.label  # "Left" or "Right"
            score = handedness.score  # Confidence score

            landmark_list: List[Tuple[float, float, float]] = []
            for lm in landmarks.landmark:
                # Store normalized coordinates
                landmark_list.append((lm.x, lm.y, lm.z))

            hand_data = {
                "landmarks": landmark_list,
                "handedness": label,
                "confidence": score,
                "raw_landmarks": landmarks,  # Original MediaPipe landmarks object
                "bounding_box": self._get_bounding_box(landmark_list, width, height)
            }

        metadata = {
            "active": True,
            "fps": self.fps,
            "width": width,
            "height": height,
            "hand": hand_data
        }

        return frame, metadata

    def _get_bounding_box(self, landmarks: List[Tuple[float, float, float]], width: int, height: int) -> Tuple[int, int, int, int]:
        """Calculates pixel-space bounding box enclosing all hand landmarks.

        Args:
            landmarks: Normalized hand landmarks.
            width: Image width.
            height: Image height.

        Returns:
            Bounding box coordinates (xmin, ymin, xmax, ymax).
        """
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        
        xmin = int(min(x_coords) * width)
        xmax = int(max(x_coords) * width)
        ymin = int(min(y_coords) * height)
        ymax = int(max(y_coords) * height)
        
        # Add a light padding margin around the bounding box
        padding = 15
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(width, xmax + padding)
        ymax = min(height, ymax + padding)
        
        return xmin, ymin, xmax, ymax

    def draw_landmarks(self, frame: Any, hand_metadata: Dict[str, Any]) -> None:
        """Draws skeleton lines and landmark points on the frame.

        Args:
            frame: OpenCV image frame.
            hand_metadata: Dictionary containing landmark details.
        """
        if not hand_metadata or "raw_landmarks" not in hand_metadata:
            return
        
        raw_lm = hand_metadata["raw_landmarks"]
        self.mp_draw.draw_landmarks(
            frame,
            raw_lm,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw_styles.get_default_hand_landmarks_style(),
            self.mp_draw_styles.get_default_hand_connections_style()
        )

    def close(self) -> None:
        """Releases camera and media resources."""
        self.stop_camera()
        self.hands.close()
        logger.info("HandTracker engine shutdown complete.")
