"""Hand tracking module using MediaPipe Tasks API.

Supports the modern MediaPipe HandLandmarker API. Automatically downloads the required
model file, flips feeds, tracks landmarks, and computes frame rates.
"""

import cv2
import time
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)

class HandTracker:
    """Manages OpenCV Video Capture and MediaPipe Tasks Hand Tracking execution."""

    def __init__(self,
                 camera_index: int = 0,
                 max_num_hands: int = 1,
                 detection_confidence: float = 0.75,
                 tracking_confidence: float = 0.75) -> None:
        """Initializes the hand landmarker tasks and model file.

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
        self.start_time = time.time()
        self.frame_timestamp_ms = 0
        
        # Define model target path
        self.assets_dir = Path("./assets").resolve()
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.assets_dir / "hand_landmarker.task"
        
        # Verify model download
        self._ensure_model_downloaded()
        
        # Initialize MediaPipe Tasks Hand Landmarker
        logger.info("Initializing MediaPipe HandLandmarker Task...")
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.detection_confidence,
            min_hand_presence_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        logger.info("MediaPipe HandLandmarker Task initialized successfully.")

    def _ensure_model_downloaded(self) -> None:
        """Downloads the official Google MediaPipe hand landmarker task file if missing."""
        if not self.model_path.exists():
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            logger.info("Downloading hand landmarker task file from %s...", url)
            try:
                # Add headers to avoid HTTP blocks
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(req, timeout=30) as response, open(self.model_path, "wb") as out_file:
                    out_file.write(response.read())
                logger.info("Model download complete. Saved to %s", self.model_path)
            except Exception as e:
                logger.critical("Failed to download MediaPipe model file. App cannot start. Error: %s", e)
                raise RuntimeError(f"Could not download hand_landmarker.task: {e}")

    def auto_select_camera(self) -> int:
        """Iterates over common camera indexes to find an active video input.

        Returns:
            The index of the selected working camera, or -1 if none found.
        """
        test_indices = [self.camera_index, 0, 1, 2]
        test_indices = list(dict.fromkeys(test_indices))
        
        for index in test_indices:
            logger.info("Testing camera interface at index %d...", index)
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
        
        active_index = self.auto_select_camera()
        self.camera_index = active_index
        
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logger.error("Failed to open camera index %d", self.camera_index)
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.start_time = time.time()
        self.frame_timestamp_ms = 0
        
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
            self.fps = self.fps * 0.9 + current_fps * 0.1
        self.prev_time = current_time

    def process_frame(self) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Reads a frame, flips it, runs MediaPipe Tasks, and extracts landmarks.

        Returns:
            A tuple of (processed_cv2_frame, tracking_metadata_dict).
        """
        if self.cap is None or not self.cap.isOpened():
            return None, {"active": False, "reason": "Camera not started"}

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, {"active": False, "reason": "No frame read"}

        self.update_fps()
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Tasks API takes MediaPipe Image object format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process detection. VIDEO mode keeps tracking state between frames and is much
        # faster for live webcam streams than independent IMAGE detections.
        self.frame_timestamp_ms += 33
        timestamp_ms = self.frame_timestamp_ms
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        hand_data: Optional[Dict[str, Any]] = None

        if results.hand_landmarks and results.handedness:
            landmarks = results.hand_landmarks[0]
            handedness = results.handedness[0][0]
            
            label = handedness.category_name
            score = handedness.score

            landmark_list: List[Tuple[float, float, float]] = []
            for lm in landmarks:
                landmark_list.append((lm.x, lm.y, lm.z))

            hand_data = {
                "landmarks": landmark_list,
                "handedness": label,
                "confidence": score,
                "raw_landmarks": landmarks,  # List of NormalizedLandmark objects
                "bounding_box": self._get_bounding_box(landmark_list, width, height)
            }
            self.draw_landmarks(frame, hand_data)

        metadata = {
            "active": True,
            "fps": self.fps,
            "width": width,
            "height": height,
            "hand": hand_data
        }

        return frame, metadata

    def _get_bounding_box(self, landmarks: List[Tuple[float, float, float]], width: int, height: int) -> Tuple[int, int, int, int]:
        """Calculates bounding box around hand landmarks.

        Args:
            landmarks: Normalized landmarks list.
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
        
        padding = 15
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(width, xmax + padding)
        ymax = min(height, ymax + padding)
        
        return xmin, ymin, xmax, ymax

    def draw_landmarks(self, frame: Any, hand_metadata: Dict[str, Any]) -> None:
        """Draws skeleton lines manually using OpenCV (eliminates drawing_utils dependencies).

        Args:
            frame: OpenCV frame.
            hand_metadata: Dict containing hand statistics.
        """
        if not hand_metadata or "landmarks" not in hand_metadata:
            return
        
        landmarks = hand_metadata["landmarks"]
        h, w, _ = frame.shape
        
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
            p1 = (int(landmarks[start][0] * w), int(landmarks[start][1] * h))
            p2 = (int(landmarks[end][0] * w), int(landmarks[end][1] * h))
            cv2.line(frame, p1, p2, (57, 255, 20), 2)  # Neon Green
            
        # Draw joint nodes
        for pt in landmarks:
            p = (int(pt[0] * w), int(pt[1] * h))
            cv2.circle(frame, p, 4, (0, 240, 255), -1)  # Cyan

    def close(self) -> None:
        """Releases detector and camera resources."""
        self.stop_camera()
        self.detector.close()
        logger.info("HandTracker engine shutdown complete.")
