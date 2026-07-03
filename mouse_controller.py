"""Mouse controller module.

Controls system cursor movement, clicking, scrolling, and dragging using pynput and pyautogui.
Implements exponential smoothing, Kalman Filter, dead zones, and adaptive sensitivity.
"""

import logging
import time
import numpy as np
import pyautogui
from pynput.mouse import Controller, Button
from typing import Tuple, Dict, Any

from constants import (
    GESTURE_MOVE, GESTURE_LEFT_CLICK, GESTURE_RIGHT_CLICK, GESTURE_DRAG,
    GESTURE_SCROLL, GESTURE_SCREENSHOT, GESTURE_PAUSE, GESTURE_LOCK
)

logger = logging.getLogger(__name__)

# Keep pyautogui's corner failsafe available as a last-resort escape hatch.
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = True


class KalmanFilter2D:
    """A standard 2D Kalman Filter for smoothing position measurements."""

    def __init__(self, dt: float = 1.0, process_noise: float = 0.05, measurement_noise: float = 0.5) -> None:
        """Initializes Kalman states and matrices.

        Args:
            dt: Time step.
            process_noise: Model covariance scale.
            measurement_noise: Measurement error covariance scale.
        """
        # State vector [x, y, vx, vy]^T
        self.x = np.zeros((4, 1), dtype=np.float32)
        # Covariance matrix
        self.P = np.eye(4, dtype=np.float32) * 10.0
        
        # State transition matrix
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * measurement_noise

    def predict(self) -> Tuple[float, float]:
        """Predicts the next state of the cursor.

        Returns:
            A tuple of (predicted_x, predicted_y).
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, z_x: float, z_y: float) -> Tuple[float, float]:
        """Updates the state with a new coordinate measurement.

        Args:
            z_x: Measured X coordinate.
            z_y: Measured Y coordinate.

        Returns:
            A tuple of (filtered_x, filtered_y).
        """
        z = np.array([[z_x], [z_y]], dtype=np.float32)
        
        # Innovation
        y = z - np.dot(self.H, self.x)
        # Innovation covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # State update
        self.x = self.x + np.dot(K, y)
        # Covariance update
        I = np.eye(4, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        return float(self.x[0, 0]), float(self.x[1, 0])

    def set_state(self, x: float, y: float) -> None:
        """Sets the state vector explicitly (e.g. on tracker initialization).

        Args:
            x: Initial X.
            y: Initial Y.
        """
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1.0


class MouseController:
    """Translates hands coordinates to real-time OS mouse events."""

    def __init__(self, settings: Any) -> None:
        """Initializes pynput drivers and smoothing models.

        Args:
            settings: The application settings wrapper.
        """
        self.settings = settings
        self.mouse = Controller()
        self.enabled = True
        
        # Primary cursor screen size details
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Filter initialization
        self.kalman = KalmanFilter2D(dt=1.0)
        self.last_x, self.last_y = self.mouse.position
        self.smooth_x, self.smooth_y = self.last_x, self.last_y
        self.kalman.set_state(self.last_x, self.last_y)
        
        # State variables
        self.is_dragging = False
        self.scroll_anchor_y = 0.0
        self.is_scrolling = False
        self.last_stats_update = time.time()
        
        # Session statistics (reset on app start)
        self.stats = {
            "distance_moved": 0.0,
            "clicks": 0,
            "right_clicks": 0,
            "scroll_count": 0,
            "accuracy": 0.98  # Default tracking success confidence ratio
        }

    def reset_stats(self) -> None:
        """Resets tracked metrics."""
        self.stats = {
            "distance_moved": 0.0,
            "clicks": 0,
            "right_clicks": 0,
            "scroll_count": 0,
            "accuracy": 0.98
        }

    def set_enabled(self, enabled: bool) -> None:
        """Enables or disables physical OS pointer injection.

        Args:
            enabled: If False, movement, clicks, drags, and scrolls are ignored.
        """
        self.enabled = enabled
        if not enabled:
            self._release_drag_if_active()

    def emergency_stop(self) -> None:
        """Immediately disables pointer injection and releases any held button."""
        self.set_enabled(False)
        logger.warning("Emergency stop activated. Mouse injection disabled.")

    def preview_cursor_position(self, landmarks: list) -> Tuple[int, int]:
        """Updates the smoothed virtual cursor position without OS injection.

        Args:
            landmarks: Raw normalized landmarks list.

        Returns:
            The virtual cursor coordinate (x, y).
        """
        control_pt = landmarks[8]
        screen_x, screen_y = self._map_to_screen(control_pt[0], control_pt[1])
        self.smooth_x, self.smooth_y = self._apply_filters(screen_x, screen_y)
        return int(self.smooth_x), int(self.smooth_y)

    def process_mouse_action(self, gesture: str, landmarks: list, width: int, height: int) -> Tuple[int, int]:
        """Main dispatcher translating gestures to screen events.

        Args:
            gesture: The current classified gesture.
            landmarks: Raw normalized landmarks list.
            width: Camera frame width.
            height: Camera frame height.

        Returns:
            The final smoothed screen coordinate (x, y) of the cursor.
        """
        if not self.enabled:
            return int(self.smooth_x), int(self.smooth_y)

        if gesture == GESTURE_LOCK:
            # Cursor is locked, do not update movement or register actions
            return int(self.smooth_x), int(self.smooth_y)
            
        # Get control joint position (usually Index finger tip, landmark #8)
        # Landmark indices are mapped in constants.py
        # index 8: INDEX_FINGER_TIP
        control_pt = landmarks[8]
        
        # Screen mapping with edge compensation
        screen_x, screen_y = self._map_to_screen(control_pt[0], control_pt[1])
        
        # Smooth position using Kalman filter and Exponential Moving Average
        self.smooth_x, self.smooth_y = self._apply_filters(screen_x, screen_y)
        
        # Perform action based on gesture
        if gesture == GESTURE_MOVE:
            self._release_drag_if_active()
            self.is_scrolling = False
            self._move_cursor(self.smooth_x, self.smooth_y)
            
        elif gesture == GESTURE_LEFT_CLICK:
            self._release_drag_if_active()
            self.is_scrolling = False
            self._move_cursor(self.smooth_x, self.smooth_y)
            self._trigger_left_click()
            
        elif gesture == GESTURE_RIGHT_CLICK:
            self._release_drag_if_active()
            self.is_scrolling = False
            self._move_cursor(self.smooth_x, self.smooth_y)
            self._trigger_right_click()
            
        elif gesture == GESTURE_DRAG:
            self.is_scrolling = False
            self._trigger_drag(self.smooth_x, self.smooth_y)
            
        elif gesture == GESTURE_SCROLL:
            self._release_drag_if_active()
            # Use Index tip current vs anchor displacement to perform scrolling
            self._trigger_scroll(control_pt[1])
            
        return int(self.smooth_x), int(self.smooth_y)

    def _map_to_screen(self, hx: float, hy: float) -> Tuple[float, float]:
        """Translates normalized coordinates from inner active region to full screen.

        Args:
            hx: Normalized hand X.
            hy: Normalized hand Y.

        Returns:
            Mapped screen coordinates.
        """
        # Read reach boundary boxes from settings (calibrated coordinates)
        reach_xmin = self.settings.get("reach_xmin")
        reach_xmax = self.settings.get("reach_xmax")
        reach_ymin = self.settings.get("reach_ymin")
        reach_ymax = self.settings.get("reach_ymax")
        
        # Normalize mapping inside the sub-rectangle
        mapped_x = (hx - reach_xmin) / (reach_xmax - reach_xmin)
        mapped_y = (hy - reach_ymin) / (reach_ymax - reach_ymin)
        
        # Clamp between 0 and 1
        mapped_x = np.clip(mapped_x, 0.0, 1.0)
        mapped_y = np.clip(mapped_y, 0.0, 1.0)
        
        # Scale to screen pixels
        screen_x = mapped_x * self.screen_w
        screen_y = mapped_y * self.screen_h
        
        return screen_x, screen_y

    def _apply_filters(self, x: float, y: float) -> Tuple[float, float]:
        """Filters cursor tremors using Kalman predictions & exponential smoothing.

        Args:
            x: Raw mapped X.
            y: Raw mapped Y.

        Returns:
            Smoothed (x, y).
        """
        # Kalman filter update
        self.kalman.predict()
        kf_x, kf_y = self.kalman.update(x, y)
        
        # Exponential Moving Average (EMA)
        alpha = self.settings.get("smoothing_factor")
        
        smoothed_x = alpha * kf_x + (1.0 - alpha) * self.smooth_x
        smoothed_y = alpha * kf_y + (1.0 - alpha) * self.smooth_y
        
        # Dead-zone threshold to ignore micro-shakes when still
        dx = smoothed_x - self.smooth_x
        dy = smoothed_y - self.smooth_y
        displacement = np.sqrt(dx*dx + dy*dy)
        
        dead_zone = 2.0  # Pixels
        if displacement < dead_zone:
            return self.smooth_x, self.smooth_y
            
        # Adaptive sensitivity and acceleration
        sensitivity_x = self.settings.get("sensitivity_x")
        sensitivity_y = self.settings.get("sensitivity_y")
        
        # Simple velocity-dependent multiplier
        acceleration_factor = 1.0
        if displacement > 15.0:
            acceleration_factor = 1.0 + (displacement - 15.0) * 0.04
            
        final_x = self.smooth_x + dx * sensitivity_x * acceleration_factor
        final_y = self.smooth_y + dy * sensitivity_y * acceleration_factor
        
        # Keep inside screen borders
        final_x = np.clip(final_x, 0.0, self.screen_w - 1.0)
        final_y = np.clip(final_y, 0.0, self.screen_h - 1.0)
        
        return final_x, final_y

    def _move_cursor(self, x: float, y: float) -> None:
        """Physically repositions the system pointer and tracks stats."""
        # Calculate moving distance for stats
        dx = x - self.last_x
        dy = y - self.last_y
        dist = np.sqrt(dx * dx + dy * dy)
        self.stats["distance_moved"] += dist
        
        self.mouse.position = (int(x), int(y))
        self.last_x, self.last_y = x, y

    def _trigger_left_click(self) -> None:
        """Performs left click action."""
        self.mouse.click(Button.left)
        self.stats["clicks"] += 1
        logger.debug("System Action: Left Click")
        # Throttle click events to prevent multi-firing
        time.sleep(0.05)

    def _trigger_right_click(self) -> None:
        """Performs right click action."""
        self.mouse.click(Button.right)
        self.stats["right_clicks"] += 1
        logger.debug("System Action: Right Click")
        time.sleep(0.05)

    def _trigger_drag(self, x: float, y: float) -> None:
        """Holds down the left mouse button and drags the pointer.

        Args:
            x: Destination X.
            y: Destination Y.
        """
        if not self.is_dragging:
            self.mouse.press(Button.left)
            self.is_dragging = True
            logger.debug("System Action: Drag Hold Started")
            
        self._move_cursor(x, y)

    def _release_drag_if_active(self) -> None:
        """Releases the left click if currently dragging."""
        if self.is_dragging:
            self.mouse.release(Button.left)
            self.is_dragging = False
            logger.debug("System Action: Drag Released")

    def _trigger_scroll(self, curr_y: float) -> None:
        """Calculates vertical displacement and scrolls.

        Args:
            curr_y: Current normalized Y joint coordinate.
        """
        if not self.is_scrolling:
            self.scroll_anchor_y = curr_y
            self.is_scrolling = True
            return

        dy = curr_y - self.scroll_anchor_y
        scroll_speed = self.settings.get("scroll_speed")
        
        # Scroll threshold: must move finger beyond min threshold to scroll
        scroll_threshold = 0.02
        if abs(dy) > scroll_threshold:
            # Scale scroll steps
            scroll_amount = int(np.sign(dy) * -3 * scroll_speed)
            if scroll_amount != 0:
                self.mouse.scroll(0, scroll_amount)
                self.stats["scroll_count"] += abs(scroll_amount)
                # Reset anchor to track relative changes incrementally
                self.scroll_anchor_y = curr_y
