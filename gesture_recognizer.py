"""Gesture recognition module for SmartAir Mouse Pro.

Processes normalized landmarks to detect open palm, closed fist, pinch events,
drags, scrolls, screenshots, and movements, utilizing debounce, hysteresis, and cooldowns.
"""

import time
import logging
from typing import Dict, Any, List, Tuple

from constants import (
    WRIST, THUMB_TIP, INDEX_FINGER_TIP, INDEX_FINGER_PIP, INDEX_FINGER_MCP,
    MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_MCP,
    RING_FINGER_TIP, RING_FINGER_PIP, PINKY_TIP, PINKY_PIP,
    GESTURE_MOVE, GESTURE_LEFT_CLICK, GESTURE_RIGHT_CLICK, GESTURE_DRAG,
    GESTURE_SCROLL, GESTURE_SCREENSHOT, GESTURE_PAUSE, GESTURE_LOCK,
    GESTURE_NONE, DEBOUNCE_FRAMES, COOLDOWN_SCREENSHOT, COOLDOWN_CLICK
)
from utils import calculate_distance

logger = logging.getLogger(__name__)

class GestureRecognizer:
    """Classifies hand landmarks into interactive mouse gestures with filter stabilization."""

    def __init__(self, click_threshold: float = 0.04) -> None:
        """Initializes gesture state variables.

        Args:
            click_threshold: Initial normalized distance for detecting pinches.
        """
        self.click_threshold = click_threshold
        # Hysteresis parameters: lower to enter pinch, higher to release
        self.pinch_enter_threshold = self.click_threshold
        self.pinch_exit_threshold = self.click_threshold * 1.3
        
        # Debouncing history
        self.history: List[str] = []
        self.current_gesture = GESTURE_NONE
        
        # Cooldowns
        self.last_screenshot_time = 0.0
        self.last_click_time = 0.0
        
        # State tracking for drag/scroll
        self.prev_pinch_states = {
            "left": False,
            "right": False,
            "drag": False
        }

    def update_thresholds(self, baseline_threshold: float) -> None:
        """Dynamically adjusts pinch thresholds based on calibration or settings.

        Args:
            baseline_threshold: Base normalized distance.
        """
        self.click_threshold = baseline_threshold
        self.pinch_enter_threshold = baseline_threshold
        self.pinch_exit_threshold = baseline_threshold * 1.35
        logger.debug("Pinch thresholds updated: Enter=%0.3f, Exit=%0.3f", 
                     self.pinch_enter_threshold, self.pinch_exit_threshold)

    def recognize(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[str, float]:
        """Analyzes landmark distances and angles to classify a gesture.

        Args:
            landmarks: Normalized landmarks list (21 landmarks).

        Returns:
            A tuple of (gesture_name, confidence_score).
        """
        if not landmarks or len(landmarks) < 21:
            return GESTURE_NONE, 0.0

        # 1. Gather finger extension states (Y-coordinate based, smaller Y = higher on screen)
        index_ext = landmarks[INDEX_FINGER_TIP][1] < landmarks[INDEX_FINGER_PIP][1]
        middle_ext = landmarks[MIDDLE_FINGER_TIP][1] < landmarks[MIDDLE_FINGER_PIP][1]
        ring_ext = landmarks[RING_FINGER_TIP][1] < landmarks[RING_FINGER_PIP][1]
        pinky_ext = landmarks[PINKY_TIP][1] < landmarks[PINKY_PIP][1]
        
        # Thumb extension: check distance from thumb tip to index finger MCP
        thumb_index_mcp_dist = calculate_distance(landmarks[THUMB_TIP], landmarks[INDEX_FINGER_MCP])
        thumb_ext = thumb_index_mcp_dist > 0.14

        # 2. Compute key tip-to-thumb normalized distances
        thumb_tip_coords = landmarks[THUMB_TIP]
        index_tip_coords = landmarks[INDEX_FINGER_TIP]
        middle_tip_coords = landmarks[MIDDLE_FINGER_TIP]
        
        dist_thumb_index = calculate_distance(thumb_tip_coords, index_tip_coords)
        dist_thumb_middle = calculate_distance(thumb_tip_coords, middle_tip_coords)

        # 3. Classify raw gesture candidates
        raw_gesture = GESTURE_MOVE
        confidence = 1.0

        # PAUSE: Open palm (all fingers extended)
        if index_ext and middle_ext and ring_ext and pinky_ext and thumb_ext:
            raw_gesture = GESTURE_PAUSE
            confidence = 0.95
            
        # LOCK: Closed fist (all fingers folded)
        elif not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            raw_gesture = GESTURE_LOCK
            confidence = 0.95

        # SCREENSHOT: Three fingers extended (Index, Middle, Ring), pinky folded
        elif index_ext and middle_ext and ring_ext and not pinky_ext and not thumb_ext:
            raw_gesture = GESTURE_SCREENSHOT
            confidence = 0.90

        # SCROLL: Two fingers extended (Index, Middle), ring and pinky folded
        elif index_ext and middle_ext and not ring_ext and not pinky_ext:
            # Check pinch override
            is_index_pinched = dist_thumb_index < self._get_pinch_threshold("drag", dist_thumb_index)
            is_middle_pinched = dist_thumb_middle < self._get_pinch_threshold("drag", dist_thumb_middle)
            
            if is_index_pinched and is_middle_pinched:
                raw_gesture = GESTURE_DRAG
                confidence = 0.85
            else:
                raw_gesture = GESTURE_SCROLL
                confidence = 0.90

        # PINCH DETECTIONS: Left Click (Thumb + Index), Right Click (Thumb + Middle)
        else:
            # Determine threshold with hysteresis
            idx_thresh = self._get_pinch_threshold("left", dist_thumb_index)
            mid_thresh = self._get_pinch_threshold("right", dist_thumb_middle)
            
            is_index_pinched = dist_thumb_index < idx_thresh
            is_middle_pinched = dist_thumb_middle < mid_thresh

            if is_index_pinched and is_middle_pinched:
                raw_gesture = GESTURE_DRAG
                confidence = 0.85
            elif is_index_pinched:
                raw_gesture = GESTURE_LEFT_CLICK
                # Calculate simple inverse distance-based confidence score
                confidence = max(0.5, 1.0 - (dist_thumb_index / self.pinch_exit_threshold))
            elif is_middle_pinched:
                raw_gesture = GESTURE_RIGHT_CLICK
                confidence = max(0.5, 1.0 - (dist_thumb_middle / self.pinch_exit_threshold))
            elif index_ext:
                raw_gesture = GESTURE_MOVE
                confidence = 0.90
            else:
                raw_gesture = GESTURE_NONE
                confidence = 0.0

        # 4. Debounce and filter state transitions
        stabilized_gesture = self._debounce_gesture(raw_gesture)
        
        # 5. Apply Cooldowns to transient actions (Screenshot, Click triggers)
        now = time.time()
        if stabilized_gesture == GESTURE_SCREENSHOT:
            if now - self.last_screenshot_time < COOLDOWN_SCREENSHOT:
                stabilized_gesture = GESTURE_SCROLL  # Fallback during cooldown
            else:
                self.last_screenshot_time = now
                logger.info("Screenshot gesture confirmed.")
                
        elif stabilized_gesture in (GESTURE_LEFT_CLICK, GESTURE_RIGHT_CLICK):
            if now - self.last_click_time < COOLDOWN_CLICK:
                # Do not emit click event, fall back to move
                stabilized_gesture = GESTURE_MOVE
            else:
                # We update last click time when the click state is first entered
                if not self.prev_pinch_states["left"] and stabilized_gesture == GESTURE_LEFT_CLICK:
                    self.last_click_time = now
                elif not self.prev_pinch_states["right"] and stabilized_gesture == GESTURE_RIGHT_CLICK:
                    self.last_click_time = now

        # Update previous states for hysteresis reference
        self.prev_pinch_states["left"] = (stabilized_gesture == GESTURE_LEFT_CLICK)
        self.prev_pinch_states["right"] = (stabilized_gesture == GESTURE_RIGHT_CLICK)
        self.prev_pinch_states["drag"] = (stabilized_gesture == GESTURE_DRAG)

        return stabilized_gesture, confidence

    def _get_pinch_threshold(self, name: str, current_dist: float) -> float:
        """Implements hysteresis by checking current state to decide threshold.

        Args:
            name: ID of the state ("left", "right", or "drag").
            current_dist: Calculated distance.

        Returns:
            The threshold boundary.
        """
        # If already pinched, use the larger release threshold to prevent flicker
        if self.prev_pinch_states.get(name, False):
            return self.pinch_exit_threshold
        return self.pinch_enter_threshold

    def _debounce_gesture(self, new_raw_gesture: str) -> str:
        """Appends raw gesture to buffer and stabilizes rapid flickering.

        Args:
            new_raw_gesture: The newly predicted raw gesture.

        Returns:
            The stabilized gesture name.
        """
        self.history.append(new_raw_gesture)
        if len(self.history) > DEBOUNCE_FRAMES:
            self.history.pop(0)

        # Count occurrences in history
        counts = {}
        for g in self.history:
            counts[g] = counts.get(g, 0) + 1

        # Find the most frequent gesture
        most_frequent = max(counts, key=counts.get)
        
        # Require a clear majority to switch states
        if counts[most_frequent] >= DEBOUNCE_FRAMES - 1:
            self.current_gesture = most_frequent
            
        return self.current_gesture
