"""Calibration Wizard module for SmartAir Mouse Pro.

Provides a guided CustomTkinter wizard dialog to measure reach limits,
pinch clicking thresholds, and hand sizes, saving results to settings.json.
"""

import tkinter as tk
import customtkinter as ctk
import time
import logging
from typing import Any, Dict, List, Tuple, Optional

from constants import (
    WRIST, THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, GESTURE_NONE
)
from utils import calculate_distance

logger = logging.getLogger(__name__)


class CalibrationWizard(ctk.CTkToplevel):
    """Step-by-step desktop dialog guiding users through coordinate mapping calibration."""

    def __init__(self, parent: Any, settings: Any, on_complete_callback: Any) -> None:
        """Initializes the wizard dialog.

        Args:
            parent: Parent app container window.
            settings: Settings manager instance.
            on_complete_callback: Callable to trigger after completing.
        """
        super().__init__(parent)
        self.settings = settings
        self.on_complete = on_complete_callback
        
        self.title("Calibration Wizard")
        self.geometry("500x400")
        self.resizable(False, False)
        
        # Ensure modal behavior
        self.transient(parent)
        self.grab_set()
        
        # Wizard state variables
        self.step = 0
        self.is_running = True
        self.calibrating_data = False
        self.countdown_remaining = 0
        self.last_countdown_time = 0.0
        
        # Temporary calibration metrics
        self.measured_xs: List[float] = []
        self.measured_ys: List[float] = []
        self.measured_pinches: List[float] = []
        self.measured_sizes: List[float] = []
        
        # Setup modern layout
        self.configure(fg_color="#121212")
        self._setup_ui()
        self._show_step_content()
        
        # Focus window
        self.lift()
        self.focus_force()
        
        # Handle close window
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self) -> None:
        """Constructs the CTk control widgets."""
        # Top banner
        self.title_label = ctk.CTkLabel(
            self,
            text="SmartAir Mouse Calibration",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#00FFCC"
        )
        self.title_label.pack(pady=(25, 10))
        
        # Description text area
        self.desc_frame = ctk.CTkFrame(self, fg_color="#1E1E1E", width=420, height=180)
        self.desc_frame.pack_propagate(False)
        self.desc_frame.pack(pady=15)
        
        self.desc_label = ctk.CTkLabel(
            self.desc_frame,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color="#FFFFFF",
            justify=tk.LEFT,
            wraplength=380
        )
        self.desc_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Visual countdown indicator or status
        self.status_label = ctk.CTkLabel(
            self,
            text="Ready",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color="#FFF01F"
        )
        self.status_label.pack(pady=5)
        
        # Bottom controls buttons
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        
        self.cancel_button = ctk.CTkButton(
            self.btn_frame,
            text="Cancel",
            fg_color="#333333",
            hover_color="#444444",
            text_color="#FFFFFF",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            width=100,
            command=self._on_close
        )
        self.cancel_button.pack(side=tk.LEFT, padx=(40, 0))
        
        self.action_button = ctk.CTkButton(
            self.btn_frame,
            text="Next",
            fg_color="#0066FF",
            hover_color="#0052CC",
            text_color="#FFFFFF",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            width=140,
            command=self._on_action_clicked
        )
        self.action_button.pack(side=tk.RIGHT, padx=(0, 40))

    def _show_step_content(self) -> None:
        """Updates text display instructions based on current step index."""
        if self.step == 0:
            self.desc_label.configure(
                text="Welcome to the Calibration Wizard!\n\n"
                     "This process will adjust SmartAir Mouse settings to match your "
                     "camera view, arm reach, and finger size.\n\n"
                     "Please sit comfortably in front of your camera.\n"
                     "Ensure your hand is fully visible in the frame."
            )
            self.action_button.configure(text="Start Calibration")
            self.status_label.configure(text="")
            
        elif self.step == 1:
            self.desc_label.configure(
                text="Step 1: Calibrate Comfortable Reach Range\n\n"
                     "When you click 'Record Reach', move your hand in a large circle representing "
                     "the maximum area you can comfortably reach.\n\n"
                     "We will measure the boundary coordinates to map your cursor speed."
            )
            self.action_button.configure(text="Record Reach")
            self.status_label.configure(text="Ready to begin")
            
        elif self.step == 2:
            self.desc_label.configure(
                text="Step 2: Calibrate Pinch Distance\n\n"
                     "When you click 'Record Pinch', perform a tight pinch (touch thumb and "
                     "index finger tips together) and hold it.\n\n"
                     "This sets the precise threshold for mouse clicking."
            )
            self.action_button.configure(text="Record Pinch")
            self.status_label.configure(text="Ready to begin")
            
        elif self.step == 3:
            self.desc_label.configure(
                text="Step 3: Calibrate Baseline Hand Size\n\n"
                     "When you click 'Record Hand Size', hold your hand completely flat with "
                     "all fingers open and face your palm towards the camera.\n\n"
                     "This scales coordinates based on your hand distance."
            )
            self.action_button.configure(text="Record Hand Size")
            self.status_label.configure(text="Ready to begin")
            
        elif self.step == 4:
            self.desc_label.configure(
                text="Calibration Complete!\n\n"
                     "All custom settings have been calculated and saved.\n\n"
                     "SmartAir Mouse Pro will now respond with optimal sensitivity "
                     "adapted for your tracking environment."
            )
            self.action_button.configure(text="Finish Wizard")
            self.status_label.configure(text="Success!", text_color="#39FF14")

    def _on_action_clicked(self) -> None:
        """Handles actions at each stage of the calibration."""
        if self.step == 0:
            self.step = 1
            self._show_step_content()
            
        elif self.step == 1:
            # Start reach calibration countdown
            self._start_data_collection(duration=5, text="Calibrating Reach... Move hand around!")
            
        elif self.step == 2:
            # Start pinch calibration countdown
            self._start_data_collection(duration=3, text="Pinch and hold thumb + index tips!")
            
        elif self.step == 3:
            # Start hand size calibration countdown
            self._start_data_collection(duration=3, text="Hold hand flat facing camera!")
            
        elif self.step == 4:
            # Save results and close
            self._save_calibration_results()
            self._on_close()

    def _start_data_collection(self, duration: int, text: str) -> None:
        """Sets countdown flags to begin processing metadata frame callbacks.

        Args:
            duration: Cooldown duration in seconds.
            text: Indicator status label.
        """
        self.measured_xs.clear()
        self.measured_ys.clear()
        self.measured_pinches.clear()
        self.measured_sizes.clear()
        
        self.calibrating_data = True
        self.countdown_remaining = duration
        self.last_countdown_time = time.time()
        self.status_label.configure(text=f"{text} ({self.countdown_remaining}s)")
        self.action_button.configure(state=tk.DISABLED)
        self._tick_countdown(text)

    def _tick_countdown(self, text: str) -> None:
        """Ticks the GUI countdown timer asynchronously."""
        if not self.is_running or not self.calibrating_data:
            return

        now = time.time()
        if now - self.last_countdown_time >= 1.0:
            self.countdown_remaining -= 1
            self.last_countdown_time = now

        if self.countdown_remaining <= 0:
            # Complete step data collecting
            self.calibrating_data = False
            self.action_button.configure(state=tk.NORMAL)
            self._process_collected_data()
        else:
            self.status_label.configure(text=f"{text} ({self.countdown_remaining}s)")
            self.after(100, lambda: self._tick_countdown(text))

    def process_frame_data(self, tracking_metadata: Dict[str, Any]) -> None:
        """Processes live camera landmark frames from the main tracking thread.

        Args:
            tracking_metadata: Hand tracker metadata dict.
        """
        if not self.calibrating_data or not self.is_running:
            return

        hand_data = tracking_metadata.get("hand")
        if not hand_data:
            return

        landmarks = hand_data.get("landmarks", [])
        if len(landmarks) < 21:
            return

        # Collect data based on current step
        if self.step == 1:
            # Record coordinates of Index Tip (#8)
            index_pt = landmarks[INDEX_FINGER_TIP]
            self.measured_xs.append(index_pt[0])
            self.measured_ys.append(index_pt[1])
            
        elif self.step == 2:
            # Record distance between Thumb Tip (#4) and Index Tip (#8)
            dist = calculate_distance(landmarks[THUMB_TIP], landmarks[INDEX_FINGER_TIP])
            self.measured_pinches.append(dist)
            
        elif self.step == 3:
            # Record hand size: wrist (#0) to middle finger tip (#12)
            dist = calculate_distance(landmarks[WRIST], landmarks[MIDDLE_FINGER_TIP])
            self.measured_sizes.append(dist)

    def _process_collected_data(self) -> None:
        """Averages and logs collected details, proceeding to the next wizard index."""
        if self.step == 1:
            if len(self.measured_xs) > 5 and len(self.measured_ys) > 5:
                # Discard outliers (top/bottom 5%)
                self.measured_xs.sort()
                self.measured_ys.sort()
                cut_x = int(len(self.measured_xs) * 0.05)
                cut_y = int(len(self.measured_ys) * 0.05)
                
                self.temp_reach_xmin = self.measured_xs[cut_x]
                self.temp_reach_xmax = self.measured_xs[-cut_x-1]
                self.temp_reach_ymin = self.measured_ys[cut_y]
                self.temp_reach_ymax = self.measured_ys[-cut_y-1]
                
                # Check for sanity
                if self.temp_reach_xmax - self.temp_reach_xmin > 0.1 and self.temp_reach_ymax - self.temp_reach_ymin > 0.1:
                    logger.info("Calibrated reach: X=(%0.2f, %0.2f), Y=(%0.2f, %0.2f)",
                                self.temp_reach_xmin, self.temp_reach_xmax, self.temp_reach_ymin, self.temp_reach_ymax)
                    self.status_label.configure(text="Reach recorded!", text_color="#39FF14")
                else:
                    # Fallback to defaults
                    self.temp_reach_xmin, self.temp_reach_xmax = 0.25, 0.75
                    self.temp_reach_ymin, self.temp_reach_ymax = 0.25, 0.75
                    self.status_label.configure(text="Invalid reach. Used defaults.", text_color="#FF3131")
            else:
                self.temp_reach_xmin, self.temp_reach_xmax = 0.25, 0.75
                self.temp_reach_ymin, self.temp_reach_ymax = 0.25, 0.75
                self.status_label.configure(text="No hand detected. Used defaults.", text_color="#FF3131")
                
            self.step = 2
            self.after(1000, self._show_step_content)

        elif self.step == 2:
            if self.measured_pinches:
                avg_pinch = sum(self.measured_pinches) / len(self.measured_pinches)
                # Set dynamic threshold slightly larger than average pinch distance
                self.temp_click_threshold = max(0.02, min(avg_pinch * 1.25, 0.07))
                logger.info("Calibrated pinch click threshold: %0.4f (avg was %0.4f)", 
                            self.temp_click_threshold, avg_pinch)
                self.status_label.configure(text="Pinch recorded!", text_color="#39FF14")
            else:
                self.temp_click_threshold = 0.04
                self.status_label.configure(text="No hand detected. Used default.", text_color="#FF3131")
                
            self.step = 3
            self.after(1000, self._show_step_content)

        elif self.step == 3:
            if self.measured_sizes:
                avg_size = sum(self.measured_sizes) / len(self.measured_sizes)
                self.temp_hand_size = max(0.1, min(avg_size, 0.5))
                logger.info("Calibrated hand size: %0.4f", self.temp_hand_size)
                self.status_label.configure(text="Hand size recorded!", text_color="#39FF14")
            else:
                self.temp_hand_size = 0.3
                self.status_label.configure(text="No hand detected. Used default.", text_color="#FF3131")
                
            self.step = 4
            self.after(1000, self._show_step_content)

    def _save_calibration_results(self) -> None:
        """Writes calibrated parameters back to Settings."""
        try:
            self.settings.set("reach_xmin", float(self.temp_reach_xmin), auto_save=False)
            self.settings.set("reach_xmax", float(self.temp_reach_xmax), auto_save=False)
            self.settings.set("reach_ymin", float(self.temp_reach_ymin), auto_save=False)
            self.settings.set("reach_ymax", float(self.temp_reach_ymax), auto_save=False)
            self.settings.set("click_threshold", float(self.temp_click_threshold), auto_save=False)
            self.settings.set("hand_size_baseline", float(self.temp_hand_size), auto_save=True)
            logger.info("Saved all calibration results to settings.json.")
            if self.on_complete:
                self.on_complete()
        except AttributeError:
            # If any attributes were missed, log errors
            logger.error("Failed to save calibration results because wizard was aborted early.")

    def _on_close(self) -> None:
        """Closes modal resources."""
        self.is_running = False
        self.calibrating_data = False
        self.grab_release()
        self.destroy()
