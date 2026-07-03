"""Main desktop application for SmartAir Mouse Pro.

Orchestrates CustomTkinter GUI layout, camera feed drawing, tracking threads,
overlay canvas displays, calibration wizards, and statistics reporting.
"""

import os
import sys
import time
import queue
import logging
import threading
import tkinter as tk
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import pyautogui
import customtkinter as ctk
from pynput import keyboard

# Add workspace directory to sys.path to allow execution from any folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from constants import (
    DEFAULT_SETTINGS, TARGET_FPS, GESTURE_SCREENSHOT, GESTURE_NONE,
    COLOR_BG_DARK, COLOR_SIDEBAR_DARK, GESTURE_DESCRIPTIONS
)
from config import ConfigManager
from settings import Settings
from utils import setup_logging, SoundManager
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from mouse_controller import MouseController
from cursor_effects import CursorEffects
from overlay import OverlayWindow
from calibration import CalibrationWizard

# Setup logs
setup_logging()
logger = logging.getLogger(__name__)

# Modern theme configurations
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SmartAirMouseApp(ctk.CTk):
    """The main desktop GUI application window."""

    def __init__(self) -> None:
        """Initializes dependencies, UI widgets, and thread worker loops."""
        super().__init__()
        
        self.title("SmartAir Mouse Pro")
        self.geometry("1100x700")
        self.resizable(True, True)
        
        # 1. Initialize core system modules
        self.config_manager = ConfigManager()
        self.settings = Settings(self.config_manager)
        self.sound_manager = SoundManager(self.config_manager.sounds_dir)
        self.sound_manager.set_enabled(self.settings.get("sound_enabled"))
        
        self.tracker = HandTracker(
            camera_index=self.settings.get("camera_index"),
            detection_confidence=0.75,
            tracking_confidence=0.75
        )
        self.recognizer = GestureRecognizer(
            click_threshold=self.settings.get("click_threshold")
        )
        self.mouse_controller = MouseController(self.settings)
        self.mouse_control_enabled = False
        self.effects = CursorEffects()
        
        # 2. UI Windows & Threading states
        self.overlay: Optional[OverlayWindow] = None
        self.wizard: Optional[CalibrationWizard] = None
        
        self.is_running = True
        self.is_tracking = False
        self.is_starting = False
        self.startup_thread: Optional[threading.Thread] = None
        self.tracking_thread: Optional[threading.Thread] = None
        self.keyboard_listener: Optional[keyboard.Listener] = None
        self.session_start_time = 0.0
        self.last_simple_screenshot_time = 0.0
        
        # Frame queues for thread-safe UI updates
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.meta_queue: queue.Queue = queue.Queue(maxsize=2)
        
        # Total frames tracked vs processed for accuracy stats
        self.total_frames = 0
        self.tracked_frames = 0
        
        # Register close handler
        self.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.bind("<Escape>", lambda _event: self.emergency_stop())
        
        # 3. Build UI Layout
        self.configure(fg_color=COLOR_BG_DARK)
        self._build_layout()
        
        # Initialize Overlay Window
        self.overlay = OverlayWindow(self, self.settings, self.effects)
        self.overlay.hide()
        
        # Auto-update slider displays from loaded settings
        self._load_ui_values()
        
        # Start GUI polling timer loop (updates camera preview and statistics)
        self._poll_ui_queues()
        self._start_keyboard_listener()

    def _start_keyboard_listener(self) -> None:
        """Starts a global keyboard listener for the emergency stop shortcut."""
        if self.keyboard_listener is not None:
            return

        def _on_press(key: keyboard.Key) -> None:
            if key == keyboard.Key.esc:
                self.after(0, self.emergency_stop)
            elif key == keyboard.Key.f8:
                self.after(0, lambda: self._set_mouse_control(not self.mouse_control_enabled))

        self.keyboard_listener = keyboard.Listener(on_press=_on_press)
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()

    def _build_layout(self) -> None:
        """Draws the sidebar, statistics panels, and camera view frames."""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # ==========================================
        # LEFT PANEL: SIDEBAR SETTINGS
        # ==========================================
        self.sidebar = ctk.CTkFrame(self, fg_color=COLOR_SIDEBAR_DARK, corner_radius=0, width=280)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_propagate(False)
        
        self.sidebar_label = ctk.CTkLabel(
            self.sidebar,
            text="SETTINGS PANEL",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color="#00FFCC"
        )
        self.sidebar_label.pack(pady=(20, 15))
        
        # Scrollable area for settings widgets
        self.scroll_frame = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.scroll_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Camera Input Selector
        self._add_sidebar_dropdown(
            "Camera Source",
            "camera_index",
            ["Camera 0", "Camera 1", "Camera 2"],
            command=self._on_camera_selected
        )
        
        # Sensitivity Slider
        self._add_sidebar_slider("Sensitivity", "sensitivity_x", 0.5, 4.0, "%0.1fx")
        
        # Smoothing Factor
        self._add_sidebar_slider("Smoothing Factor", "smoothing_factor", 0.05, 0.8, "%0.2f")
        
        # Trail Length
        self._add_sidebar_slider("Trail Length", "trail_length", 0, 50, "%d", is_int=True)
        
        # Trail Color Dropdown
        self._add_sidebar_dropdown(
            "Trail Color",
            "trail_color",
            ["Cyan", "Green", "Red", "Yellow", "Blue"],
            command=self._on_trail_color_selected
        )
        
        # Cursor Glow Size
        self._add_sidebar_slider("Cursor Glow Size", "cursor_size", 5, 30, "%d px", is_int=True)
        
        # Click Pinch Threshold
        self._add_sidebar_slider("Pinch Sensitivity", "click_threshold", 0.02, 0.08, "%0.3f")
        
        # Scroll Speed
        self._add_sidebar_slider("Scroll Speed", "scroll_speed", 0.2, 3.0, "%0.1fx")
        
        # Sound Toggle Switch
        self.sound_switch = ctk.CTkSwitch(
            self.scroll_frame,
            text="Sound Effects",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self._on_sound_toggled
        )
        self.sound_switch.pack(pady=10, anchor=tk.W, padx=15)
        
        # Particles Toggle Switch
        self.particle_switch = ctk.CTkSwitch(
            self.scroll_frame,
            text="Visual Particles",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self._on_particles_toggled
        )
        self.particle_switch.pack(pady=10, anchor=tk.W, padx=15)

        # Draw HUD Skeleton Toggle Switch
        self.hud_skeleton_switch = ctk.CTkSwitch(
            self.scroll_frame,
            text="Show Screen Skeleton",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self._on_hud_skeleton_toggled
        )
        self.hud_skeleton_switch.pack(pady=10, anchor=tk.W, padx=15)

        self.mouse_control_switch = ctk.CTkSwitch(
            self.scroll_frame,
            text="Mouse Control",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=lambda: self._set_mouse_control(self.mouse_control_switch.get() == 1)
        )
        self.mouse_control_switch.pack(pady=10, anchor=tk.W, padx=15)

        self.sidebar_test_click_btn = ctk.CTkButton(
            self.scroll_frame,
            text="Test Click",
            fg_color="#333333",
            hover_color="#444444",
            text_color="#FFFFFF",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            height=34,
            command=self._test_left_click
        )
        self.sidebar_test_click_btn.pack(fill=tk.X, padx=15, pady=(4, 12))
        
        # ==========================================
        # RIGHT PANEL: STATS, PREVIEW & BUTTONS
        # ==========================================
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1)
        
        # Main Header
        self.header_label = ctk.CTkLabel(
            self.main_container,
            text="SMARTAIR MOUSE PRO",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#FFFFFF"
        )
        self.header_label.grid(row=0, column=0, sticky="w", pady=(0, 15))
        
        # 1. Statistics Panel (Top-Right Grid)
        self.stats_grid = ctk.CTkFrame(self.main_container, fg_color="#1E1E1E", height=100)
        self.stats_grid.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        
        # Create 6 stat columns
        self.stat_widgets: Dict[str, ctk.CTkLabel] = {}
        stat_configs = [
            ("Time Active", "time", "00:00:00"),
            ("Distance", "distance", "0 px"),
            ("Left Clicks", "clicks", "0"),
            ("Right Clicks", "right_clicks", "0"),
            ("Scrolls", "scrolls", "0"),
            ("System Rate", "fps", "0.0 FPS")
        ]
        
        for col_idx, (title, key, default_val) in enumerate(stat_configs):
            self.stats_grid.grid_columnconfigure(col_idx, weight=1)
            
            box = ctk.CTkFrame(self.stats_grid, fg_color="transparent")
            box.grid(row=0, column=col_idx, padx=10, pady=10, sticky="nsew")
            
            t_lbl = ctk.CTkLabel(box, text=title.upper(), font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"), text_color="#888888")
            t_lbl.pack(anchor=tk.CENTER)
            
            v_lbl = ctk.CTkLabel(box, text=default_val, font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), text_color="#00FFCC")
            v_lbl.pack(anchor=tk.CENTER, pady=(2, 0))
            
            self.stat_widgets[key] = v_lbl
            
        # 2. Main content split: video stream view & gesture instruction legend
        self.center_split = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.center_split.grid(row=2, column=0, sticky="nsew", pady=(0, 15))
        self.center_split.grid_columnconfigure(0, weight=3)  # Camera
        self.center_split.grid_columnconfigure(1, weight=2)  # Legend
        self.center_split.grid_rowconfigure(0, weight=1)

        # Video feed view label frame
        self.video_frame = ctk.CTkFrame(self.center_split, fg_color="#000000", height=380)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.video_frame.pack_propagate(False)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="CAMERA INACTIVE\nClick 'Start Tracking' to open stream", text_color="#555555", font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Legend frame list
        self.legend_frame = ctk.CTkFrame(self.center_split, fg_color="#1E1E1E")
        self.legend_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        legend_title = ctk.CTkLabel(self.legend_frame, text="GESTURE CONTROL GUIDE", font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"), text_color="#00FFCC")
        legend_title.pack(pady=(15, 10))
        
        # Scrollable gesture container
        self.legend_scroll = ctk.CTkScrollableFrame(self.legend_frame, fg_color="transparent", height=300)
        self.legend_scroll.pack(fill=ctk.BOTH, expand=True, padx=5, pady=0)
        
        for g_name, g_desc in GESTURE_DESCRIPTIONS.items():
            g_box = ctk.CTkFrame(self.legend_scroll, fg_color="#121212", corner_radius=6)
            g_box.pack(fill=tk.X, pady=4, padx=5)
            
            # Format visual name
            lbl_name = ctk.CTkLabel(g_box, text=g_name.replace("_", " "), font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"), text_color="#0066FF")
            lbl_name.pack(anchor=tk.W, padx=10, pady=(4, 0))
            
            lbl_desc = ctk.CTkLabel(g_box, text=g_desc, font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#DFDFDF")
            lbl_desc.pack(anchor=tk.W, padx=10, pady=(0, 4))
            
        # 3. Action Control Buttons (Bottom Panel)
        self.actions_bar = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.actions_bar.grid(row=3, column=0, sticky="ew")

        self.mouse_control_main_switch = ctk.CTkSwitch(
            self.actions_bar,
            text="Mouse Control: OFF",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            command=lambda: self._set_mouse_control(self.mouse_control_main_switch.get() == 1)
        )
        self.mouse_control_main_switch.pack(side=tk.LEFT, padx=(0, 12))
        
        self.start_btn = ctk.CTkButton(
            self.actions_bar, text="Start Tracking", fg_color="#228B22", hover_color="#1E7B1E",
            text_color="#FFFFFF", font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            height=40, command=self.start_tracking
        )
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 8))
        
        self.stop_btn = ctk.CTkButton(
            self.actions_bar, text="Stop Tracking", fg_color="#B22222", hover_color="#9A1D1D",
            text_color="#FFFFFF", font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            height=40, state=tk.DISABLED, command=self.stop_tracking
        )
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=8)
        
        self.calibrate_btn = ctk.CTkButton(
            self.actions_bar, text="Run Calibration", fg_color="#0066FF", hover_color="#0052CC",
            text_color="#FFFFFF", font=ctk.CTkFont(family="Segoe UI", size=13),
            height=40, command=self.open_calibration
        )
        self.calibrate_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=8)
        
        self.reset_btn = ctk.CTkButton(
            self.actions_bar, text="Reset Statistics", fg_color="#333333", hover_color="#444444",
            text_color="#FFFFFF", font=ctk.CTkFont(family="Segoe UI", size=13),
            height=40, command=self.reset_statistics
        )
        self.reset_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(8, 0))

        self.test_click_btn = ctk.CTkButton(
            self.actions_bar, text="Test Click", fg_color="#444444", hover_color="#555555",
            text_color="#FFFFFF", font=ctk.CTkFont(family="Segoe UI", size=12),
            width=90, height=40, command=self._test_left_click
        )
        self.test_click_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.test_click_count = 0

    # ==========================================
    # SETTINGS ELEMENT BUILDERS
    # ==========================================
    def _add_sidebar_slider(self, label_text: str, key: str, min_val: float, max_val: float,
                            val_format: str, is_int: bool = False) -> None:
        """Adds a parameter slider with text feedback label into the scroll panel.

        Args:
            label_text: Parameter title.
            key: Settings key link.
            min_val: Minimum boundary.
            max_val: Maximum boundary.
            val_format: Formatting code for display label.
            is_int: If True, locks slider inputs to integer increments.
        """
        lbl_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        lbl_frame.pack(fill=tk.X, pady=(8, 2), padx=10)
        
        t_lbl = ctk.CTkLabel(lbl_frame, text=label_text, font=ctk.CTkFont(family="Segoe UI", size=12), text_color="#FFFFFF")
        t_lbl.pack(side=tk.LEFT)
        
        v_lbl = ctk.CTkLabel(lbl_frame, text="", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"), text_color="#00FFCC")
        v_lbl.pack(side=tk.RIGHT)
        
        def _on_slider_changed(value: float) -> None:
            if is_int:
                value = int(round(value))
            self.settings.set(key, value, auto_save=False)
            if key == "sensitivity_x":
                self.settings.set("sensitivity_y", value, auto_save=False)
            v_lbl.configure(text=val_format % value)
            
            # Hot reload thresholds/trail parameters
            if key == "click_threshold":
                self.recognizer.update_thresholds(value)

        initial_val = self.settings.get(key)
        slider = ctk.CTkSlider(
            self.scroll_frame,
            from_=min_val,
            to=max_val,
            number_of_steps=100 if not is_int else int(max_val - min_val),
            command=_on_slider_changed
        )
        slider.set(initial_val)
        slider.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        # Trigger initial display label render
        _on_slider_changed(initial_val)

    def _add_sidebar_dropdown(self, label_text: str, key: str, options: list, command: Any) -> None:
        """Adds a dropdown menu inside settings panel."""
        lbl_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        lbl_frame.pack(fill=tk.X, pady=(8, 2), padx=10)
        
        t_lbl = ctk.CTkLabel(lbl_frame, text=label_text, font=ctk.CTkFont(family="Segoe UI", size=12), text_color="#FFFFFF")
        t_lbl.pack(side=tk.LEFT)
        
        dropdown = ctk.CTkOptionMenu(
            self.scroll_frame,
            values=options,
            command=lambda val: command(val)
        )
        dropdown.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        # Cache to check value
        current_val = self.settings.get(key)
        if key == "camera_index":
            dropdown.set(f"Camera {current_val}")
        elif key == "trail_color":
            # Map hex to name
            hex_map = {
                "#00FFCC": "Cyan",
                "#39FF14": "Green",
                "#FF3131": "Red",
                "#FFF01F": "Yellow",
                "#0066FF": "Blue"
            }
            dropdown.set(hex_map.get(current_val, "Cyan"))

    # ==========================================
    # SETTINGS CHANGE HANDLERS
    # ==========================================
    def _on_camera_selected(self, choice: str) -> None:
        idx = int(choice.split(" ")[-1])
        self.settings.set("camera_index", idx)
        logger.info("Default Camera selection updated to index %d", idx)

    def _on_trail_color_selected(self, choice: str) -> None:
        color_map = {
            "Cyan": "#00FFCC",
            "Green": "#39FF14",
            "Red": "#FF3131",
            "Yellow": "#FFF01F",
            "Blue": "#0066FF"
        }
        hex_color = color_map.get(choice, "#00FFCC")
        self.settings.set("trail_color", hex_color)

    def _on_sound_toggled(self) -> None:
        enabled = self.sound_switch.get() == 1
        self.settings.set("sound_enabled", enabled)
        self.sound_manager.set_enabled(enabled)

    def _on_particles_toggled(self) -> None:
        enabled = self.particle_switch.get() == 1
        self.settings.set("particles_enabled", enabled)

    def _on_hud_skeleton_toggled(self) -> None:
        enabled = self.hud_skeleton_switch.get() == 1
        if self.overlay:
            self.overlay.hand_skeleton_visible = enabled

    def _set_mouse_control(self, enabled: bool) -> None:
        """Toggles real OS mouse control while keeping preview mode available."""
        self.mouse_control_enabled = enabled
        self.mouse_controller.set_enabled(enabled and self.is_tracking)
        if enabled:
            self.mouse_control_switch.select()
            self.mouse_control_main_switch.select()
        else:
            self.mouse_control_switch.deselect()
            self.mouse_control_main_switch.deselect()
        self.mouse_control_main_switch.configure(text=f"Mouse Control: {'ON' if enabled else 'OFF'}")
        logger.info("Mouse control injection %s.", "enabled" if enabled else "disabled")

    def _load_ui_values(self) -> None:
        """Sets toggle switches based on configurations loaded at startup."""
        self.sound_switch.select() if self.settings.get("sound_enabled") else self.sound_switch.deselect()
        self.particle_switch.select() if self.settings.get("particles_enabled") else self.particle_switch.deselect()
        self.hud_skeleton_switch.select()
        self._set_mouse_control(False)

    # ==========================================
    # CORE SYSTEM CONTROLS
    # ==========================================
    def start_tracking(self) -> None:
        """Starts background tracking loops."""
        if self.is_tracking or self.is_starting:
            return
            
        logger.info("Initializing hand tracking processor...")
        self.is_starting = True
        self.mouse_controller.set_enabled(self.mouse_control_enabled)
        self.video_label.configure(text="STARTING CAMERA...\nPlease wait", image="")
        self.start_btn.configure(text="Starting...", state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.calibrate_btn.configure(state=tk.DISABLED)

        self.startup_thread = threading.Thread(target=self._start_tracking_worker, daemon=True)
        self.startup_thread.start()

    def _start_tracking_worker(self) -> None:
        """Opens the camera outside the UI thread, then reports the result."""
        self.tracker.camera_index = self.settings.get("camera_index")
        started = self.tracker.start_camera()
        self.after(0, lambda: self._on_camera_start_result(started))

    def _on_camera_start_result(self, started: bool) -> None:
        """Completes tracking startup on the UI thread after camera initialization."""
        if not self.is_starting:
            if started:
                self.tracker.stop_camera()
            return

        self.is_starting = False
        self.startup_thread = None

        if not started:
            logger.error("Could not start tracking stream. Interface index issues.")
            self.video_label.configure(text="CAMERA CAPTURE ERROR\nPlease select a different camera source index.")
            self.start_btn.configure(text="Start Tracking", state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
            self.calibrate_btn.configure(state=tk.NORMAL)
            self.mouse_controller.set_enabled(False)
            return

        self.is_tracking = True
        self.session_start_time = time.time()
        self._set_mouse_control(True)
        
        self.start_btn.configure(text="Start Tracking", state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.calibrate_btn.configure(state=tk.DISABLED)
        
        # Flush effects queue
        self.effects.clear()
        
        # Launch tracker worker
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        
        logger.info("Tracking thread started.")

    def stop_tracking(self) -> None:
        """Halts background threads and releases video feeds."""
        if not self.is_tracking and not self.is_starting:
            return
            
        logger.info("Stopping hand tracking processor...")
        self.is_starting = False
        self.is_tracking = False
        
        if self.tracking_thread is not None:
            self.tracking_thread.join(timeout=1.0)
            self.tracking_thread = None
            
        self.tracker.stop_camera()
        self.mouse_controller.set_enabled(False)
        
        # Reset buttons states
        self.start_btn.configure(text="Start Tracking", state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.calibrate_btn.configure(state=tk.NORMAL)
        
        # Clear frame display
        self.video_label.configure(text="CAMERA INACTIVE\nClick 'Start Tracking' to open stream", image="")
        
        if self.overlay:
            self.overlay.hide()
            
        logger.info("Hand tracking processor stopped.")

    def emergency_stop(self) -> None:
        """Immediately stops tracking and disables OS mouse injection."""
        logger.warning("Emergency stop requested. Stopping tracking and disabling mouse injection.")
        self.mouse_controller.emergency_stop()
        self._set_mouse_control(False)
        self.is_starting = False
        self.is_tracking = False

        if self.tracking_thread is not None:
            self.tracking_thread.join(timeout=0.5)
            self.tracking_thread = None

        self.tracker.stop_camera()
        self.start_btn.configure(text="Start Tracking", state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.calibrate_btn.configure(state=tk.NORMAL)
        self.video_label.configure(text="EMERGENCY STOPPED\nPress Start Tracking to resume", image="")

        if self.overlay:
            self.overlay.hide()

    def reset_statistics(self) -> None:
        """Resets click counts, distance trackers, and timers."""
        self.mouse_controller.reset_stats()
        self.session_start_time = time.time()
        self.total_frames = 0
        self.tracked_frames = 0
        logger.info("Session statistics reset.")

    def _test_left_click(self) -> None:
        """Runs a direct OS click test at the current cursor location."""
        self.test_click_count += 1
        self.test_click_btn.configure(text=f"Tested {self.test_click_count}")
        self.sidebar_test_click_btn.configure(text=f"Tested {self.test_click_count}")
        self.mouse_controller._trigger_left_click()

    def open_calibration(self) -> None:
        """Launches the calibration wizard modal."""
        # Calibration requires active video stream
        if not self.is_tracking:
            # Temporarily start tracking stream for calibration without system mouse injection
            logger.info("Temporarily starting video stream for calibration...")
            self.start_tracking()
            
        self.wizard = CalibrationWizard(self, self.settings, self._on_calibration_complete)

    def _on_calibration_complete(self) -> None:
        """Triggered when wizard is successfully completed."""
        logger.info("Calibration successfully finished. Updating thresholds...")
        # Sync loaded settings to tracker
        self.recognizer.update_thresholds(self.settings.get("click_threshold"))
        self.wizard = None

    # ==========================================
    # BACKGROUND FRAME ACQUISITION WORKER
    # ==========================================
    def _tracking_loop(self) -> None:
        """Loops CV2 reads and executes hand skeleton mappings in a sub-thread."""
        target_delay = 1.0 / TARGET_FPS
        
        while self.is_tracking and self.is_running:
            try:
                start_tick = time.time()
                
                frame, meta = self.tracker.process_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                self.total_frames += 1
                has_hand = meta.get("hand") is not None
                if has_hand:
                    self.tracked_frames += 1

                # Dispatch gestures
                state = "ACTIVE"
                cursor_x, cursor_y = int(self.mouse_controller.smooth_x), int(self.mouse_controller.smooth_y)
                
                if has_hand:
                    hand_data = meta["hand"]
                    landmarks = hand_data["landmarks"]
                    
                    gesture = "MOVE"
                    
                    # Pass data to wizard if wizard modal exists
                    if self.wizard and self.wizard.winfo_exists() and self.wizard.calibrating_data:
                        self.wizard.process_frame_data(meta)
                        state = "CALIBRATING"
                        self.effects.set_gesture_label("CALIBRATING")
                    else:
                        if self.mouse_control_enabled:
                            cursor_x, cursor_y, gesture = self.mouse_controller.process_touchpad_action(landmarks)
                        else:
                            cursor_x, cursor_y = self.mouse_controller.preview_cursor_position(landmarks)
                            gesture = "PREVIEW"

                        self.effects.set_gesture_label(gesture.replace("_", " "))

                        if gesture in ("LEFT_CLICK", "RIGHT_CLICK", "DRAG"):
                            self._handle_gesture_sound_and_sparks(gesture, cursor_x, cursor_y)

                        if gesture == GESTURE_SCREENSHOT and time.time() - self.last_simple_screenshot_time > 2.0:
                            self.last_simple_screenshot_time = time.time()
                            self._trigger_screenshot()
                else:
                    self.mouse_controller.reset_touch_state()
                    self.effects.set_gesture_label("NO HAND")

                # Update overlay statistical FPS
                self.effects.set_fps(meta.get("fps", 0.0))
                
                # Record trail point
                self.effects.add_trail_point(cursor_x, cursor_y, self.settings.get("trail_length"))

                # Queue frame and metadata for main thread GUI update safely
                # Discard oldest frame if queue full to prevent delays
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                if self.meta_queue.full():
                    self.meta_queue.get_nowait()
                    
                self.frame_queue.put_nowait(frame)
                self.meta_queue.put_nowait((cursor_x, cursor_y, meta, state))

                # Maintain frame loop timing
                elapsed = time.time() - start_tick
                sleep_time = max(0.002, target_delay - elapsed)
                time.sleep(sleep_time)
            except Exception as e:
                logger.exception("Tracking loop failed: %s", e)
                self.effects.set_gesture_label("ERROR")
                time.sleep(0.2)

    def _handle_gesture_sound_and_sparks(self, gesture: str, cx: int, cy: int) -> None:
        """Triggers audio playback and overlays click ripple particles on click gestures."""
        prev = self.recognizer.prev_pinch_states
        particles = self.settings.get("particles_enabled")
        
        # Left Click transition check
        if gesture == "LEFT_CLICK" and not prev["left"]:
            self.sound_manager.play_click()
            self.effects.trigger_left_click_effect(cx, cy, particles)
            
        # Right Click transition check
        elif gesture == "RIGHT_CLICK" and not prev["right"]:
            self.sound_manager.play_click()
            self.effects.trigger_right_click_effect(cx, cy, particles)
            
        # Drag transition check
        elif gesture == "DRAG" and not prev["drag"]:
            self.sound_manager.play_click()

    def _trigger_screenshot(self) -> None:
        """Saves current screen capture asynchronously to avoid lag spikes."""
        def _worker() -> None:
            try:
                # Play audio warning or flash HUD if desired
                logger.info("Triggering screen capture...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                filepath = self.config_manager.screenshots_dir / filename
                
                # Take screenshot using pyautogui
                shot = pyautogui.screenshot()
                shot.save(str(filepath))
                logger.info("Screenshot saved successfully to: %s", filepath)
            except Exception as e:
                logger.error("Failed to save screenshot. Error: %s", e)

        # Run in separate throw-away worker thread
        threading.Thread(target=_worker, daemon=True).start()

    # ==========================================
    # MAIN THREAD POLLING TIMER LOOP
    # ==========================================
    def _poll_ui_queues(self) -> None:
        """Pulls frame queue coordinates to redraw Canvas layers on the main thread."""
        if not self.is_running:
            return

        # 1. Update Video Preview Widget in App GUI
        if not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                # OpenCV handles images in BGR format, resize and map to RGB for Tkinter
                h, w, _ = frame.shape
                # scale camera feed to fit inside widget frame smoothly
                target_w = 480
                target_h = int((h / w) * target_w)
                
                small_frame = cv2.resize(frame, (target_w, target_h))
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                img_pil = Image.fromarray(rgb_small)
                img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(target_w, target_h))
                
                self.video_label.configure(text="", image=img_tk)
                self.video_label.image = img_tk  # Keep reference to prevent GC
            except Exception as e:
                logger.debug("Failed drawing camera preview: %s", e)

        # 2. Draw Overlay Elements
        if not self.meta_queue.empty():
            try:
                cx, cy, meta, state = self.meta_queue.get_nowait()
                if self.overlay:
                    self.overlay.draw(cx, cy, meta, state)
            except Exception as e:
                logger.debug("Failed drawing overlay: %s", e)

        # 3. Refresh statistics panels
        self._update_statistics_hud()

        # Reschedule timer
        self.after(16, self._poll_ui_queues)

    def _update_statistics_hud(self) -> None:
        """Calculates active timer values and writes metrics to the top panel labels."""
        if not self.is_tracking:
            self.stat_widgets["time"].configure(text="00:00:00")
            self.stat_widgets["fps"].configure(text="0.0 FPS")
            return

        # 1. Elapsed Active Time
        elapsed = int(time.time() - self.session_start_time)
        hrs = elapsed // 3600
        mins = (elapsed % 3600) // 60
        secs = elapsed % 60
        self.stat_widgets["time"].configure(text=f"{hrs:02d}:{mins:02d}:{secs:02d}")

        # 2. Distance Traveled
        dist_px = int(self.mouse_controller.stats["distance_moved"])
        self.stat_widgets["distance"].configure(text=f"{dist_px} px")

        # 3. Actions Counts
        self.stat_widgets["clicks"].configure(text=str(self.mouse_controller.stats["clicks"]))
        self.stat_widgets["right_clicks"].configure(text=str(self.mouse_controller.stats["right_clicks"]))
        self.stat_widgets["scrolls"].configure(text=str(self.mouse_controller.stats["scroll_count"]))

        # 4. Processing frame rate
        self.stat_widgets["fps"].configure(text=f"{self.tracker.fps:.1f} FPS")

    # ==========================================
    # APPLICATION SHUTDOWN HANDLERS
    # ==========================================
    def on_exit(self) -> None:
        """Closes files and shuts down background loops cleanly."""
        logger.info("Window close signal received. Shutting down SmartAir Mouse...")
        self.is_running = False
        
        # Stop tracking
        self.stop_tracking()
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        
        # Close tracker resources
        self.tracker.close()
        
        # Save final state settings to settings.json
        self.settings.save()
        
        self.destroy()
        sys.exit(0)


if __name__ == "__main__":
    app = SmartAirMouseApp()
    app.mainloop()
