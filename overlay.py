"""Overlay module for SmartAir Mouse Pro.

Creates a transparent, borderless, topmost, click-through Tkinter overlay window.
Uses pywin32 styles to render neon halos, cursor trails, particle bursts, and HUD graphics in real-time.
"""

import tkinter as tk
import logging
from typing import Dict, Any, List, Tuple, Optional

from constants import (
    COLOR_NEON_BLUE, COLOR_NEON_GREEN, COLOR_NEON_RED, COLOR_NEON_YELLOW,
    COLOR_NEON_CYAN, WRIST, INDEX_FINGER_TIP
)
from cursor_effects import CursorEffects

logger = logging.getLogger(__name__)

class OverlayWindow:
    """Manages the full-screen transparent click-through rendering layer."""

    def __init__(self, parent: tk.Tk, settings: Any, effects: CursorEffects) -> None:
        """Initializes overlay canvas window and configures Windows API styles.

        Args:
            parent: Parent Tkinter root application window.
            settings: Settings data management class.
            effects: Shared CursorEffects instance.
        """
        self.parent = parent
        self.settings = settings
        self.effects = effects
        
        self.screen_w = parent.winfo_screenwidth()
        self.screen_h = parent.winfo_screenheight()
        
        # Create topmost borderless window
        self.window = tk.Toplevel(parent)
        self.window.title("SmartAir Mouse Overlay")
        self.window.overrideredirect(True)
        self.window.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.window.attributes("-topmost", True)
        
        # Transparent key color
        self.trans_color = "#010101"  # Almost black
        self.window.attributes("-transparentcolor", self.trans_color)
        
        # Create Canvas covering full-screen
        self.canvas = tk.Canvas(
            self.window,
            width=self.screen_w,
            height=self.screen_h,
            bg=self.trans_color,
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Force windows styles for click-through
        self._apply_click_through()
        
        # Internal states
        self.is_active = False
        self.hand_skeleton_visible = True
        
        logger.info("Overlay Window initialized successfully (%dx%d).", self.screen_w, self.screen_h)

    def _apply_click_through(self) -> None:
        """Invokes Windows OS window styles to make the overlay click-through."""
        # Use pywin32 to manipulate window long styles
        try:
            import win32gui
            import win32con
            
            # Fetch window handle
            hwnd = self.window.winfo_id()
            
            # Retrieve current extended styles
            styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            
            # Append LAYERED and TRANSPARENT (click-through) flags
            new_styles = styles | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, new_styles)
            
            # Reposition to force style apply
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
            )
            logger.info("Successfully applied WS_EX_TRANSPARENT click-through flags.")
        except Exception as e:
            logger.error("Failed to apply OS-level click-through styles: %s", e)

    def show(self) -> None:
        """Unhides the overlay layer."""
        self.is_active = True
        self.window.deiconify()
        self.window.attributes("-topmost", True)

    def hide(self) -> None:
        """Hides the overlay layer."""
        self.is_active = False
        self.window.withdraw()
        self.effects.clear()

    def draw(self, cursor_x: int, cursor_y: int, tracking_metadata: Dict[str, Any], tracking_state: str) -> None:
        """Renders all cursor effects, landmarks, and the HUD on the canvas.

        Args:
            cursor_x: Current screen x coordinate.
            cursor_y: Current screen y coordinate.
            tracking_metadata: Metadata dictionary containing hand tracking frame details.
            tracking_state: Active tracking state ("ACTIVE", "PAUSED", "LOCKED").
        """
        if not self.is_active:
            return
            
        # Clear previous frame drawings
        self.canvas.delete("all")
        
        # 1. Update visual simulation states
        self.effects.update()
        
        # 2. Draw Trails
        trail_pts = self.effects.trail
        if len(trail_pts) > 1:
            color = self.settings.get("trail_color")
            # Draw trail segments with decreasing line thickness/opacity (simulated by lines)
            for i in range(1, len(trail_pts)):
                x1, y1 = trail_pts[i-1]
                x2, y2 = trail_pts[i]
                # Scale thickness based on progression
                width = int((i / len(trail_pts)) * 3) + 1
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, capstyle=tk.ROUND)
                
        # 3. Draw Click Ripples
        for rip in self.effects.ripples:
            # Opacity simulated by color fade from white to grey
            gray_val = int(255 * (1.0 - rip.alpha))
            hex_color = f"#{gray_val:02x}ff{gray_val:02x}" # Keep a neon greenish glow
            self.canvas.create_oval(
                rip.x - rip.radius, rip.y - rip.radius,
                rip.x + rip.radius, rip.y + rip.radius,
                outline=hex_color,
                width=2
            )
            
        # 4. Draw Particles
        for part in self.effects.particles:
            self.canvas.create_oval(
                part.x - part.size, part.y - part.size,
                part.x + part.size, part.y + part.size,
                fill=part.color,
                outline=""
            )
            
        # 5. Draw Neon Cursor Halo
        cursor_size = self.settings.get("cursor_size")
        # Draw concentric rings to build the neon glow effect
        self.canvas.create_oval(
            cursor_x - cursor_size - 4, cursor_y - cursor_size - 4,
            cursor_x + cursor_size + 4, cursor_y + cursor_size + 4,
            outline=COLOR_NEON_CYAN,
            width=1
        )
        self.canvas.create_oval(
            cursor_x - cursor_size, cursor_y - cursor_size,
            cursor_x + cursor_size, cursor_y + cursor_size,
            outline=COLOR_NEON_BLUE,
            width=3
        )
        self.canvas.create_oval(
            cursor_x - 3, cursor_y - 3,
            cursor_x + 3, cursor_y + 3,
            fill=COLOR_NEON_CYAN,
            outline=""
        )

        # 6. Draw Gesture label next to cursor
        gesture_desc = self.effects.active_gesture_label
        if gesture_desc != "NONE":
            self.canvas.create_text(
                cursor_x + cursor_size + 12,
                cursor_y + 4,
                text=gesture_desc,
                fill="#FFFFFF",
                font=("Segoe UI", 10, "bold"),
                anchor=tk.W
            )

        # 7. Draw Hand Landmarks Skeleton directly on screen if enabled
        hand_data = tracking_metadata.get("hand")
        if self.hand_skeleton_visible and hand_data:
            self._draw_overlay_skeleton(hand_data)

        # 8. Draw Top HUD Display
        self._draw_hud(tracking_state, tracking_metadata.get("fps", 0.0), hand_data)

    def _draw_overlay_skeleton(self, hand_data: Dict[str, Any]) -> None:
        """Translates landmarks to screen space and draws the skeleton overlay.

        Args:
            hand_data: Active hand landmark dictionary.
        """
        landmarks = hand_data.get("landmarks", [])
        if not landmarks:
            return
            
        # Screen mapping scale parameters
        reach_xmin = self.settings.get("reach_xmin")
        reach_xmax = self.settings.get("reach_xmax")
        reach_ymin = self.settings.get("reach_ymin")
        reach_ymax = self.settings.get("reach_ymax")

        screen_pts: List[Tuple[float, float]] = []
        for lm in landmarks:
            # Map landmarks identically to screen coordinate space
            mx = (lm[0] - reach_xmin) / (reach_xmax - reach_xmin)
            my = (lm[1] - reach_ymin) / (reach_ymax - reach_ymin)
            
            # Map to screen pixels (slightly clamp to avoid drawing out of bounds)
            px = int(mx * self.screen_w)
            py = int(my * self.screen_h)
            screen_pts.append((px, py))

        # Joint connections list (MediaPipe standards)
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
            # Knuckle links
            (5, 9), (9, 13), (13, 17)
        ]

        # Draw bones
        for start, end in connections:
            if start < len(screen_pts) and end < len(screen_pts):
                self.canvas.create_line(
                    screen_pts[start][0], screen_pts[start][1],
                    screen_pts[end][0], screen_pts[end][1],
                    fill="#39FF14",  # Neon green bones
                    width=1
                )

        # Draw joint nodes
        for pt in screen_pts:
            self.canvas.create_oval(
                pt[0] - 3, pt[1] - 3,
                pt[0] + 3, pt[1] + 3,
                fill=COLOR_NEON_YELLOW,
                outline=""
            )

        # Draw Bounding Box if metadata has it
        bbox = hand_data.get("bounding_box")
        if bbox:
            # Bounding box coordinates are in camera pixel space, we map the bounding box
            # corners to screen space for proper alignment.
            xmin, ymin, xmax, ymax = bbox
            # Bounding box is in camera scale. Let's map normalized corners to screen
            cam_w = tracking_w = 1280 # default fallback
            cam_h = tracking_h = 720
            # Normalize bounding box corners first
            n_xmin = xmin / cam_w
            n_ymin = ymin / cam_h
            n_xmax = xmax / cam_w
            n_ymax = ymax / cam_h
            
            # Map corners using mapping function
            s_xmin = int(((n_xmin - reach_xmin) / (reach_xmax - reach_xmin)) * self.screen_w)
            s_ymin = int(((n_ymin - reach_ymin) / (reach_ymax - reach_ymin)) * self.screen_h)
            s_xmax = int(((n_xmax - reach_xmin) / (reach_xmax - reach_xmin)) * self.screen_w)
            s_ymax = int(((n_ymax - reach_ymin) / (reach_ymax - reach_ymin)) * self.screen_h)
            
            self.canvas.create_rectangle(
                s_xmin, s_ymin, s_xmax, s_ymax,
                outline=COLOR_NEON_RED,
                width=1,
                dash=(4, 4)
            )

    def _draw_hud(self, state: str, tracker_fps: float, hand_data: Optional[Dict[str, Any]]) -> None:
        """Draws top HUD control status, accuracy, and confidence metrics.

        Args:
            state: The tracker running state.
            tracker_fps: Frames per second.
            hand_data: Hand dictionary containing confidence.
        """
        # Determine status color
        state_color = COLOR_NEON_GREEN
        if state == "PAUSED":
            state_color = COLOR_NEON_YELLOW
        elif state == "LOCKED":
            state_color = COLOR_NEON_RED

        # Draw background container for HUD
        hud_x, hud_y = 30, 30
        self.canvas.create_rectangle(
            hud_x, hud_y, hud_x + 320, hud_y + 110,
            fill="#1E1E1E",
            outline=state_color,
            width=2
        )

        # Draw Title
        self.canvas.create_text(
            hud_x + 15, hud_y + 20,
            text="SMARTAIR MOUSE PRO",
            fill="#FFFFFF",
            font=("Segoe UI", 12, "bold"),
            anchor=tk.W
        )

        # Draw Status
        self.canvas.create_text(
            hud_x + 15, hud_y + 45,
            text=f"STATUS: {state}",
            fill=state_color,
            font=("Segoe UI", 10, "bold"),
            anchor=tk.W
        )

        # Draw FPS
        self.canvas.create_text(
            hud_x + 15, hud_y + 65,
            text=f"FPS: {tracker_fps:.1f}",
            fill="#AAAAAA",
            font=("Segoe UI", 10),
            anchor=tk.W
        )

        # Draw confidence meter
        confidence = hand_data.get("confidence", 0.0) if hand_data else 0.0
        conf_percent = int(confidence * 100)
        
        self.canvas.create_text(
            hud_x + 15, hud_y + 85,
            text=f"TRACK CONF: {conf_percent}%",
            fill="#AAAAAA",
            font=("Segoe UI", 10),
            anchor=tk.W
        )

        # Visual confidence meter bar
        bar_w = 120
        self.canvas.create_rectangle(
            hud_x + 180, hud_y + 80, hud_x + 180 + bar_w, hud_y + 90,
            fill="#333333",
            outline=""
        )
        if conf_percent > 0:
            fill_w = int(bar_w * confidence)
            self.canvas.create_rectangle(
                hud_x + 180, hud_y + 80, hud_x + 180 + fill_w, hud_y + 90,
                fill=COLOR_NEON_GREEN if confidence > 0.6 else COLOR_NEON_YELLOW,
                outline=""
            )
