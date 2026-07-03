"""Cursor effects module for SmartAir Mouse Pro.

Handles simulation and physics calculations for custom visual effects,
including neon cursor, fading mouse trail, click ripples, and particle explosions.
"""

import math
import random
import time
from typing import List, Dict, Tuple, Any


class Particle:
    """Simulates a single physics-based particle."""

    def __init__(self, x: float, y: float, color: str) -> None:
        """Initializes particle velocity, life, and styling.

        Args:
            x: Birth X coordinate.
            y: Birth Y coordinate.
            color: Hex color string.
        """
        self.x = x
        self.y = y
        self.color = color
        
        # Random initial velocities
        angle = random.uniform(0.0, 2.0 * math.pi)
        speed = random.uniform(3.0, 10.0)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - random.uniform(2.0, 5.0)  # Upward bias
        
        self.gravity = 0.35
        self.friction = 0.95
        
        self.max_life = random.randint(15, 30)  # Frames
        self.life = self.max_life
        self.size = random.uniform(3.0, 7.0)

    def update(self) -> bool:
        """Updates particle coordinates and ages it.

        Returns:
            True if particle is still alive, False if it has died.
        """
        # Apply physics
        self.vy += self.gravity
        self.vx *= self.friction
        self.vy *= self.friction
        
        self.x += self.vx
        self.y += self.vy
        
        self.life -= 1
        
        # Shrink particle as it dies
        self.size = max(0.5, (self.life / self.max_life) * self.size)
        
        return self.life > 0


class ClickRipple:
    """Simulates an expanding shockwave ripple."""

    def __init__(self, x: float, y: float) -> None:
        """Initializes ripple geometry and expansion rate.

        Args:
            x: Center X.
            y: Center Y.
        """
        self.x = x
        self.y = y
        self.radius = 2.0
        self.max_radius = 50.0
        self.speed = 4.5
        self.alpha = 1.0  # Transparency level [0, 1]

    def update(self) -> bool:
        """Expands and fades the ripple.

        Returns:
            True if ripple is visible, False if fully faded.
        """
        self.radius += self.speed
        
        # Fade based on expansion percentage
        progress = self.radius / self.max_radius
        self.alpha = max(0.0, 1.0 - progress)
        
        return self.radius < self.max_radius


class CursorEffects:
    """Orchestrates particle generators and trail history tracks."""

    def __init__(self) -> None:
        """Initializes empty particle buffers."""
        self.trail: List[Tuple[float, float]] = []
        self.particles: List[Particle] = []
        self.ripples: List[ClickRipple] = []
        self.active_gesture_label = "NONE"
        self.tracker_fps = 0.0

    def add_trail_point(self, x: float, y: float, max_len: int) -> None:
        """Adds a position to trail record and crops it to settings limit.

        Args:
            x: Screen X.
            y: Screen Y.
            max_len: Max items to keep in history.
        """
        if max_len <= 0:
            self.trail.clear()
            return
            
        self.trail.append((x, y))
        if len(self.trail) > max_len:
            self.trail.pop(0)

    def trigger_left_click_effect(self, x: float, y: float, particles_enabled: bool) -> None:
        """Registers a left click ripple and optional sparks.

        Args:
            x: Screen click X.
            y: Screen click Y.
            particles_enabled: Toggles particles.
        """
        self.ripples.append(ClickRipple(x, y))
        
        if particles_enabled:
            # Spawn vibrant red/yellow particles
            colors = ["#FF3131", "#FF9431", "#FFF01F"]
            for _ in range(15):
                self.particles.append(Particle(x, y, random.choice(colors)))

    def trigger_right_click_effect(self, x: float, y: float, particles_enabled: bool) -> None:
        """Registers a right click ripple and optional blue sparks.

        Args:
            x: Screen click X.
            y: Screen click Y.
            particles_enabled: Toggles particles.
        """
        self.ripples.append(ClickRipple(x, y))
        
        if particles_enabled:
            # Spawn cool blue particles
            colors = ["#0066FF", "#00FFCC", "#8A2BE2"]
            for _ in range(15):
                self.particles.append(Particle(x, y, random.choice(colors)))

    def set_gesture_label(self, label: str) -> None:
        """Updates the text overlay label.

        Args:
            label: Text description.
        """
        self.active_gesture_label = label

    def set_fps(self, fps: float) -> None:
        """Updates the tracking system frame rate display value."""
        self.tracker_fps = fps

    def update(self) -> None:
        """Ticks physics engines of all active particles and ripples."""
        # Update particles, discarding dead ones
        self.particles = [p for p in self.particles if p.update()]
        
        # Update ripples, discarding completed ones
        self.ripples = [r for r in self.ripples if r.update()]

    def clear(self) -> None:
        """Flushes all current trail, particle, and ripple structures."""
        self.trail.clear()
        self.particles.clear()
        self.ripples.clear()
        self.active_gesture_label = "NONE"
        self.tracker_fps = 0.0
