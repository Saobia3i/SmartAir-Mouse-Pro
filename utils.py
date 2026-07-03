"""Utility functions and helper modules for SmartAir Mouse Pro.

Includes logging configuration, synthetic sound wave generation, winsound player,
and mathematical distance/angle helpers.
"""

import math
import struct
import wave
import logging
import threading
import os
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

def setup_logging(log_file: str = "app.log") -> None:
    """Configures application-wide logging.

    Args:
        log_file: Path to the log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger.info("Logging initialized. Output target: %s", log_file)


class SoundManager:
    """Handles sound effects generation and asynchronous playback."""

    def __init__(self, sounds_dir: Path) -> None:
        """Initializes and pre-generates sound files if missing.

        Args:
            sounds_dir: Directory where audio assets will be saved.
        """
        self.sounds_dir = sounds_dir
        self.click_sound_path = self.sounds_dir / "click.wav"
        self._playback_thread = None
        self._enabled = True
        
        self._ensure_sound_files()

    def set_enabled(self, enabled: bool) -> None:
        """Enables or disables sound playbacks."""
        self._enabled = enabled

    def _ensure_sound_files(self) -> None:
        """Generates a high-quality click wav file if it does not exist."""
        if not self.click_sound_path.exists():
            try:
                self._generate_synthetic_click(self.click_sound_path)
                logger.info("Generated synthetic click sound file: %s", self.click_sound_path)
            except Exception as e:
                logger.error("Failed to generate synthetic sound. Error: %s", e)

    def _generate_synthetic_click(self, filepath: Path) -> None:
        """Synthesizes a short, punchy electronic click sound wave.

        Args:
            filepath: Path to save the WAV file.
        """
        sample_rate = 44100
        duration = 0.04  # 40 ms
        start_freq = 1500.0
        end_freq = 300.0

        num_samples = int(sample_rate * duration)
        data = bytearray()

        for i in range(num_samples):
            t = i / sample_rate
            # Exponential decay envelope (starts loud, decays to zero rapidly)
            envelope = math.exp(-80.0 * t)
            # Frequency sweep down (chirp)
            current_freq = start_freq - (start_freq - end_freq) * (t / duration)
            # Generate sine wave sample
            sample_val = math.sin(2.0 * math.pi * current_freq * t)
            # Scale to 16-bit integer range
            int_val = int(32767.0 * envelope * sample_val)
            # Convert to little-endian bytes
            data.extend(struct.pack("<h", int_val))

        with wave.open(str(filepath), "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)   # 2 bytes (16-bit)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(data)

    def play_click(self) -> None:
        """Plays the click sound asynchronously using Windows winsound API."""
        if not self._enabled:
            return

        # winsound is only available on Windows, but this is a Windows application.
        # Use simple try-except import fallback for platform safety.
        try:
            import winsound
            # SND_ASYNC plays sound in background, SND_FILENAME specifies it's a file
            winsound.PlaySound(
                str(self.click_sound_path),
                winsound.SND_FILENAME | winsound.SND_ASYNC
            )
        except Exception as e:
            logger.debug("Sound playback error: %s", e)


# Geometry & Landmark math helpers

def calculate_distance(pt1: Tuple[float, float, float], pt2: Tuple[float, float, float]) -> float:
    """Calculates 3D Euclidean distance between two landmarks.

    Args:
        pt1: Landmark 1 (x, y, z).
        pt2: Landmark 2 (x, y, z).

    Returns:
        Euclidean distance.
    """
    return math.sqrt(
        (pt1[0] - pt2[0]) ** 2 +
        (pt1[1] - pt2[1]) ** 2 +
        (pt1[2] - pt2[2]) ** 2
    )

def calculate_distance_2d(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """Calculates 2D Euclidean distance between two points.

    Args:
        pt1: Point 1 (x, y).
        pt2: Point 2 (x, y).

    Returns:
        Euclidean distance.
    """
    return math.sqrt(
        (pt1[0] - pt2[0]) ** 2 +
        (pt1[1] - pt2[1]) ** 2
    )

def get_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Calculates angle (in degrees) at joint b formed by points a, b, and c.

    Args:
        a: Point A (x, y).
        b: Joint Point B (x, y).
        c: Point C (x, y).

    Returns:
        Angle in degrees.
    """
    try:
        ang = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0]) -
            math.atan2(a[1] - b[1], a[0] - b[0])
        )
        return abs(ang) if abs(ang) <= 180 else 360 - abs(ang)
    except Exception:
        return 0.0
