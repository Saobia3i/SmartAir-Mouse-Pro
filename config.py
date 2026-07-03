"""Configuration management module.

Handles directory setup, path resolution, and default configuration fallback.
"""

import os
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """Manages system paths, configuration loading/saving, and directories initialization."""

    def __init__(self, root_dir: str = ".") -> None:
        """Initializes the directories and paths for the application.

        Args:
            root_dir: The root directory of the project.
        """
        self.root = Path(root_dir).resolve()
        self.config_dir = self.root / "config"
        self.settings_file = self.config_dir / "settings.json"
        self.screenshots_dir = self.root / "screenshots"
        self.assets_dir = self.root / "assets"
        self.sounds_dir = self.assets_dir / "sounds"
        self.icons_dir = self.assets_dir / "icons"
        self.cursor_dir = self.assets_dir / "cursor"
        
        self.initialize_directories()

    def initialize_directories(self) -> None:
        """Creates the necessary directories for logs, screenshots, configuration, and assets."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.sounds_dir.mkdir(parents=True, exist_ok=True)
        self.icons_dir.mkdir(parents=True, exist_ok=True)
        self.cursor_dir.mkdir(parents=True, exist_ok=True)
        
        # Add .gitkeep to screenshots to maintain it in VCS if empty
        gitkeep = self.screenshots_dir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
