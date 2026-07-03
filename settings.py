"""Settings module for managing runtime configuration settings.

Reads and writes configuration values to/from settings.json in a thread-safe manner.
"""

import json
import logging
import threading
from typing import Any, Dict

from config import ConfigManager
from constants import DEFAULT_SETTINGS

logger = logging.getLogger(__name__)

class Settings:
    """Thread-safe management of runtime application configuration."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """Initializes settings and loads from file if available.

        Args:
            config_manager: ConfigManager instance providing the file path.
        """
        self.config_manager = config_manager
        self.filepath = config_manager.settings_file
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        
        self.load()

    def load(self) -> None:
        """Loads configuration from JSON file or falls back to defaults."""
        with self._lock:
            if self.filepath.exists():
                try:
                    with open(self.filepath, "r", encoding="utf-8") as f:
                        loaded_data = json.load(f)
                    # Merge with default settings to ensure any missing keys are populated
                    self._data = {**DEFAULT_SETTINGS, **loaded_data}
                    logger.info("Configuration successfully loaded from %s", self.filepath)
                except Exception as e:
                    logger.error("Failed to load settings file. Resetting to defaults. Error: %s", e)
                    self._data = DEFAULT_SETTINGS.copy()
            else:
                logger.info("Settings file not found. Initializing with default settings.")
                self._data = DEFAULT_SETTINGS.copy()
                # Run save to write the default settings.json
                self._lock.release()
                try:
                    self.save()
                finally:
                    self._lock.acquire()

    def save(self) -> None:
        """Saves current configuration data to the JSON settings file."""
        with self._lock:
            try:
                # Pretty print settings
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=4)
                logger.info("Configuration saved successfully to %s", self.filepath)
            except Exception as e:
                logger.error("Failed to save settings to %s. Error: %s", self.filepath, e)

    def get(self, key: str) -> Any:
        """Retrieves a configuration value.

        Args:
            key: Setting key.

        Returns:
            The configured value, or default fallback.
        """
        with self._lock:
            return self._data.get(key, DEFAULT_SETTINGS.get(key))

    def set(self, key: str, value: Any, auto_save: bool = True) -> None:
        """Updates a configuration setting.

        Args:
            key: Setting key to update.
            value: Value to store.
            auto_save: If True, writes settings to disk immediately.
        """
        with self._lock:
            self._data[key] = value
            logger.debug("Setting updated: %s = %s", key, value)
            
        if auto_save:
            self.save()

    def reset_to_defaults(self) -> None:
        """Resets all settings back to default configuration and saves to disk."""
        with self._lock:
            self._data = DEFAULT_SETTINGS.copy()
            logger.info("Settings reset to system defaults.")
        self.save()

    def get_all(self) -> Dict[str, Any]:
        """Returns a copy of all settings data.

        Returns:
            Dictionary containing all configurations.
        """
        with self._lock:
            return self._data.copy()
