#!/usr/bin/env python3
"""
main.py
MeMyselfAI - macOS Desktop Application
Entry point for the application
"""

import sys
import ctypes
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ui.main_window import MainWindow


def _resource_path(name: str) -> Path:
    """Resolve resources in both development and PyInstaller builds."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / name
    return Path(__file__).parent / name


def _set_windows_app_id() -> None:
    """Give Windows a stable app identity so the taskbar uses our icon."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "com.memyselfai.app"
        )
    except Exception:
        pass


def _load_app_icon() -> QIcon | None:
    """Prefer the ICO for Windows, fall back to the bundled PNG."""
    for icon_name in ("MeMyselfAi.ico", "MeMyselfAi.png"):
        icon_path = _resource_path(icon_name)
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            if not icon.isNull():
                return icon
    return None


def main():
    """Main application entry point"""
    _set_windows_app_id()

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("MeMyselfAI")
    app.setOrganizationName("MeMyselfAI")
    app.setStyle("Fusion")  # Ensures stylesheet arrow/button subcontrols render correctly on macOS

    app_icon = _load_app_icon()
    if app_icon is not None:
        app.setWindowIcon(app_icon)
    
    # Set macOS-specific attributes
    if sys.platform == "darwin":
        app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    
    # Create and show main window
    window = MainWindow()
    if app_icon is not None:
        window.setWindowIcon(app_icon)
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
