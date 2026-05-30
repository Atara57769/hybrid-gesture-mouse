import sys
from .base import BaseMouseService
from .mac_service import MacMouseService
from .windows_service import WindowsMouseService
from utils.logger import get_logger

logger = get_logger("mouse_service")

def create_mouse_service() -> BaseMouseService:
    """
    Detects the current operating system (macOS vs Windows vs generic/Linux) 
    and instantiates the correct concrete BaseMouseService subclass.
    """
    platform = sys.platform
    logger.info(f"Detecting platform: '{platform}'")
    
    if platform == "win32":
        logger.info("Initializing high-performance WindowsMouseService (win32 ctypes)...")
        return WindowsMouseService()
    elif platform == "darwin":
        logger.info("Initializing MacMouseService (macOS PyAutoGUI/Quartz)...")
        return MacMouseService()
    else:
        # Fallback service (MacMouseService uses standard PyAutoGUI which works on Linux too)
        logger.info(f"Platform '{platform}' is generic. Initializing cross-platform PyAutoGUI service...")
        return MacMouseService()
