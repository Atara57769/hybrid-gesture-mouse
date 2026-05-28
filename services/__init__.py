import sys
from .base import BaseMouseService
from .mac_service import MacMouseService
from .windows_service import WindowsMouseService

def create_mouse_service() -> BaseMouseService:
    """
    Detects the current operating system (macOS vs Windows vs generic/Linux) 
    and instantiates the correct concrete BaseMouseService subclass.
    """
    platform = sys.platform
    print(f"[create_mouse_service] Detecting platform: '{platform}'")
    
    if platform == "win32":
        print("[create_mouse_service] Initializing high-performance WindowsMouseService (win32 ctypes)...")
        return WindowsMouseService()
    elif platform == "darwin":
        print("[create_mouse_service] Initializing MacMouseService (macOS PyAutoGUI/Quartz)...")
        return MacMouseService()
    else:
        # Fallback service (MacMouseService uses standard PyAutoGUI which works on Linux too)
        print(f"[create_mouse_service] Platform '{platform}' is generic. Initializing cross-platform PyAutoGUI service...")
        return MacMouseService()
