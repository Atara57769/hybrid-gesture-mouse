import sys
from abc import ABC, abstractmethod

class BaseMouseService(ABC):
    """
    Abstract Base Class outlining OS-level mouse control actions.
    Decouples core ML and tracking engine from concrete OS interfaces.
    """
    
    @abstractmethod
    def move_to(self, x: int, y: int) -> None:
        """Move the cursor to specific absolute screen coordinates (x, y)."""
        pass
        
    @abstractmethod
    def click(self) -> None:
        """Trigger a primary left-click at the current cursor position."""
        pass
        
    @abstractmethod
    def press_down(self) -> None:
        """Hold down the primary mouse button (starts a drag)."""
        pass
        
    @abstractmethod
    def release_up(self) -> None:
        """Release the primary mouse button (ends a drag)."""
        pass
        
    @abstractmethod
    def scroll(self, amount: int) -> None:
        """Scroll the mouse wheel vertically (positive = up, negative = down)."""
        pass
        
    @abstractmethod
    def get_screen_size(self) -> tuple[int, int]:
        """Returns the monitor's width and height in pixels."""
        pass
        
    @abstractmethod
    def get_position(self) -> tuple[int, int]:
        """Returns the current absolute (x, y) coordinates of the cursor."""
        pass


class MacMouseService(BaseMouseService):
    """
    Concrete implementation of BaseMouseService for macOS.
    Utilizes PyAutoGUI optimized for high-frequency interaction.
    """
    def __init__(self):
        import pyautogui
        self.pyautogui = pyautogui
        # Eliminate PyAutoGUI pause delay for native feel (60+ FPS loop)
        self.pyautogui.PAUSE = 0
        self.pyautogui.FAILSAFE = True
        
    def move_to(self, x: int, y: int) -> None:
        self.pyautogui.moveTo(x, y)
        
    def click(self) -> None:
        self.pyautogui.click()
        
    def press_down(self) -> None:
        self.pyautogui.mouseDown()
        
    def release_up(self) -> None:
        self.pyautogui.mouseUp()
        
    def scroll(self, amount: int) -> None:
        self.pyautogui.scroll(amount)
        
    def get_screen_size(self) -> tuple[int, int]:
        return self.pyautogui.size()
        
    def get_position(self) -> tuple[int, int]:
        return self.pyautogui.position()


class WindowsMouseService(BaseMouseService):
    """
    Concrete implementation of BaseMouseService for Windows.
    Utilizes direct Win32 ctypes calls for zero-latency, zero-dependency operations.
    """
    def __init__(self):
        import ctypes
        self.ctypes = ctypes
        self.user32 = ctypes.windll.user32
        
        # Mouse event constants (Win32 API)
        self.MOUSEEVENTF_LEFTDOWN = 0x0002
        self.MOUSEEVENTF_LEFTUP = 0x0004
        self.MOUSEEVENTF_WHEEL = 0x0800
        
        # Define POINT structure to fetch cursor position
        class POINT(self.ctypes.Structure):
            _fields_ = [("x", self.ctypes.c_long), ("y", self.ctypes.c_long)]
        self.POINT = POINT
        
    def move_to(self, x: int, y: int) -> None:
        # Instant direct hardware position manipulation
        self.user32.SetCursorPos(x, y)
        
    def click(self) -> None:
        # Trigger Down and Up sequentially
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        
    def press_down(self) -> None:
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        
    def release_up(self) -> None:
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        
    def scroll(self, amount: int) -> None:
        # Win32 wheel scrolls in multiples of WHEEL_DELTA (120)
        # amount > 0 is up, amount < 0 is down
        wheel_amount = int(amount * 120)
        self.user32.mouse_event(self.MOUSEEVENTF_WHEEL, 0, 0, wheel_amount, 0)
        
    def get_screen_size(self) -> tuple[int, int]:
        # Fetch display metrics from OS
        width = self.user32.GetSystemMetrics(0)
        height = self.user32.GetSystemMetrics(1)
        return (width, height)
        
    def get_position(self) -> tuple[int, int]:
        pt = self.POINT()
        self.user32.GetCursorPos(self.ctypes.byref(pt))
        return (pt.x, pt.y)


class MouseServiceFactory:
    """
    Simple Factory to instantiate and return the proper platform-specific 
    BaseMouseService concrete subclass on startup.
    """
    
    @staticmethod
    def get_service() -> BaseMouseService:
        platform = sys.platform
        print(f"[MouseServiceFactory] Detecting system platform: '{platform}'")
        
        if platform == "win32":
            print("[MouseServiceFactory] Initializing high-performance WindowsMouseService (win32 ctypes)...")
            return WindowsMouseService()
        elif platform == "darwin":
            print("[MouseServiceFactory] Initializing MacMouseService (macOS PyAutoGUI/Quartz)...")
            return MacMouseService()
        else:
            # Fallback service (MacMouseService uses standard PyAutoGUI which works on Linux too)
            print(f"[MouseServiceFactory] Platform '{platform}' is generic. Initializing cross-platform PyAutoGUI service...")
            return MacMouseService()


def create_mouse_service() -> BaseMouseService:
    """
    Detects the current operating system (macOS vs Windows vs generic/Linux) 
    and instantiates the correct concrete BaseMouseService subclass.
    """
    return MouseServiceFactory.get_service()
