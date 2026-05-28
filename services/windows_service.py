from .base import BaseMouseService

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
