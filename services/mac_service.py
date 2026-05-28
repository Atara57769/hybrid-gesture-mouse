from .base import BaseMouseService

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
