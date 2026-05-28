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
