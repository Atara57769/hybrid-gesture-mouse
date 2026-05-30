import time
import pyautogui
from state_machine.gesture_state import GestureState
from utils.logger import get_logger

logger = get_logger("mouse_state_machine")

class MouseStateMachine:
    """
    Manages OS-level mouse control commands, transitions, and state flags.
    Translates gesture states to action calls on the injected MouseService.
    """
    def __init__(self, mouse_service, debounce: float = 0.4, scroll_sens: float = 1.5, scroll_step: int = 2):
        self.mouse_service = mouse_service
        self.debounce = debounce
        self.scroll_sens = scroll_sens
        self.scroll_step = scroll_step
        
        # State variables
        self.is_dragging = False
        self.last_click_time = 0.0
        self.has_clicked = False
        
    def execute(self, state: GestureState, x: int, y: int) -> None:
        """
        Executes action corresponding to the stabilized state and target (x, y) coordinates.
        Checks for extreme screen corners to trigger a fail-safe exception.
        """
        screen_width, screen_height = self.mouse_service.get_screen_size()
        
        # Emergency Failsafe check (e.g. corner boundary check)
        if x == 0 or y == 0 or x >= screen_width - 2 or y >= screen_height - 2:
            raise pyautogui.FailSafeException("FailSafe activated: Mouse corner exit detected.")
            
        current_time = time.time()
        
        # Click state latching: if not in click state, reset the clicked latch
        if state != GestureState.CLICK:
            self.has_clicked = False
            
        if state == GestureState.IDLE:
            if self.is_dragging:
                self.mouse_service.release_up()
                self.is_dragging = False
                logger.info("Drag Released (transition to IDLE)")
                
        elif state == GestureState.MOVE:
            if self.is_dragging:
                self.mouse_service.release_up()
                self.is_dragging = False
                logger.info("Drag Released (transition to MOVE)")
            self.mouse_service.move_to(x, y)
            
        elif state == GestureState.CLICK:
            if self.is_dragging:
                self.mouse_service.release_up()
                self.is_dragging = False
                logger.info("Drag Released (transition to CLICK)")
                
            if not self.has_clicked:
                if current_time - self.last_click_time > self.debounce:
                    self.mouse_service.move_to(x, y)
                    self.mouse_service.click()
                    self.last_click_time = current_time
                    self.has_clicked = True
                    logger.info("Click Event Triggered via MouseService")
                    
        elif state == GestureState.DRAG:
            if not self.is_dragging:
                self.mouse_service.move_to(x, y)
                self.mouse_service.press_down()
                self.is_dragging = True
                logger.info("Drag Engaged via MouseService (Mouse Down)")
            else:
                self.mouse_service.move_to(x, y)
                
        elif state == GestureState.SCROLL_UP:
            if self.is_dragging:
                self.mouse_service.release_up()
                self.is_dragging = False
                logger.info("Drag Released (transition to SCROLL_UP)")
            scroll_val = int(self.scroll_step * self.scroll_sens)
            self.mouse_service.scroll(scroll_val)
            
        elif state == GestureState.SCROLL_DOWN:
            if self.is_dragging:
                self.mouse_service.release_up()
                self.is_dragging = False
                logger.info("Drag Released (transition to SCROLL_DOWN)")
            scroll_val = int(-self.scroll_step * self.scroll_sens)
            self.mouse_service.scroll(scroll_val)
            
    def shutdown(self) -> None:
        """Ensures all buttons are released safely on termination."""
        if self.is_dragging:
            try:
                self.mouse_service.release_up()
                self.is_dragging = False
                logger.info("Failsafe/Shutdown: Drag Released successfully.")
            except Exception as e:
                logger.error(f"Error releasing drag during shutdown: {e}")
            
