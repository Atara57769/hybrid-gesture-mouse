from enum import IntEnum

class GestureState(IntEnum):
    """
    IntEnum representing system states and classes.
    Removes magic integers and improves testability and readability.
    """
    IDLE = 0
    MOVE = 1
    CLICK = 2
    DRAG = 3
    SCROLL_UP = 4
    SCROLL_DOWN = 5
