from collections import deque, Counter
from state_machine.gesture_state import GestureState

class VoteStabilizer:
    """
    Implements sliding-window majority voting over subsequent frames.
    Stabilizes real-time prediction variations.
    """
    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
    def add_vote(self, state: GestureState) -> None:
        """Appends a new prediction vote to the queue history."""
        self.history.append(state)
        
    def get_stabilized_state(self) -> GestureState:
        """Computes the majority voted state from the active queue window."""
        if not self.history:
            return GestureState.IDLE
        counter = Counter(self.history)
        # Returns the most common state in the history
        return counter.most_common(1)[0][0]
        
    def clear(self) -> None:
        """Clears all accumulated votes history."""
        self.history.clear()
        
    def get_history(self) -> list[GestureState]:
        """Returns the current raw list of active history votes."""
        return list(self.history)
