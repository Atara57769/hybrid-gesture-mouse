import time

class FPSCounter:
    """Tracks running loop iteration execution rates."""
    def __init__(self):
        self.prev_time = time.time()
        
    def tick(self) -> float:
        """
        Updates the FPS counter and returns the instantaneous FPS.
        """
        current_time = time.time()
        elapsed = current_time - self.prev_time
        if elapsed <= 0.0:
            elapsed = 0.001
        fps = 1.0 / elapsed
        self.prev_time = current_time
        return fps
