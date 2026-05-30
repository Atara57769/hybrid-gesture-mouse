try:
    import mediapipe.solutions.hands as mp_hands
except (ModuleNotFoundError, AttributeError):
    import utils.mediapipe_shim as mp_hands

class HandTracker:
    """
    Responsible only for MediaPipe hand detection and returning landmarks.
    Does not perform UI drawing or coordinate transformations.
    """
    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.7):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def detect(self, rgb_frame):
        """
        Processes an RGB frame and returns raw hand landmarks for the first detected hand,
        or None if no hand is detected.
        """
        results = self.hands.process(rgb_frame)
        if results and results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None
        
    def close(self):
        """Releases the underlying MediaPipe resources."""
        if hasattr(self, 'hands') and hasattr(self.hands, 'close'):
            self.hands.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
