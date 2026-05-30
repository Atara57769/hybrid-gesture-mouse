import numpy as np
from state_machine.gesture_state import GestureState

class GestureClassifier:
    """
    Encapsulates gesture prediction logic. Runs normalized landmarks through the loaded ML model.
    """
    def __init__(self, model):
        self.model = model
        
    def predict(self, normalized_landmarks: list[float]) -> tuple[GestureState, float]:
        """
        Runs prediction inference. Returns the predicted GestureState enum and prediction probability.
        """
        features_array = np.array([normalized_landmarks])
        probs = self.model.predict_proba(features_array)[0]
        predicted_idx = int(np.argmax(probs))
        prob = probs[predicted_idx]
        return GestureState(predicted_idx), prob
