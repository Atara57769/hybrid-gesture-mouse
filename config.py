# Centralized configurations and constants for the Hybrid Hand Gesture Mouse system

# Class mapping for hand gestures (0-4)
CLASSES = {
    0: "idle",
    1: "move",
    2: "click",
    3: "drag",
    4: "scroll"
}

# Color palettes for HUD visualization (BGR format for OpenCV)
STATE_COLORS = {
    "idle": (128, 128, 128),     # Gray
    "move": (0, 255, 0),         # Green
    "click": (255, 0, 0),        # Blue
    "drag": (0, 165, 255),       # Orange
    "scroll": (0, 255, 255)      # Yellow
}

# Default File Paths
DATASET_PATH = "gestures_dataset.csv"
MODEL_PATH = "gesture_model.pkl"

# Controller default parameters
SMOOTHING = 0.25
CONFIDENCE = 0.75
HISTORY_SIZE = 7
CLICK_DEBOUNCE = 0.4
SCROLL_SENSITIVITY = 1.5
