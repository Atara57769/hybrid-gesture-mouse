# Centralized configurations and constants for the Layered Hybrid Gesture Mouse

# Class mapping for hand gestures (0-5)
CLASSES = {
    0: "idle",
    1: "move",
    2: "click",
    3: "drag",
    4: "scroll_up",
    5: "scroll_down"
}

# Color palettes for HUD visualization (BGR format for OpenCV)
STATE_COLORS = {
    "idle": (128, 128, 128),     # Gray
    "move": (0, 255, 0),         # Green
    "click": (255, 0, 0),        # Blue
    "drag": (0, 165, 255),       # Orange
    "scroll_up": (255, 0, 255),  # Magenta
    "scroll_down": (255, 255, 0) # Cyan
}

# Default File Paths (relative to project root)
DATASET_PATH = "training/gestures_dataset.csv"
MODEL_PATH = "models/gesture_model.pkl"
HAND_LANDMARKER_PATH = "models/hand_landmarker.task"

# Controller default parameters
SMOOTHING = 0.25
CONFIDENCE = 0.75
HISTORY_SIZE = 7
CLICK_DEBOUNCE = 0.4
SCROLL_SENSITIVITY = 1.5
SCROLL_STEP = 2
