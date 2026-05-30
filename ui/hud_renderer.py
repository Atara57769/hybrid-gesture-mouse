import cv2
import numpy as np
from state_machine.gesture_state import GestureState
from config.settings import CLASSES, STATE_COLORS

# Try standard MediaPipe solutions or fallback drawing shim
try:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.drawing_styles as mp_drawing_styles
    USE_SHIM = False
except (ModuleNotFoundError, AttributeError):
    import utils.mediapipe_shim as mp_hands
    from utils.mediapipe_shim import draw_custom_landmarks as mp_drawing
    USE_SHIM = True

class HudRenderer:
    """
    Encapsulates all OpenCV HUD graphics overlays, screen guide rectangles,
    telemetry overlay drawing, and neon skeletal meshes. Contains NO execution/business logic.
    """
    def __init__(self, active_zone_pct: float = 0.6):
        self.active_zone_pct = active_zone_pct
        
    def draw_hand_landmarks(self, frame, hand_landmarks) -> None:
        """Draws a premium neon hand skeleton mesh onto the frame."""
        if not hand_landmarks:
            return
            
        if USE_SHIM:
            mp_drawing(frame, hand_landmarks)
        else:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
    def draw_hud(self, frame, state: GestureState, confidence: float, screen_x: int, screen_y: int, fps: float, vote_history: list[GestureState]) -> None:
        """
        Renders a glassmorphic high-tech Heads-Up Display showing system metrics,
        active zone tracking boundaries, and engine performance telemetry.
        """
        height, width, _ = frame.shape
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Main panel background
        cv2.rectangle(overlay, (15, 15), (380, 210), (20, 20, 20), -1)
        
        # Active Zone boundary guide (central 60% rectangle)
        margin_x = (1.0 - self.active_zone_pct) / 2.0
        margin_y = (1.0 - self.active_zone_pct) / 2.0
        az_x_min, az_x_max = int(width * margin_x), int(width * (1.0 - margin_x))
        az_y_min, az_y_max = int(height * margin_y), int(height * (1.0 - margin_y))
        
        cv2.rectangle(frame, (az_x_min, az_y_min), (az_x_max, az_y_max), (0, 255, 255), 1)
        cv2.putText(frame, "ACTIVE ZONE GUIDE", (az_x_min + 5, az_y_min - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Blend overlay for HUD translucent glass effect (75% opacity)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # HUD Title
        cv2.putText(frame, "GESTURE MOUSE SYSTEM v1.2", (25, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (25, 48), (365, 48), (80, 80, 80), 1)
        
        # 1. Stabilized State Badge
        state_name = CLASSES[state.value].upper()
        color_bgr = STATE_COLORS.get(CLASSES[state.value], (255, 255, 255))
        
        cv2.putText(frame, "SYSTEM STATE:", (25, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Draw state badge background
        cv2.rectangle(frame, (155, 62), (280, 87), color_bgr, -1)
        cv2.putText(frame, state_name, (165, 81), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            
        # 2. Confidence Value
        cv2.putText(frame, "CONFIDENCE:", (25, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        conf_color = (0, 255, 0) if confidence >= 0.75 else (0, 165, 255) if confidence >= 0.5 else (0, 0, 255)
        cv2.putText(frame, f"{confidence * 100:.1f}%", (155, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_color, 2, cv2.LINE_AA)
        
        # 3. Cursor Coordinate mapping
        cv2.putText(frame, "SCREEN POSITION:", (25, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"X: {screen_x:4d} | Y: {screen_y:4d}", (155, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 4. Engine FPS
        cv2.putText(frame, "ENGINE PERFORMANCE:", (25, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:.1f} FPS", (200, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100) if fps > 25 else (0, 255, 255), 2, cv2.LINE_AA)
        
        # 5. Voters Queue Visualization
        cv2.putText(frame, "VOTES:", (25, 195), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        vote_str = " -> ".join([str(v.value) for v in vote_history])
        cv2.putText(frame, vote_str, (85, 195), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        
        # Safety Stop banner at bottom
        cv2.putText(frame, "PUSH HAND TO SCREEN CORNER FOR EMERGENCY FAIL-SAFE", (15, height - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
