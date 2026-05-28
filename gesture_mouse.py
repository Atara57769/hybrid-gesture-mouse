import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import time
import argparse
import pyautogui
from collections import deque, Counter
from normalization import normalize_landmarks
from mouse_service import create_mouse_service

# Try standard MediaPipe, fallback to custom tasks shim if solutions unavailable (e.g. Python 3.13)
try:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.drawing_styles as mp_drawing_styles
    USE_SHIM = False
except (ModuleNotFoundError, AttributeError):
    import mediapipe_shim as mp_hands
    from mediapipe_shim import draw_custom_landmarks as mp_drawing
    USE_SHIM = True

# Class mapping
CLASSES = {
    0: "idle",
    1: "move",
    2: "click",
    3: "drag",
    4: "scroll"
}

# Color palettes for HUD
STATE_COLORS = {
    "idle": (128, 128, 128),     # Gray
    "move": (0, 255, 0),         # Green
    "click": (255, 0, 0),        # Blue (RGB coordinate is BGR in OpenCV, so actually Blue is (255,0,0) in BGR)
    "drag": (0, 165, 255),       # Orange
    "scroll": (0, 255, 255)      # Yellow
}

def get_stabilized_state(history_deque):
    """Returns the most frequent state in the history (majority vote)."""
    if not history_deque:
        return 0
    counter = Counter(history_deque)
    return counter.most_common(1)[0][0]

def draw_futuristic_hud(frame, state, confidence, screen_x, screen_y, fps, vote_history):
    """
    Renders a stunning, high-tech glassmorphic heads-up display overlay on the video feed.
    """
    height, width, _ = frame.shape
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    
    # Main panel background
    cv2.rectangle(overlay, (15, 15), (380, 210), (20, 20, 20), -1)
    
    # Active Zone boundary guide (central 60% rectangle)
    az_x_min, az_x_max = int(width * 0.2), int(width * 0.8)
    az_y_min, az_y_max = int(height * 0.2), int(height * 0.8)
    cv2.rectangle(frame, (az_x_min, az_y_min), (az_x_max, az_y_max), (0, 255, 255), 1)
    cv2.putText(frame, "ACTIVE ZONE GUIDE", (az_x_min + 5, az_y_min - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Blend overlay for HUD translucent glass effect (75% opacity)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # HUD Title
    cv2.putText(frame, "GESTURE MOUSE SYSTEM v1.1", (25, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (25, 48), (365, 48), (80, 80, 80), 1)
    
    # 1. Stabilized State Badge
    state_name = CLASSES[state].upper()
    color_bgr = STATE_COLORS.get(CLASSES[state], (255, 255, 255))
    
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
    vote_str = " -> ".join([str(v) for v in vote_history])
    cv2.putText(frame, vote_str, (85, 195), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    
    # Safety Stop banner at bottom
    cv2.putText(frame, "PUSH HAND TO SCREEN CORNER FOR EMERGENCY FAIL-SAFE", (15, height - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)


class GestureMouseController:
    """
    Decoupled and Dependency Injected controller that drives the real-time loop,
    performs landmark analysis, classifies states, and delegates all OS mouse controls
    to the injected BaseMouseService subclass.
    """
    def __init__(self, mouse_service, model_path, smoothing=0.25, confidence=0.75, history=7, debounce=0.4, scroll_sens=1.5):
        self.mouse_service = mouse_service
        self.model_path = model_path
        self.smoothing = smoothing
        self.confidence = confidence
        self.history = history
        self.debounce = debounce
        self.scroll_sens = scroll_sens
        
    def run(self):
        # 1. Load ML Model
        print(f"Loading gesture classification model: {self.model_path}...")
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
            
        # 2. Get injected screen resolution
        screen_width, screen_height = self.mouse_service.get_screen_size()
        print(f"Active Screen Bounds: {screen_width}x{screen_height}")
        
        # 3. Initialize MediaPipe Hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # 4. Slide-window voting queue
        vote_history = deque(maxlen=self.history)
        
        # 5. Controller state variables
        last_click_time = 0
        is_dragging = False
        prev_smoothed_x, prev_smoothed_y = self.mouse_service.get_position()
        prev_index_tip_y = None
        
        # 6. FPS Tracking
        prev_frame_time = time.time()
        
        # 7. Start Webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam stream.")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n--- Hand Gesture Mouse Controller Engaged ---")
        print("Keep your hand inside the yellow tracking frame.")
        print("Move your index finger to track. Pinch to click or drag. Double finger to scroll.")
        print("Move cursor to any screen corner or press 'Q' to terminate.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
                
            current_time = time.time()
            fps = 1.0 / (current_time - prev_frame_time)
            prev_frame_time = current_time
            
            # Mirror BGR frame
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            
            # Define Active Zone guide in pixels
            az_x_min, az_x_max = int(width * 0.2), int(width * 0.8)
            az_y_min, az_y_max = int(height * 0.2), int(height * 0.8)
            
            # RGB conversion and detection processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            predicted_state = 0
            prediction_prob = 1.0
            raw_x, raw_y = None, None
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw skeleton beautifully
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
                
                # Index finger tip (8) for coordinates
                index_tip = hand_landmarks.landmark[8]
                raw_x = int(index_tip.x * width)
                raw_y = int(index_tip.y * height)
                
                # Normalize features and predict state
                norm_feats = normalize_landmarks(hand_landmarks)
                if norm_feats is not None:
                    features_array = np.array([norm_feats])
                    probs = model.predict_proba(features_array)[0]
                    predicted_state = int(np.argmax(probs))
                    prediction_prob = probs[predicted_state]
                    
                    if prediction_prob >= self.confidence:
                        vote_history.append(predicted_state)
                
                # Calculate mode state
                stabilized_state = get_stabilized_state(vote_history)
            else:
                stabilized_state = 0
                vote_history.clear()
                prev_index_tip_y = None
                
            # Coordinate mapping and smoothing
            screen_target_x, screen_target_y = prev_smoothed_x, prev_smoothed_y
            
            if raw_x is not None and raw_y is not None:
                # Interpolate coordinate in Active Zone
                x_pct = (raw_x - az_x_min) / (az_x_max - az_x_min)
                y_pct = (raw_y - az_y_min) / (az_y_max - az_y_min)
                
                x_pct = max(0.0, min(1.0, x_pct))
                y_pct = max(0.0, min(1.0, y_pct))
                
                target_x = int(x_pct * screen_width)
                target_y = int(y_pct * screen_height)
                
                # Exponential Moving Average (EMA) smoothing
                screen_target_x = int(self.smoothing * target_x + (1.0 - self.smoothing) * prev_smoothed_x)
                screen_target_y = int(self.smoothing * target_y + (1.0 - self.smoothing) * prev_smoothed_y)
                
                prev_smoothed_x, prev_smoothed_y = screen_target_x, screen_target_y
                
            # --- STATE MACHINE EXECUTOR (DECOUPLED via self.mouse_service) ---
            try:
                # Emergency Failsafe check (for Windows, since PyAutoGUI Failsafe only runs on pyautogui)
                # If cursor is placed at any extreme corner, raise safety shutdown exception
                if screen_target_x == 0 or screen_target_y == 0 or screen_target_x >= screen_width - 2 or screen_target_y >= screen_height - 2:
                    raise pyautogui.FailSafeException()
                
                if stabilized_state == 0:  # IDLE
                    if is_dragging:
                        self.mouse_service.release_up()
                        is_dragging = False
                    prev_index_tip_y = None
                    
                elif stabilized_state == 1:  # MOVE
                    if is_dragging:
                        self.mouse_service.release_up()
                        is_dragging = False
                    self.mouse_service.move_to(screen_target_x, screen_target_y)
                    prev_index_tip_y = None
                    
                elif stabilized_state == 2:  # CLICK
                    if is_dragging:
                        self.mouse_service.release_up()
                        is_dragging = False
                        
                    if current_time - last_click_time > self.debounce:
                        self.mouse_service.move_to(screen_target_x, screen_target_y)
                        self.mouse_service.click()
                        last_click_time = current_time
                        print("--> Click Event Triggered via MouseService")
                    prev_index_tip_y = None
                    
                elif stabilized_state == 3:  # DRAG
                    if not is_dragging:
                        self.mouse_service.move_to(screen_target_x, screen_target_y)
                        self.mouse_service.press_down()
                        is_dragging = True
                        print("--> Drag Engaged via MouseService (Mouse Down)")
                    else:
                        self.mouse_service.move_to(screen_target_x, screen_target_y)
                    prev_index_tip_y = None
                    
                elif stabilized_state == 4:  # SCROLL
                    if is_dragging:
                        self.mouse_service.release_up()
                        is_dragging = False
                        
                    if raw_y is not None:
                        if prev_index_tip_y is not None:
                            delta_y = raw_y - prev_index_tip_y
                            if abs(delta_y) > 3:
                                # Delegate scroll
                                scroll_val = int(-delta_y * self.scroll_sens)
                                self.mouse_service.scroll(scroll_val)
                        prev_index_tip_y = raw_y
                        
            except pyautogui.FailSafeException:
                print("\n[Safety Trigger] FailSafe activated! Mouse corner exit detected. Terminating safely.")
                if is_dragging:
                    self.mouse_service.release_up()
                break
                
            # HUD overlay
            draw_futuristic_hud(frame, stabilized_state, prediction_prob, screen_target_x, screen_target_y, fps, list(vote_history))
            
            # Show output
            cv2.imshow("Hand Gesture Mouse - Live Controller", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Clean resources
        if is_dragging:
            self.mouse_service.release_up()
        cap.release()
        cv2.destroyAllWindows()
        print("--- Gesture Mouse System Disengaged safely. Goodbye! ---")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Hand Gesture Mouse Control")
    parser.add_argument("--model", type=str, default="gesture_model.pkl", help="Path to trained model pickle")
    parser.add_argument("--smoothing", type=float, default=0.25, help="EMA smoothing factor (0 = no update, 1 = instant)")
    parser.add_argument("--confidence", type=float, default=0.75, help="Min probability to change states")
    parser.add_argument("--history", type=int, default=7, help="Majority voting history queue size")
    parser.add_argument("--debounce", type=float, default=0.4, help="Debounce cooldown for clicks in seconds")
    parser.add_argument("--scroll-sens", type=float, default=1.5, help="Scroll sensitivity factor")
    
    args = parser.parse_args()
    
    # 1. Verify model file exists
    if not os.path.exists(args.model):
        print(f"\n[Error] Trained model file '{args.model}' not found!")
        print("Please train a model using 'python train.py' before running this controller,")
        print("or run 'python train.py --synthetic' to create a test model file.\n")
        return
        
    # 2. Resolve MouseService dependency via simple factory
    mouse_service = MouseServiceFactory.get_service()
    
    # 3. Instantiate and run controller (Dependency Injection)
    controller = GestureMouseController(
        mouse_service=mouse_service,
        model_path=args.model,
        smoothing=args.smoothing,
        confidence=args.confidence,
        history=args.history,
        debounce=args.debounce,
        scroll_sens=args.scroll_sens
    )
    
    controller.run()


if __name__ == "__main__":
    main()
