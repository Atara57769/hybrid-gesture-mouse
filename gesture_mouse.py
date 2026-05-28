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

# Explicitly import submodules to avoid attribute resolution issues
import mediapipe.solutions.hands as mp_hands
import mediapipe.solutions.drawing_utils as mp_drawing
import mediapipe.solutions.drawing_styles as mp_drawing_styles

# Disable PyAutoGUI delay for instantaneous response (60+ FPS)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True  # Move mouse to any corner to abort execution

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
    cv2.putText(frame, "GESTURE MOUSE SYSTEM v1.0", (25, 40), 
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

def main():
    parser = argparse.ArgumentParser(description="Real-Time Hand Gesture Mouse Control")
    parser.add_argument("--model", type=str, default="gesture_model.pkl", help="Path to trained model pickle")
    parser.add_argument("--smoothing", type=float, default=0.25, help="EMA smoothing factor (0 = no update, 1 = instant)")
    parser.add_argument("--confidence", type=float, default=0.75, help="Min probability to change states")
    parser.add_argument("--history", type=int, default=7, help="Majority voting history queue size")
    parser.add_argument("--debounce", type=float, default=0.4, help="Debounce cooldown for clicks in seconds")
    parser.add_argument("--scroll-sens", type=float, default=1.5, help="Scroll sensitivity factor")
    
    args = parser.parse_args()
    
    # Verify model file exists
    if not os.path.exists(args.model):
        print(f"\n[Error] Trained model file '{args.model}' not found!")
        print("Please train a model using 'python train.py' before running this controller,")
        print("or run 'python train.py --synthetic' to create a test model file.\n")
        return
        
    print(f"Loading gesture classification model: {args.model}...")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
        
    # Get system screen metrics
    screen_width, screen_height = pyautogui.size()
    print(f"Detected screen resolution: {screen_width}x{screen_height}")
    
    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Sliding window majority voting
    vote_history = deque(maxlen=args.history)
    
    # Debouncing and states
    last_click_time = 0
    is_dragging = False
    prev_smoothed_x, prev_smoothed_y = pyautogui.position()
    prev_index_tip_y = None
    
    # FPS measurement
    prev_frame_time = time.time()
    
    # Camera capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n--- Hand Gesture Mouse Controller Engaged ---")
    print("Keep your hand inside the active yellow tracking frame.")
    print("Move your index finger to track. Pinch to click or drag. Double finger to scroll.")
    print("Move cursor to any screen corner or press 'Q' to terminate.")
    
    current_state = 0 # default idle
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        current_time = time.time()
        fps = 1.0 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        # Define Active Zone coordinates in pixels
        az_x_min, az_x_max = int(width * 0.2), int(width * 0.8)
        az_y_min, az_y_max = int(height * 0.2), int(height * 0.8)
        
        # RGB conversions
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Inference variables
        predicted_state = 0
        prediction_prob = 1.0
        
        raw_x, raw_y = None, None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks beautifully
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract Index finger tip (landmark 8) for cursor positioning
            index_tip = hand_landmarks.landmark[8]
            raw_x = int(index_tip.x * width)
            raw_y = int(index_tip.y * height)
            
            # Perform normalized inference
            norm_feats = normalize_landmarks(hand_landmarks)
            if norm_feats is not None:
                features_array = np.array([norm_feats])
                
                # Predict probability distribution
                probs = model.predict_proba(features_array)[0]
                predicted_state = int(np.argmax(probs))
                prediction_prob = probs[predicted_state]
                
                # Apply confidence filter before appending to voting buffer
                if prediction_prob >= args.confidence:
                    vote_history.append(predicted_state)
                else:
                    # Fallback to last known state if unsure, or append nothing
                    pass
                    
            # Compute stabilized state via majority voting
            stabilized_state = get_stabilized_state(vote_history)
        else:
            # No hand detected, system falls back to idle
            stabilized_state = 0
            vote_history.clear()
            prediction_prob = 1.0
            prev_index_tip_y = None
            
        # Coordinates Interpolation (Active Center-Zone mapping)
        screen_target_x, screen_target_y = prev_smoothed_x, prev_smoothed_y
        
        if raw_x is not None and raw_y is not None:
            # Map [az_x_min, az_x_max] to [0, screen_width]
            x_pct = (raw_x - az_x_min) / (az_x_max - az_x_min)
            y_pct = (raw_y - az_y_min) / (az_y_max - az_y_min)
            
            # Clamping
            x_pct = max(0.0, min(1.0, x_pct))
            y_pct = max(0.0, min(1.0, y_pct))
            
            # Screen projection coordinates
            target_x = int(x_pct * screen_width)
            target_y = int(y_pct * screen_height)
            
            # EMA cursor smoothing
            screen_target_x = int(args.smoothing * target_x + (1.0 - args.smoothing) * prev_smoothed_x)
            screen_target_y = int(args.smoothing * target_y + (1.0 - args.smoothing) * prev_smoothed_y)
            
            prev_smoothed_x, prev_smoothed_y = screen_target_x, screen_target_y
            
        # --- STATE MACHINE EXECUTOR ---
        try:
            # Ensure safety corner exit can trigger without locks
            if stabilized_state == 0:  # IDLE
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                prev_index_tip_y = None
                
            elif stabilized_state == 1:  # MOVE
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                pyautogui.moveTo(screen_target_x, screen_target_y)
                prev_index_tip_y = None
                
            elif stabilized_state == 2:  # CLICK
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                
                # Debounce timer check
                if current_time - last_click_time > args.debounce:
                    # Move to location first, then click
                    pyautogui.moveTo(screen_target_x, screen_target_y)
                    pyautogui.click()
                    last_click_time = current_time
                    print("--> Click Event Triggered!")
                prev_index_tip_y = None
                
            elif stabilized_state == 3:  # DRAG
                # If drag transition just began
                if not is_dragging:
                    pyautogui.moveTo(screen_target_x, screen_target_y)
                    pyautogui.mouseDown()
                    is_dragging = True
                    print("--> Drag Mode Engaged (Mouse Down)")
                else:
                    # Continuous drag tracking
                    pyautogui.moveTo(screen_target_x, screen_target_y)
                prev_index_tip_y = None
                    
            elif stabilized_state == 4:  # SCROLL
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                    
                # Track vertical movement of the index finger
                if raw_y is not None:
                    if prev_index_tip_y is not None:
                        delta_y = raw_y - prev_index_tip_y
                        
                        # Apply noise dead-zone threshold of 3 pixels
                        if abs(delta_y) > 3:
                            # Scroll direction: index tip moves up (y decreases) -> scroll up (positive value)
                            # MacOS scrolling behaves naturally this way
                            scroll_val = int(-delta_y * args.scroll_sens)
                            pyautogui.scroll(scroll_val)
                    
                    prev_index_tip_y = raw_y
                    
        except pyautogui.FailSafeException:
            print("\n[Safety Trigger] PyAutoGUI FailSafe activated! Mouse corner exit detected. Terminating safely.")
            if is_dragging:
                pyautogui.mouseUp()
            break
            
        # Draw glassmorphic HUD panel
        draw_futuristic_hud(frame, stabilized_state, prediction_prob, screen_target_x, screen_target_y, fps, list(vote_history))
        
        # Show stream
        cv2.imshow("Hand Gesture Mouse - Live Controller", frame)
        
        # Exit on 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean termination
    if is_dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    print("--- Gesture Mouse System Disengaged safely. Goodbye! ---")

if __name__ == "__main__":
    main()
