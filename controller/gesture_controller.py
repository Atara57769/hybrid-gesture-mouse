import cv2
import pyautogui
from utils.logger import get_logger
from state_machine.gesture_state import GestureState
from tracking.hand_tracker import HandTracker
from tracking.normalization import normalize_landmarks
from tracking.coordinate_mapper import CoordinateMapper
from classification.gesture_classifier import GestureClassifier
from classification.vote_stabilizer import VoteStabilizer
from state_machine.mouse_state_machine import MouseStateMachine
from ui.hud_renderer import HudRenderer
from utils.fps_counter import FPSCounter

logger = get_logger("gesture_controller")

class GestureController:
    """
    Core engine orchestrator that drives the real-time processing loop.
    Connects hand tracking, classification, vote stabilization, coordinate mapping,
    OS actions state machine, and OpenCV visual rendering components.
    """
    def __init__(self, 
                 mouse_service, 
                 gesture_classifier: GestureClassifier,
                 smoothing: float = 0.25,
                 confidence: float = 0.75,
                 history_size: int = 7,
                 debounce: float = 0.4,
                 scroll_sens: float = 1.5,
                 scroll_step: int = 2):
                 
        self.mouse_service = mouse_service
        self.classifier = gesture_classifier
        self.smoothing = smoothing
        self.confidence = confidence
        
        # Instantiate child layer components
        self.hand_tracker = HandTracker()
        self.vote_stabilizer = VoteStabilizer(window_size=history_size)
        
        screen_width, screen_height = self.mouse_service.get_screen_size()
        self.coordinate_mapper = CoordinateMapper(
            screen_width=screen_width, 
            screen_height=screen_height, 
            smoothing=smoothing
        )
        
        self.state_machine = MouseStateMachine(
            mouse_service=mouse_service, 
            debounce=debounce, 
            scroll_sens=scroll_sens, 
            scroll_step=scroll_step
        )
        
        self.hud_renderer = HudRenderer()
        self.fps_counter = FPSCounter()
        
    def run(self) -> None:
        """Starts and runs the real-time webcam frame acquisition and orchestration loop."""
        # Query current mouse location to seed the active smoother coordinates
        init_x, init_y = self.mouse_service.get_position()
        self.coordinate_mapper.reset(init_x, init_y)
        
        screen_width, screen_height = self.mouse_service.get_screen_size()
        logger.info(f"Monitor resolution loaded: {screen_width}x{screen_height}")
        
        # Open camera stream
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error: Could not open the webcam stream.")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        logger.info("Hybrid Gesture Mouse Control Engine Engaged.")
        logger.info("Place hand inside yellow active boundary box to begin control.")
        logger.info("Move hand/cursor to any screen corner or press 'Q' to terminate.")
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue
                    
                # Mirror frame horizontally for standard visual parity
                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape
                
                # Fetch performance speed metrics
                fps = self.fps_counter.tick()
                
                # Process hand landmarks via tracker
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_landmarks = self.hand_tracker.detect(rgb_frame)
                
                predicted_state = GestureState.IDLE
                prediction_prob = 1.0
                raw_x, raw_y = None, None
                
                if hand_landmarks:
                    # Draw skeletal hand joints neon overlay
                    self.hud_renderer.draw_hand_landmarks(frame, hand_landmarks)
                    
                    # Target index finger tip (landmark 8)
                    index_tip = hand_landmarks.landmark[8]
                    raw_x = int(index_tip.x * width)
                    raw_y = int(index_tip.y * height)
                    
                    # Normalize landmarks coordinates
                    normalized_coords = normalize_landmarks(hand_landmarks)
                    if normalized_coords is not None:
                        # Query ML model prediction
                        predicted_state, prediction_prob = self.classifier.predict(normalized_coords)
                        
                        # Apply confidence criteria thresholds
                        if prediction_prob >= self.confidence:
                            self.vote_stabilizer.add_vote(predicted_state)
                            
                    # Retrieve stable state via majority voting
                    stabilized_state = self.vote_stabilizer.get_stabilized_state()
                else:
                    stabilized_state = GestureState.IDLE
                    self.vote_stabilizer.clear()
                    
                # Interpolate and smooth coordinates mapping
                if raw_x is not None and raw_y is not None:
                    screen_x, screen_y = self.coordinate_mapper.map_and_smooth(
                        raw_x=raw_x, 
                        raw_y=raw_y, 
                        frame_width=width, 
                        frame_height=height
                    )
                else:
                    # Revert to last tracked smoothed coordinates
                    screen_x, screen_y = self.coordinate_mapper.prev_x, self.coordinate_mapper.prev_y
                    if screen_x is None or screen_y is None:
                        screen_x, screen_y = self.mouse_service.get_position()
                        self.coordinate_mapper.reset(screen_x, screen_y)
                        
                # Execute mapped OS command via state machine
                try:
                    self.state_machine.execute(stabilized_state, screen_x, screen_y)
                except pyautogui.FailSafeException:
                    logger.warning("Safety Fail-Safe Triggered! Extreme screen corner exit detected. Shutting down safely.")
                    break
                    
                # Draw the Heads-Up display telemetry overlay
                self.hud_renderer.draw_hud(
                    frame=frame,
                    state=stabilized_state,
                    confidence=prediction_prob,
                    screen_x=screen_x,
                    screen_y=screen_y,
                    fps=fps,
                    vote_history=self.vote_stabilizer.get_history()
                )
                
                # Show frame
                cv2.imshow("Hand Gesture Mouse - Live Controller", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Terminate loop and clean up all opened system utilities
            self.state_machine.shutdown()
            self.hand_tracker.close()
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Hybrid Gesture Control Engine disengaged successfully. Goodbye!")
