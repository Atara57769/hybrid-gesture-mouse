import os
import argparse
from utils.logger import get_logger
from services import create_mouse_service
from config.settings import MODEL_PATH, SMOOTHING, CONFIDENCE, HISTORY_SIZE, CLICK_DEBOUNCE, SCROLL_SENSITIVITY, SCROLL_STEP
from classification.model_loader import load_pickle_model
from classification.gesture_classifier import GestureClassifier
from controller.gesture_controller import GestureController

logger = get_logger("main")

def main():
    parser = argparse.ArgumentParser(description="Real-Time Layered Hand Gesture Mouse Control")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to trained model pickle")
    parser.add_argument("--smoothing", type=float, default=SMOOTHING, help="EMA smoothing factor (0 = static, 1 = raw jittery)")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE, help="Min probability to accept predicted state changes")
    parser.add_argument("--history", type=int, default=HISTORY_SIZE, help="Queue size for majority voting filter")
    parser.add_argument("--debounce", type=float, default=CLICK_DEBOUNCE, help="Cooldown in seconds to trigger subsequent clicks")
    parser.add_argument("--scroll-sens", type=float, default=SCROLL_SENSITIVITY, help="Scroll vertical sensitivity multiplier")
    parser.add_argument("--scroll-step", type=int, default=SCROLL_STEP, help="Discrete scroll step size")
    
    args = parser.parse_args()
    
    # 1. Verify model file exists
    if not os.path.exists(args.model):
        logger.error(f"Trained model file '{args.model}' not found!")
        logger.info("Please train a model first using: python training/train.py")
        logger.info("Or generate a mock model for testing with: python training/train.py --synthetic")
        return
        
    try:
        # 2. Load ML Model
        raw_model = load_pickle_model(args.model)
        gesture_classifier = GestureClassifier(raw_model)
        
        # 3. Resolve OS Mouse Service dependency
        mouse_service = create_mouse_service()
        
        # 4. Instantiate and run the Orchestrator Controller
        controller = GestureController(
            mouse_service=mouse_service,
            gesture_classifier=gesture_classifier,
            smoothing=args.smoothing,
            confidence=args.confidence,
            history_size=args.history,
            debounce=args.debounce,
            scroll_sens=args.scroll_sens,
            scroll_step=args.scroll_step
        )
        
        controller.run()
    except Exception as e:
        logger.exception(f"Unhandled critical exception in main execution: {e}")

if __name__ == "__main__":
    main()
