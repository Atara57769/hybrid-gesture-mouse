import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import cv2
import os

# Import modules under test
import collect_data
import gesture_mouse

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class MockHandLandmarks:
    def __init__(self):
        self.landmark = [MockLandmark(0.5, 0.5, 0.0) for _ in range(21)]

class MockResults:
    def __init__(self, has_hand=True):
        self.multi_hand_landmarks = [MockHandLandmarks()] if has_hand else None

class TestCameraAndGUI(unittest.TestCase):
    
    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('collect_data.mp_hands.Hands')
    def test_collect_data_no_hand(self, mock_hands_class, mock_destroy, mock_wait_key, mock_imshow, mock_video_capture):
        """Tests collect_data.py loop when no hand is in frame."""
        # Setup mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, True, False] # Runs loop once then terminates
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        # Setup mock hands
        mock_hands_instance = MagicMock()
        mock_hands_instance.process.return_value = MockResults(has_hand=False)
        mock_hands_class.return_value = mock_hands_instance
        
        # Terminate immediately by returning 'q'
        mock_wait_key.return_value = ord('q')
        
        # Mock session counts and dataset write to avoid modifying files
        with patch('collect_data.load_existing_counts') as mock_counts, \
             patch('os.path.exists') as mock_exists:
            mock_counts.return_value = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            mock_exists.return_value = False
            
            collect_data.main()
            
        mock_video_capture.assert_called_once_with(0)
        mock_cap.read.assert_called()
        mock_imshow.assert_called()
        mock_destroy.assert_called_once()

    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('collect_data.mp_hands.Hands')
    @patch('collect_data.normalize_landmarks')
    def test_collect_data_recording_with_hand(self, mock_norm, mock_hands_class, mock_destroy, mock_wait_key, mock_imshow, mock_video_capture):
        """Tests collect_data.py loop while active recording a hand landmark."""
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, True, False]
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        mock_hands_instance = MagicMock()
        mock_hands_instance.process.return_value = MockResults(has_hand=True)
        mock_hands_class.return_value = mock_hands_instance
        
        mock_norm.return_value = [0.1] * 63
        
        # Return Space (toggle recording) on first iteration, then 'q' to quit
        mock_wait_key.side_effect = [ord(' '), ord('q')]
        mock_cap.isOpened.side_effect = [True, True, True, False]
        
        with patch('collect_data.load_existing_counts') as mock_counts, \
             patch('csv.writer') as mock_csv, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('os.path.exists') as mock_exists:
            mock_counts.return_value = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            mock_exists.return_value = True
            
            collect_data.main()
            
        mock_video_capture.assert_called_once_with(0)
        mock_imshow.assert_called()
        mock_destroy.assert_called_once()

    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('gesture_mouse.mp_hands.Hands')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_gesture_mouse_loop(self, mock_file, mock_pickle, mock_hands_class, mock_destroy, mock_wait_key, mock_imshow, mock_video_capture):
        """Tests gesture_mouse.py running loop and failsafe exit."""
        # Mock ML Model
        mock_model = MagicMock()
        # Predicting state 1 (MOVE) with 95% confidence
        mock_model.predict_proba.return_value = [[0.01, 0.95, 0.01, 0.01, 0.01, 0.01]]
        mock_pickle.return_value = mock_model
        
        # Mock Camera
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, True, False]
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        # Setup mock hands returns a hand landmark
        mock_hands_instance = MagicMock()
        mock_hands_instance.process.return_value = MockResults(has_hand=True)
        mock_hands_class.return_value = mock_hands_instance
        
        mock_wait_key.return_value = ord('q')
        
        # Mock MouseService
        mock_service = MagicMock()
        mock_service.get_screen_size.return_value = (1920, 1080)
        mock_service.get_position.return_value = (500, 500)
        
        from gesture_mouse import GestureMouseController
        controller = GestureMouseController(mock_service, "gesture_model.pkl")
        controller.run()
        
        mock_video_capture.assert_called_once_with(0)
        mock_cap.read.assert_called()
        mock_destroy.assert_called_once()
        # Verify that MouseService was queried and instructed
        mock_service.get_screen_size.assert_called()
        
if __name__ == '__main__':
    unittest.main()
