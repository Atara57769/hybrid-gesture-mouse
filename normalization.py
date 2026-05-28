import numpy as np

def normalize_landmarks(hand_landmarks):
    """
    Normalizes 21 MediaPipe hand landmarks to make them translation and scale invariant.
    
    Translation Invariance:
        Centers the hand by shifting all landmarks so that the wrist (landmark 0) is at (0, 0, 0).
        
    Scale Invariance:
        Calculates the Euclidean distance between the wrist (landmark 0) and middle finger MCP (landmark 9).
        Divides all centered landmarks by this distance.
        
    Args:
        hand_landmarks: The MediaPipe Hand landmark object containing 21 landmarks.
        
    Returns:
        A list of 63 float values representing the normalized (x, y, z) coordinates,
        or None if land_landmarks is empty/invalid.
    """
    if not hand_landmarks or not hasattr(hand_landmarks, 'landmark') or len(hand_landmarks.landmark) < 21:
        return None
        
    # Extract raw coordinates into a NumPy array of shape (21, 3)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    # 1. Translation Invariance: Shift wrist (landmark 0) to (0, 0, 0)
    wrist = coords[0]
    centered_coords = coords - wrist
    
    # 2. Scale Invariance: Compute distance between wrist (0) and middle finger MCP (9)
    # Landmark 9 is the middle finger MCP joint, which is very stable relative to the wrist
    mcp_middle = centered_coords[9]
    scale = np.linalg.norm(mcp_middle)
    
    # Avoid division by zero
    if scale == 0:
        scale = 1.0
        
    normalized_coords = centered_coords / scale
    
    # Flatten the (21, 3) array into a 1D list of 63 values
    return normalized_coords.flatten().tolist()
