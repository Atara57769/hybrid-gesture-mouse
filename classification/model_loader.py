import pickle
import os
from utils.logger import get_logger

logger = get_logger("model_loader")

def load_pickle_model(model_path: str):
    """
    Utility helper to deserialize a Scikit-Learn classifier from a pickle file.
    """
    logger.info(f"Loading gesture classification model: {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model file not found at: '{model_path}'")
    with open(model_path, 'rb') as f:
        return pickle.load(f)
