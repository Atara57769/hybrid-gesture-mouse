import os
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Class mapping for reporting
CLASSES = {
    0: "idle",
    1: "move",
    2: "click",
    3: "drag",
    4: "scroll"
}

def generate_synthetic_data(filepath, num_samples_per_class=100):
    """
    Generates a synthetic, distinct dataset of normalized hand features to verify 
    the compilation, training, and loading flow without webcam recordings.
    """
    print(f"Generating synthetic dataset at '{filepath}'...")
    
    np.random.seed(42)
    data = []
    
    # 63 features representing 21 landmarks (x, y, z)
    num_features = 63
    
    for label in CLASSES.keys():
        # Create a unique 'base' posture for each class
        base_vector = np.zeros(num_features)
        
        if label == 0:    # Idle (Relaxed hand)
            base_vector[::3] = np.linspace(0.0, 0.5, 21) # gradual x spread
            base_vector[1::3] = np.linspace(0.0, 0.8, 21) # fingers extended straight
        elif label == 1:  # Move (Index extended, others closed)
            base_vector[8*3 + 1] = -1.0 # index finger tip high up
            base_vector[12*3 + 1] = 0.2  # middle closed
            base_vector[16*3 + 1] = 0.2  # ring closed
            base_vector[20*3 + 1] = 0.2  # pinky closed
        elif label == 2:  # Click (Index & Thumb tip pinched close)
            # Thumb tip (4) and Index tip (8) are extremely close
            base_vector[4*3:4*3+3] = [0.1, -0.4, 0.0]
            base_vector[8*3:8*3+3] = [0.12, -0.38, 0.0]
        elif label == 3:  # Drag (Fist / Pinch & Hold)
            base_vector[4*3:4*3+3] = [0.1, -0.3, 0.0]
            base_vector[8*3:8*3+3] = [0.11, -0.29, 0.0]
            base_vector[12*3:12*3+3] = [0.08, -0.28, 0.0] # other fingers also close
        elif label == 4:  # Scroll (Index & Middle finger extended)
            base_vector[8*3 + 1] = -1.0  # index up
            base_vector[12*3 + 1] = -0.9 # middle up
            base_vector[16*3 + 1] = 0.2  # ring closed
            base_vector[20*3 + 1] = 0.2  # pinky closed
            
        # Generate samples by adding Gaussian noise to the base vector
        for _ in range(num_samples_per_class):
            noise = np.random.normal(0, 0.05, num_features)
            sample = base_vector + noise
            data.append([label] + sample.tolist())
            
    # Write to CSV
    header = ['label'] + [f'feat_{i}' for i in range(num_features)]
    df = pd.DataFrame(data, columns=header)
    df.to_csv(filepath, index=False)
    print(f"Synthetic dataset saved with {len(df)} samples.")

def train_model(csv_path, model_path):
    """
    Loads dataset, trains a Random Forest Classifier, prints evaluation reports, 
    and serializes the trained model.
    """
    if not os.path.exists(csv_path):
        print(f"\n[Error] Dataset file '{csv_path}' not found!")
        print("Please run 'python collect_data.py' to record your custom gestures first,")
        print("or run 'python train.py --synthetic' to create a dummy test dataset.\n")
        return False
        
    print(f"Loading dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    # Check data distribution
    class_counts = df['label'].value_counts().to_dict()
    print("\nData distribution by class:")
    for code, name in CLASSES.items():
        count = class_counts.get(code, 0)
        print(f"  Class {code} ({name.upper()}): {count} samples")
        
    # Check if we have enough data to split
    min_samples = min(class_counts.values()) if class_counts else 0
    if len(class_counts) < 5 or min_samples < 5:
        print("\n[Warning] Some classes have very few samples.")
        print("For robust classification, collect at least 50+ samples per class.")
        
    # Extract features and targets
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    # 80/20 train-test split with stratification to keep equal class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(class_counts) >= 2 and min_samples >= 2 else None
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Initialize Random Forest Classifier
    # High estimators + balanced class weights for stability
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        random_state=42, 
        class_weight='balanced'
    )
    
    print("\nTraining Random Forest model...")
    clf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    
    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {accuracy * 100:.2f}%")
    
    # Classification Report
    target_names = [CLASSES[i].upper() for i in sorted(np.unique(y))]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Serialize model using Pickle
    print(f"\nSaving model to '{model_path}'...")
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    print("Model training complete and serialized successfully!\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hand Gesture Recognition Model")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic gesture data for testing")
    parser.add_argument("--dataset", type=str, default="gestures_dataset.csv", help="Path to gestures CSV file")
    parser.add_argument("--model", type=str, default="gesture_model.pkl", help="Path to save trained model")
    
    args = parser.parse_args()
    
    if args.synthetic:
        generate_synthetic_data(args.dataset)
        
    train_model(args.dataset, args.model)
