import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras.models import load_model

from config import *

def evaluate_model():
    print("--- Phase 5: Comprehensive Test Set Evaluation ---")
    
    # Paths
    model_path = os.path.join('pipeline_audio', 'models', 'speech_emotion_cnn.keras')
    x_test_path = os.path.join(PROCESSED_DATA_DIR, 'X_test.npy')
    y_test_path = os.path.join(PROCESSED_DATA_DIR, 'y_test.npy')
    plot_dir = os.path.join(PROCESSED_DATA_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Load Model and Data
    print(f"Loading Model from {model_path}...")
    model = load_model(model_path, compile=False)
    
    print(f"Loading Test Tensors...")
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    print(f"Test Set Shape: {X_test.shape}")
    
    # 2. Inference
    print("\n--- Running Full-Set Inference ---")
    y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 3. Classification Report
    print("\n--- Classification Report ---")
    report = classification_report(y_test, y_pred, target_names=EMOTION_LABELS, digits=4)
    print(report)
    
    # Save report to text file
    with open(os.path.join(PROCESSED_DATA_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    # 4. Confusion Matrix
    print("\n--- Generating Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTION_LABELS, 
                yticklabels=EMOTION_LABELS)
    plt.title('Speech Emotion Recognition - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = os.path.join(plot_dir, 'confusion_matrix_v3.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Confusion Matrix saved to: {plot_path}")
    print("Phase 5 Diagnostics Complete.")

if __name__ == '__main__':
    evaluate_model()
