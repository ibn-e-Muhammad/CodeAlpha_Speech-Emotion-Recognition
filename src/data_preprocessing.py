import os
import gc
import librosa
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
from config import *

def run_eda_check():
    print("--- Running EDA Sanity Check ---")
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.wav')]
    if not files:
        print("No audio files found!")
        return
        
    sample_files = files[:3]
    for f in sample_files:
        path = os.path.join(RAW_DATA_DIR, f)
        parts = f.replace('.wav', '').split('_')
        actor_id, sentence, emotion, intensity = parts
        
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        duration = len(y) / sr
        
        print(f"File: {f}")
        print(f"  Actor ID: {actor_id}, Emotion: {emotion}")
        print(f"  Sample Rate: {sr} Hz, Duration: {duration:.2f} s")
        print(f"  Waveform stats -> Min: {np.min(y):.4f}, Max: {np.max(y):.4f}, Mean: {np.mean(y):.4f}\n")

def build_speaker_splits():
    print("--- Building Speaker-Independent Splits ---")
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.wav')]
    
    # Collect unique actors
    actors = list(set(f.split('_')[0] for f in files))
    actors.sort()
    
    # Shuffle with fixed seed
    random.seed(RANDOM_SEED)
    random.shuffle(actors)
    
    # We have 91 actors.
    # 73 train, 9 val, 9 test
    train_actors = set(actors[:73])
    val_actors = set(actors[73:82])
    test_actors = set(actors[82:])
    
    print(f"Total Actors: {len(actors)}")
    print(f"Train Actors ({len(train_actors)}): {sorted(list(train_actors))[:5]}...")
    print(f"Val Actors ({len(val_actors)}): {sorted(list(val_actors))}")
    print(f"Test Actors ({len(test_actors)}): {sorted(list(test_actors))}\n")
    
    # Group files by partition
    train_files, val_files, test_files = [], [], []
    for f in files:
        actor_id = f.split('_')[0]
        if actor_id in train_actors:
            train_files.append(f)
        elif actor_id in val_actors:
            val_files.append(f)
        else:
            test_files.append(f)
            
    print(f"Train Files: {len(train_files)}")
    print(f"Val Files: {len(val_files)}")
    print(f"Test Files: {len(test_files)}\n")
    
    return train_files, val_files, test_files

def extract_features(file_list, desc="Processing"):
    X, y = [], []
    max_len = int(SAMPLE_RATE * DURATION)
    
    for i, f in enumerate(file_list):
        path = os.path.join(RAW_DATA_DIR, f)
        
        # Load audio
        waveform, sr = librosa.load(path, sr=SAMPLE_RATE)
        
        # Pad or truncate
        if len(waveform) > max_len:
            waveform = waveform[:max_len]
        else:
            waveform = np.pad(waveform, (0, max_len - len(waveform)), 'constant')
            
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        
        # Get emotion label
        emotion = f.split('_')[2]
        label = EMOTION_MAP[emotion]
        
        X.append(mfcc)
        y.append(label)
        
        if (i + 1) % 500 == 0:
            print(f"{desc}: Processed {i + 1}/{len(file_list)} files.")
            gc.collect()
            
    return np.array(X), np.array(y)

def save_processed_data(train_files, val_files, test_files):
    print("--- Extracting Train Features ---")
    X_train, y_train = extract_features(train_files, "Train")
    
    print("--- Extracting Val Features ---")
    X_val, y_val = extract_features(val_files, "Val")
    
    print("--- Extracting Test Features ---")
    X_test, y_test = extract_features(test_files, "Test")
    
    # Print original shapes
    print(f"\nOriginal Shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Scaling
    print("\n--- Applying Z-Score Normalization ---")
    N_train, num_mfcc, time_steps = X_train.shape
    
    # Reshape for scaling: (N * time_steps, num_mfcc)
    X_train_reshaped = np.transpose(X_train, (0, 2, 1)).reshape(-1, num_mfcc)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train = X_train_scaled.reshape(N_train, time_steps, num_mfcc).transpose(0, 2, 1)
    
    N_val = X_val.shape[0]
    X_val_reshaped = np.transpose(X_val, (0, 2, 1)).reshape(-1, num_mfcc)
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_val = X_val_scaled.reshape(N_val, time_steps, num_mfcc).transpose(0, 2, 1)
    
    N_test = X_test.shape[0]
    X_test_reshaped = np.transpose(X_test, (0, 2, 1)).reshape(-1, num_mfcc)
    X_test_scaled = scaler.transform(X_test_reshaped)
    X_test = X_test_scaled.reshape(N_test, time_steps, num_mfcc).transpose(0, 2, 1)
    
    # Add channel dimension for CNN
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    print(f"Scaled Shapes (with channel):")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    
    # Class weights for NEU imbalance
    print("\n--- Computing Class Weights ---")
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {classes[i]: weights[i] for i in range(len(classes))}
    print(f"Class Weights: {class_weights}")
    
    # Save arrays
    print("\n--- Saving Arrays to Disk ---")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(os.path.join(PROCESSED_DATA_DIR, 'class_weights.pkl'), 'wb') as f:
        pickle.dump(class_weights, f)
        
    print("Preprocessing Phase 2 Complete.")

if __name__ == '__main__':
    run_eda_check()
    train_files, val_files, test_files = build_speaker_splits()
    save_processed_data(train_files, val_files, test_files)
