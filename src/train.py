import os
import pickle
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

from config import *
from model import build_model, save_model_summary

def setup_gpu():
    print("--- Configuring Hardware Sandbox ---")
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        try:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(physical_gpus)} GPU(s). Memory growth activated to defend VRAM limits.")
        except RuntimeError as e:
            print(f"Error configuring GPU memory growth: {e}")
    else:
        print("No physical GPUs detected. Proceeding with CPU.")

def load_data():
    print("--- Loading Serialized Processed Data ---")
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'class_weights.pkl'), 'rb') as f:
        class_weights = pickle.load(f)
        
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"Loaded class weights: {class_weights}")
    
    return X_train, y_train, X_val, y_val, class_weights

def train():
    setup_gpu()
    
    # 1. Load Data
    X_train, y_train, X_val, y_val, class_weights = load_data()
    
    # 2. Build Model
    print("\n--- Initializing CNN Architecture ---")
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), num_classes=NUM_CLASSES)
    
    # Save summary
    summary_path = os.path.join(PROCESSED_DATA_DIR, 'model_summary.txt')
    save_model_summary(model, summary_path)
    
    # 3. Compile Model
    print("\n--- Compiling Network ---")
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. Configure Callbacks
    model_dir = 'pipeline_audio/models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'speech_emotion_cnn.keras')
    
    csv_log_path = os.path.join(PROCESSED_DATA_DIR, 'training_log.csv')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1),
        CSVLogger(csv_log_path, separator=',', append=False)
    ]
    
    # 5. Execute Training Loop
    print("\n--- Executing Phase 4: Training Loop ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\nTraining Complete! Best model weights saved to: {model_path}")

if __name__ == '__main__':
    train()
