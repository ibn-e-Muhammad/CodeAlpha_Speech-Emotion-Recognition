import os
import tensorflow as tf
from keras import layers, models, Input
from config import *

# Assuming 16000 SR, 3.0 duration, 512 hop_length -> 94 steps
TIME_STEPS = 94
CHANNELS = 1

def build_model(input_shape=(N_MFCC, TIME_STEPS, CHANNELS), num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape)
    
    # Convolutional Stack 1
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Convolutional Stack 2
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Convolutional Stack 3
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.30)(x)
    
    # Dense Classification Head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Speech_Emotion_CNN")
    return model

def save_model_summary(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved to {save_path}")

if __name__ == "__main__":
    model = build_model()
    model.summary()
    save_path = os.path.join(PROCESSED_DATA_DIR, 'model_summary.txt')
    save_model_summary(model, save_path)
