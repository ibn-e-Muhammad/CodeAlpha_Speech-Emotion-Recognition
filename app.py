import os
import numpy as np
import librosa
import pickle
import streamlit as st
import tensorflow as tf
from keras.models import load_model

# --- Configuration & Constants ---
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 40
MAX_LEN = int(SAMPLE_RATE * DURATION)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

MODEL_PATH = 'pipeline_audio/models/speech_emotion_cnn.keras'
SCALER_PATH = 'pipeline_audio/data/processed/scaler.pkl'

# --- Page Config ---
st.set_page_config(page_title="SER Studio", page_icon="🎙️", layout="centered")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import build_model

@st.cache_resource
def load_assets():
    # Build architecture from source and load raw weights to bypass 
    # Keras 2/3 'Functional' deserialization mismatches across OS limits.
    model = build_model()
    model.load_weights(MODEL_PATH)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    assets_loaded = False

# --- Audio Processing Pipeline ---
def process_audio(audio_bytes):
    """Internal signal-processing pipeline matching Phase 2"""
    # 1. Read audio buffer and resample to exactly 16000 Hz
    # Using librosa.load with a file-like object (audio_bytes)
    import io
    import soundfile as sf
    
    # Read bytes into numpy array
    y, sr = sf.read(io.BytesIO(audio_bytes))
    
    # If stereo, convert to mono
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    # Resample to 16000 Hz if needed
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        
    # 3. Pad or Truncate to 3.0 seconds (48,000 samples)
    if len(y) > MAX_LEN:
        y = y[:MAX_LEN]
    else:
        y = np.pad(y, (0, MAX_LEN - len(y)), 'constant')
        
    # 4. Extract 40 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    
    # 5. Apply Z-Score Normalization
    time_steps = mfcc.shape[1]
    # Reshape for scaling: (time_steps, 40)
    mfcc_reshaped = mfcc.T 
    mfcc_scaled = scaler.transform(mfcc_reshaped)
    # Revert back to (40, time_steps)
    mfcc_final = mfcc_scaled.T
    
    # 6. Expand Dimensions to (1, 40, 94, 1)
    tensor = np.expand_dims(mfcc_final, axis=0)  # batch dimension
    tensor = np.expand_dims(tensor, axis=-1)     # channel dimension
    
    return tensor

def display_predictions(prediction_probs):
    """Maps prediction vector and displays bar chart"""
    probs = prediction_probs[0]
    pred_idx = np.argmax(probs)
    predicted_emotion = EMOTION_LABELS[pred_idx]
    
    st.markdown(f"### Detected Emotion: **{predicted_emotion}**")
    
    # Format data for bar chart
    import pandas as pd
    chart_data = pd.DataFrame(
        {"Probability": probs},
        index=EMOTION_LABELS
    )
    st.bar_chart(chart_data)

# --- UI Layout ---
st.title("🎙️ Speech Emotion Recognition Studio")
st.markdown("Upload a `.wav` file or record directly from your microphone to analyze human emotion.")

if not assets_loaded:
    st.warning("Please ensure the model and scaler are generated via the Phase 1-4 pipeline before running the app.")
    st.stop()

tab1, tab2 = st.tabs(["📁 The Upload Studio", "🔴 The Live Recording Studio"])

with tab1:
    st.header("Upload Audio")
    uploaded_file = st.file_uploader("Choose a .wav file...", type=['wav', 'mp3', 'ogg'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        if st.button("Analyze Uploaded Audio"):
            with st.spinner("Processing audio matrix..."):
                try:
                    tensor = process_audio(uploaded_file.getvalue())
                    preds = model.predict(tensor)
                    display_predictions(preds)
                except Exception as e:
                    st.error(f"Error during processing: {e}")

with tab2:
    st.header("Live Microphone")
    audio_value = st.audio_input("Record a voice sample")
    
    if audio_value is not None:
        st.audio(audio_value)
        if st.button("Analyze Live Recording"):
            with st.spinner("Processing live audio matrix..."):
                try:
                    tensor = process_audio(audio_value.getvalue())
                    preds = model.predict(tensor)
                    display_predictions(preds)
                except Exception as e:
                    st.error(f"Error during processing: {e}")

st.markdown("---")
st.caption("Powered by CodeAlpha - Convolutional Signal Processing Pipeline")
