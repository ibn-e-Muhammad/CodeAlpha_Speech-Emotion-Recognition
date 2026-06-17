# System Static Constants
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 40
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 6
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Emotion Dictionary
EMOTION_MAP = {
    'ANG': 0,
    'DIS': 1,
    'FEA': 2,
    'HAP': 3,
    'NEU': 4,
    'SAD': 5
}
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

# Data Paths
RAW_DATA_DIR = 'pipeline_audio/data/raw'
PROCESSED_DATA_DIR = 'pipeline_audio/data/processed'

RANDOM_SEED = 42
