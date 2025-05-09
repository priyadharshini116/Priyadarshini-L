import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# Parameters
SAMPLE_RATE = 22050
DURATION = 1  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_MFCC = 13
MAX_LEN = 130  # Must match what you used during training

# Load the trained model
model = load_model('angry_model.h5')


def extract_features(audio_file, max_len=MAX_LEN):
    """
    Preprocess an audio file to extract padded MFCCs.
    """
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)

    # Pad or trim the signal to 1 second
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
    else:
        y = y[:SAMPLES_PER_TRACK]

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC)
    mfcc = mfcc.T  # shape: (time_steps, n_mfcc)

    # Pad or truncate MFCC to fixed length
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len]

    return mfcc


def predict_angry(audio_file):
    """
    Predict whether the given audio contains an angry voice.
    """
    try:
        features = extract_features(audio_file)  # shape: (130, 13)
        features = np.reshape(features, (1, MAX_LEN, NUM_MFCC))  # model expects (1, 130, 13)

        # Predict
        prediction = model.predict(features)

        if prediction > 0.5:
            print(" No Angry voice detected!")
        else:
            print("angry voice detected.")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")


# Test the function with an audio file
if __name__ == "__main__":
    audio_file = "test.wav"  # Replace with your test file
    predict_angry(audio_file)
