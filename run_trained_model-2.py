# -*- coding: utf-8 -*-
"""run_trained_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13mlhum7C5BfOmT8vZAm7D_U0oTKTPU3b
"""

import joblib
import numpy as np
import json

def run_trained_model(X):

    def download_model_weights():
        import gdown

        # Use the direct download links
        logreg_url = 'https://drive.google.com/uc?id=1-0NnrSjeRaUeQHEvh1BcWXz3Sgdvu_au'
        svm_url = 'https://drive.google.com/uc?id=1-8t9PnxeIscZGLgVHMdo5P7DwBg1RFaS'
        rf_url = 'https://drive.google.com/uc?id=-5unXlrdSfeVeow-GGBvfki7i5TQEZLy'
        weights_url = 'https://drive.google.com/uc?id=19nn81lc83qLCgJ5n5gnh2J3eko9cRVw7'

        # Paths to save downloaded files
        logreg_path = "logreg_model.pkl"
        svm_path = "svm_model.pkl"
        rf_path = "rf_model.pkl"
        weights_path = "ensemble_weights.json"

        # Download files
        gdown.download(logreg_url, logreg_path, fuzzy=True)
        gdown.download(svm_url, svm_path, fuzzy=True)
        gdown.download(rf_url, rf_path, fuzzy=True)
        gdown.download(weights_url, weights_path, fuzzy=True)
        return logreg_path, svm_path, rf_path, weights_path

    # Step 1: Download the model weights
    logreg_path, svm_path, rf_path, weights_path = download_model_weights()

    # Step 2: Load the models and weights
    logreg = joblib.load(logreg_path)
    svm = joblib.load(svm_path)
    rf = joblib.load(rf_path)
    with open(weights_path, 'r') as f:
        ensemble_weights = json.load(f)

    logreg_weight = ensemble_weights['logreg']
    svm_weight = ensemble_weights['svm']
    rf_weight = ensemble_weights['rf']

    def extract_features(file_path):
        audio, sr = librosa.load(file_path, sr=None)
        n_fft = min(1024, len(audio))

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_fft=n_fft)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Add delta and delta-delta MFCCs
        delta_mfcc = librosa.feature.delta(mfccs)
        delta_mfcc_mean = np.mean(delta_mfcc, axis=1)

        # Extract Tonnetz features
        harmonic = librosa.effects.harmonic(audio)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)

        # Combine features
        features = np.concatenate((mfccs_mean, delta_mfcc_mean, tonnetz_mean))
        return features

    features = extract_features(X, )

    # Step 4: Standardize the features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Step 5: Get predictions from individual models
    logreg_proba = logreg.predict_proba(features_scaled)
    svm_proba = svm.predict_proba(features_scaled)
    rf_proba = rf.predict_proba(features_scaled)

    # Step 6: Combine probabilities using weights
    combined_proba = (
        logreg_proba * logreg_weight +
        svm_proba * svm_weight +
        rf_proba * rf_weight
    ) / 3

    # Step 7: Generate final predictions
    predictions = np.argmax(combined_proba, axis=1)
    return predictions