def run_trained_model(X):
  def extract_features(file_path, n_mfcc=20):
    audio, sr = librosa.load(file_path, sr=None)
    n_fft = min(1024, len(audio))

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Add delta and delta-delta MFCCs
    delta_mfcc = librosa.feature.delta(mfccs)
    delta_delta_mfcc = librosa.feature.delta(mfccs, order=2)

    delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
    delta_delta_mfcc_mean = np.mean(delta_delta_mfcc, axis=1)

    # Extract Tonnetz features
    harmonic = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)

    # Combine features
    features = np.concatenate((mfccs_mean, delta_mfcc_mean, delta_delta_mfcc_mean, tonnetz_mean))
    return features
  def load_dataset(X, Y):
    features = []
    labels = []

    for idx, file_path in enumerate(X):
        feature_vector = extract_features(file_path)
        features.append(feature_vector)
        labels.append(Y[idx])  # Use the corresponding label from Y

    return np.array(features), np.array(labels)

  # TODO: load your model weights
  def apply_class_specific_weighting(logreg_proba, svm_proba, rf_proba):
      logreg_weighted = logreg_proba.copy()
      svm_weighted = svm_proba.copy()
      rf_weighted = rf_proba.copy()

      logreg_weighted[:, [1]] *= 1.5  # Adjust weights for class 1
      rf_weighted[:, [4]] *= 1.2     # Adjust weights for class 4
      rf_weighted[:, [0, 1, 2, 3, 5, 6]] *= 0.1  # Adjust weights for other classes

      # Combine probabilities
      combined_proba = (logreg_weighted + svm_weighted + rf_weighted) / 2
      return combined_proba
  def download_model_weights():
    import gdown
    import joblib
    # URLs for the files
    scaler_url = "https://drive.google.com/uc?id=18B3QqU6amN1Tc1ZntzDcTKiH2EM3Ald7"
    selector_url = "https://drive.google.com/uc?id=1vEu0SDD6oqeUdBi1i1PILVZ-ognyk-m0"
    logreg_url = "https://drive.google.com/uc?id=1e03qgcfvdhi9y2aNHVyAvk6JpBfoTmBZ"
    svm_url = "https://drive.google.com/uc?id=11LX0UaLsuNnBQk0t304dJGf2Kjpgbpcy"
    rf_url = "https://drive.google.com/uc?id=1tYPLYvVecpd2pVOkuQP4QOzdFq0llKXY"
    scaler_2_url = "https://drive.google.com/uc?id=1mSu7HkFWc9jz8O71z8_0hlVPJCOwGn1B"
    # Destination paths
    scaler_path = "scaler_53.pkl"
    selector_path = "selector_53.pkl"
    logreg_path = "logreg_model.pkl"
    svm_path = "svm_model.pkl"
    rf_path = "rf_model.pkl"
    scaler_2_path = "scaler_2.pkl"
    # Download files
    gdown.download(scaler_url, scaler_path, quiet=False)
    gdown.download(selector_url, selector_path, quiet=False)
    gdown.download(logreg_url, logreg_path, quiet=False)
    gdown.download(svm_url, svm_path, quiet=False)
    gdown.download(rf_url, rf_path, quiet=False)
    gdown.download(scaler_2_url, scaler_2_path, quiet=False)
    #load
    scaler = joblib.load("scaler_53.pkl")
    selector = joblib.load("selector_53.pkl")
    logreg = joblib.load("logreg_model.pkl")
    svm = joblib.load("svm_model.pkl")
    rf = joblib.load("rf_model.pkl")
    scaler_2 = joblib.load("scaler_2.pkl")
    return scaler, selector, logreg, svm, rf, scaler_2

    #Feature Selection
        # Process directories
  X_sample = []
  Y_sample = []

  for idx, subfolder in enumerate(os.listdir(base_dir)):
      subfolder_path = os.path.join(base_dir, subfolder)

      if os.path.isdir(subfolder_path):
          # Get file paths and labels for this subfolder
          X_subfolder = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.wav')]
          Y_subfolder = [CLASS_TO_LABEL[subfolder]] * len(X_subfolder)

          # Load features and labels
          features, labels = load_dataset(X_subfolder, Y_subfolder)

          X_sample.append(features)
          Y_sample.append(labels)

  # Combine all subfolder data
  X_sample = np.concatenate(X_sample, axis=0)
  Y_sample = np.concatenate(Y_sample, axis=0)
  scaler, selector, logreg, svm, rf, scaler_2 = download_model_weights()
  X_sample_scaled = scaler.transform(X_sample)
  X_sample_selected = selector.transform(X_sample_scaled)
  class_6_indices = np.where(Y_sample == 6)[0]
  X_sample_selected[class_6_indices, 5] = 0
  #Prediction
  X_new_scaled = scaler_2.transform(X_sample_selected)
  # Get predicted probabilities
  logreg_proba = logreg.predict_proba(X_new_scaled)
  svm_proba = svm.predict_proba(X_new_scaled)
  rf_proba = rf.predict_proba(X_new_scaled)
  # Apply class-specific weighting
  combined_proba = apply_class_specific_weighting(logreg_proba, svm_proba, rf_proba)
  # Make final predictions based on the highest probability
  Y_pred = np.argmax(combined_proba, axis=1)
  return Y_pred
