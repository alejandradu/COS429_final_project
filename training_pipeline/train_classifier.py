# train different classifiers on still image datasets

import numpy as np
import pickle
from prepare_data import parse_image_folder
from extract_features import *
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# PARAMETERS
# detection confidence  -  check from realtime detection
# tracking confidence  -  check from realtime detection
# presence confidence  -  check from realtime detection
# CLASS BALANCING: smote k neighbors
# CLASSIFIER: random forest - n_estimators

seed = 333
# still image dataset path
directory = "../image_data"

# ------ get standardized images and labels ----- #
X, labels, df = parse_image_folder(directory)

# -----  extract landmarks: MEDIAPIPE  ------- #
mp_model_path = "../pretrained_models/pose_landmarker_full.task"
detector = mediapipe_detector(mp_model_path,
                              min_pose_detection_confidence=0.5,
                              min_pose_tracking_confidence=0.5,
                              min_pose_presence_confidence=0.5,)

x_data = np.zeros((len(X), MP_N_LANDMARKS * 4))  # For pose landmarks
y_data = np.zeros(labels.shape)  # For labels

# run inference on every image
for i in range(len(X)):
    landmarks = mediapipe_detect(detector, X[i])
    if landmarks is not None:
        x_data[i], y_data[i] = mediapipe_format_landmark(landmarks, labels[i])

# drop images with no landmarks detected
non_zero_mask = np.any(x_data != 0, axis=1)
x_data = x_data[non_zero_mask]
y_data = y_data[y_data != 0]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, shuffle=True, random_state=seed
)

# -------  class balancing ------- # 
smote_k_neighbors = 3   
smote = SMOTE(random_state=seed, k_neighbors=smote_k_neighbors)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# save class distributions before and after rebalancing
y_before_rebalance = pd.Series(y_train).value_counts()
y_after_rebalance = pd.Series(y_resampled).value_counts()
with open(f'reabalancing_analysis/SMOTE_{smote_k_neighbors}.pkl', 'wb') as f:
    pickle.dump((y_before_rebalance, y_after_rebalance), f)

# -------  train classifier ------- # 
forest_n_estimators = 200
forest = RandomForestClassifier(n_estimators=forest_n_estimators, random_state=seed, verbose=1)
forest.fit(X_resampled, y_resampled)

# COULD ALSO HAVE BAYES, SVM, NEURAL NETWORK

# save the model
with open(f'../trained_classifiers/randomForest_{forest_n_estimators}.pkl', 'wb') as f:
    pickle.dump(forest, f)
    
# -------  evaluate model ------- #
y_pred = forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Mediapipe Model Accuracy (with SMOTE): {accuracy}")
