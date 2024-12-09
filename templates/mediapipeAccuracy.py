# Parse and clean the images, based on: https://www.kaggle.com/code/szescszesc/yoga-positions-acc-96

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

import mediapipe as mp
import matplotlib.pyplot as plt

# Seed for reproducibility
seed = 333

# Set your dataset root directory
root_dir = "data"  # Replace with your dataset path
# Step 1: Load image paths and labels
image_path = []
label = []

for dirname, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if not filename.endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
            continue  # Skip non-image files
        image_path.append(os.path.join(dirname, filename))
        label.append(os.path.basename(dirname))  # Use folder name as label

# Create a DataFrame
image_path = pd.Series(image_path, name='path')
label = pd.Series(label, name='label')
df = pd.concat([image_path, label], axis=1)


# Step 2: Load and preprocess images
labels = []
X = []

for i in range(len(df)):
    img = Image.open(df['path'][i])
    img = img.resize((128, 128))  # Resize to 128x128
    img = np.array(img)
    
    # Handle image formats
    if len(img.shape) > 2 and img.shape[2] == 4:  # Handle RGBA images
        img = img[:, :, :3]
    elif len(img.shape) < 3:  # Convert grayscale to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    if img.shape[2] == 3:  # Only keep valid RGB images
        X.append(img)
        labels.append(df['label'][i])

# Encode labels
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(labels)

# Step 3: Extract Mediapipe landmarks
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

x_position = np.zeros((len(X), 33, 4))  # For pose landmarks
y_position = np.zeros(label_encoded.shape)  # For labels

for i in range(len(X)):
    results = pose.process(X[i])
    if results.pose_landmarks is not None:
        positions = results.pose_landmarks.landmark
        j = 0
        for landmark in positions:
            x = landmark.x
            y = landmark.y
            z = landmark.z
            v = landmark.visibility
            x_position[i, j] = [x, y, z, v]
            y_position[i] = label_encoded[i] + 1  # +1 to match Kaggle's code
            j += 1

# Drop images with no landmarks detected
non_zero_mask = np.any(x_position != 0, axis=(1, 2))
x_position = x_position[non_zero_mask]
y_position = y_position[y_position != 0]

# Reshape landmarks to 2D
x_position = x_position.reshape(x_position.shape[0], x_position.shape[1] * x_position.shape[2])

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x_position, y_position, test_size=0.2, shuffle=True, random_state=seed
)

# Step 5: Apply SMOTE for class balancing
smote = SMOTE(random_state=seed, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 6: Train Random Forest Classifier
forest = RandomForestClassifier(n_estimators=200, random_state=seed, verbose=1)
forest.fit(X_resampled, y_resampled)

# SAVE THE MODEL
with open('randomForest.pkl','wb') as f:
    pickle.dump(forest,f)

# Step 7: Evaluate model
y_pred = forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Mediapipe Model Accuracy (with SMOTE): {accuracy}")
