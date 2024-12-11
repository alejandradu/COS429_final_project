# Functions to process image and video folders, and standardize to extract features
# some code snippets are from
# https://www.kaggle.com/code/szescszesc/yoga-positions-acc-96

import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageOps
import cv2
import json


def parse_image_folder(directory, return_label_encoder=False):
    """
    Extract valid images from a directory and return the standardized images and labels.

    Args:
        directory: path to data folder.

    Returns:
        X: a list with valid128x128 RGB images of type nd.array(uint8)
        labels: array with encoded classes as a LabelEncoder() object
        df: dataframe with image paths and labels
    """
    image_path = []
    label = []
    
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if not filename.endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
                continue  # Skip non-image files
            image_path.append(os.path.join(dirname, filename))
            label.append(os.path.basename(dirname))  # Use folder name as label

    # Create a DataFrame
    image_path = pd.Series(image_path, name='path')
    label = pd.Series(label, name='label')
    df = pd.concat([image_path, label], axis=1)
    
    X = []
    y = []
    
    for i in range(len(df)):
        img = Image.open(df['path'][i])
        new_img = ImageOps.pad(img, (256,256))
        img = np.array(new_img)
        
        # Handle image formats
        if len(img.shape) > 2 and img.shape[2] == 4:  # Handle RGBA images
            img = img[:, :, :3]
        elif len(img.shape) < 3:  # Convert grayscale to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        if img.shape[2] == 3:  # Only keep valid RGB images
            # ensure that input to mp.Image is contiguous
            img = np.ascontiguousarray(img)
            X.append(img)
            y.append(df['label'][i])
    
    # Encode labels - instead of using the poorly formatted json file
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(y)
    
    if return_label_encoder:
        return X, labels, df, label_encoder
    
    return X, labels, df


def parse_video_folder(directory):
    """
    Extract valid videos from a directory and return the standardized videos and labels.

    Args:
        directory: path to data folder.

    Returns:
        X: a list with valid videos
        y: array with encoded classes as a LabelEncoder() object
        df: dataframe with video paths and labels
    """
    video_path = []
    # TODO: correct the labels to like a vector of labels or split into images
    # I guess this dataset is not for training but for testing so we can figure out the labels after
    
    X = []
    y = []
    
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if not filename.endswith(('.mp4', '.avi')):  # Only process video files
                continue  # Skip non-video files
            video_path.append(os.path.join(dirname, filename))
            label.append(os.path.basename(dirname))  # Use folder name as label

    # Create a DataFrame
    video_path = pd.Series(video_path, name='path')
    label = pd.Series(label, name='label')
    df = pd.concat([video_path, label], axis=1)
    
    # don't think we want to pre-feed frames
    # for i in range(len(df)):
    #     cap = cv2.VideoCapture(df['path'][i])
    #     if not cap.isOpened():
    #         print(f"Error: Cannot access the video {df['path'][i]}.")
    #         continue
        
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             print(f"Error: Unable to fetch the frame from video {df['path'][i]}.")
    #             break
            
    #         frame = cv2.resize(frame, (128, 128))
    #         X.append(frame)
    #         y.append(df['label'][i])
            
    #     cap.release()
        
    # Encode labels - instead of using the poorly formatted json file
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X, y, df
    