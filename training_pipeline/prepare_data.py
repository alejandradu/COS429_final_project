# Functions to process image and video folders, and standardize to extract features
# some code snippets are from
# https://www.kaggle.com/code/szescszesc/yoga-positions-acc-96

import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageOps
import cv2

def pad_image_to_array(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Calculate the padding to make the image 256x256
    desired_size = 256
    
    ### replaced with library
    # old_size = img.size  # old_size[0] is the width, old_size[1] is the height
    
    # # resize image to 256x256 but keep the aspect ratio
    # ratio = float(desired_size) / max(old_size)
    # new_size = tuple([int(x * ratio) for x in old_size])
    # img = img.resize(new_size, Image.ANTIALIAS)
    
    # # Calculate the padding
    # delta_w = desired_size - new_size[0]
    # delta_h = desired_size - new_size[1]
    # padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    
    # # Create a new image with a white background
    # new_img = ImageOps.expand(img, padding, fill='white')
    
    # pad the image
    new_img = ImageOps.pad(image, (256,256))
    
    return np.array(new_img)

def parse_image_folder(directory):
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
        # img = img.resize((256, 256))  # Resize to 128x128
        # img = np.array(img)
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
    
    return X, labels, df

def parse_video_folder(directory):
    """
    TODO: REVISE THIS FUNCTION DONE BY COPILOT
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
    
    for i in range(len(df)):
        cap = cv2.VideoCapture(df['path'][i])
        if not cap.isOpened():
            print(f"Error: Cannot access the video {df['path'][i]}.")
            continue
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Unable to fetch the frame from video {df['path'][i]}.")
                break
            
            frame = cv2.resize(frame, (128, 128))
            X.append(frame)
            y.append(df['label'][i])
            
        cap.release()
        
    # Encode labels - instead of using the poorly formatted json file
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X, y, df
            