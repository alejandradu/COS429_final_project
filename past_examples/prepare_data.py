# run mediapipe keypoint extraction on the image training dataset to get 
# a coordinate per pose dataset

# OR JUST USE THE PIPELINE WE ALREADY HAVE

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

location = 'data'
# create path object
image_dir = Path(location)

# get the labels
with open(image_dir / 'Poses.json', 'r') as f:
    labels_dict = json.load(f)
poses = labels_dict.get("Poses", [])
    
# Initialize lists to store file paths and labels
filepaths = []
labels = []

# Iterate over the folders in /data
for folder in image_dir.iterdir():
    if folder.is_dir():
        # Get the label for the current folder
        folder_name = folder.name
        label = labels_dict.get(folder_name, None)
        
        if label is not None:
            # Find all image files in the current folder
            image_files = list(folder.glob('**/*.JPG')) + \
                          list(folder.glob('**/*.jpg')) + \
                          list(folder.glob('**/*.PNG')) + \
                          list(folder.glob('**/*.png'))
            
            # Add the file paths and labels to the lists
            for image_file in image_files:
                filepaths.append(str(image_file))
                labels.append(label)

# Create a DataFrame with the file paths and labels
filepaths_series = pd.Series(filepaths, name='Filepath')
labels_series = pd.Series(labels, name='Label')

image_df = pd.concat([filepaths_series, labels_series], axis=1)

# Display the DataFrame
print(image_df.head())    