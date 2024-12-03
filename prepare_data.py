# run mediapipe keypoint extraction on the image training dataset to get 
# a coordinate per pose dataset

import os
import pandas as pd
import numpy as np
from pathlib import Path

location = '/data'
# create path object
image_dir = Path(location)
print(image_dir)

# # Get filepaths and labels
# filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))

# labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

# filepaths = pd.Series(filepaths, name='Filepath').astype(str)
# labels = pd.Series(labels, name='Label')

# # Concatenate filepaths and labels into a dataframe
# image_df = pd.concat([filepaths, labels], axis=1)

# print(image_df.head())