# Functions with different pre-trained models to extract
# features (body keypoints/pose landmarks) from image/video/live_stream

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

# constants
MP_N_LANDMARKS = 33
MVNET_N_LANDMARKS = 17

# Follows the OutputListener interface in mp
# Way to retrieve detection results asynchronously - reduces latency
def mediapipe_listener(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    #print('pose landmarker result: {}'.format(result))
    return result

def mediapipe_detector(model_path, 
              video=False, 
              live_stream=False, 
              min_pose_detection_confidence=0.5,
              min_tracking_confidence=0.5, 
              min_pose_presence_confidence=0.5,):
    
    # create a landmarker model
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode 
    
      # fine-tuning the model
    if video:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence)
    elif live_stream:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            result_callback = mediapipe_listener)
    else:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence)

    # return initialized landmarker
    return PoseLandmarker.create_from_options(options)

def mediapipe_detect(detector, frame, video=False, video_fps=30, live_stream=False):
    """frame must be a numpy array from OpenCV, corresponding to an RGB image"""
    # NOTE: guarantee this for video/livestream?? (already for images)
    # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
    # returns landmark object - use format_landmarks in utils.py
    # frame must be a np.array in C
    
    # Load the input image from a numpy array.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    if video:
        return detector.detect_for_video(mp_image, video_fps)
    elif live_stream:
        # results accessible via the result callback
        detector.detect_async(mp_image, frame_timestamp_ms)
    else:
        return detector.detect(mp_image)
    
def mediapipe_format_landmark(landmarker_result, encoded_label):
    """
    Format the landmark data for a single frame detection, to input into the classfier.

    Args:
        landmarks: a PoseLandmarkerResult object. Must not be None.
                   comes from detector if image/video or listener if live_stream
        encoded_label: sklearn.preprocessing.LabelEncoder object corresponding to that frame

    Returns:
        X: a flattened numpy array of shape (1, MP_N_LANDMARKS * 4) containing the x, y, z, and visibility of each landmark.
        y: the encoded label of the frame, adjusted for accuracy function
    """
    X = np.zeros((MP_N_LANDMARKS, 4))   
    
    # returns List<List<NormalizedLandmark>> for some reason???
    landmark_list = landmarker_result.pose_landmarks[0]
    
    # extract list of landmarks for 33 locations
    j = 0
    for landmark in landmark_list:
        # if j == 1 or j == 2 or j == 3 or j == 4 or j == 5 or j == 6:
        #     continue
        x = landmark.x
        y = landmark.y
        z = landmark.z
        v = landmark.visibility
        X[j] = [x, y, z, v]
        j += 1
        
    # update the label to match accuracy function
    y = encoded_label + 1
    # flatten the array
    X = X.reshape(1, MP_N_LANDMARKS * 4)
    
    return X, y
