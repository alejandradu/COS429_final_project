# Functions with different pre-trained models to extract
# features (body keypoints/pose landmarks) from image/video/live_stream

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display
from collections import deque

# constants
MP_N_LANDMARKS = 33
MVNET_N_LANDMARKS = 17

# object to retrieve the landmark results asynchronously
# follows the OutputListener interface
class MediapipeListener():
    def __init__(self, max_results=100):
        self.results = deque(maxlen=max_results)

    # this is the function the callback will call
    def run(self, result, input):
        self.results.append(result)

    def get(self):
        return self.results.pop()

# Follows the OutputListener interface in mp
# # Way to retrieve detection results asynchronously - reduces latency
# def mediapipe_listener(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     #print('pose landmarker result: {}'.format(result))
#     return result

def mediapipe_detector(model_path, 
              video=False, 
              live_stream=False, 
              listener: MediapipeListener=None,
              min_pose_detection_confidence=0.5,
              min_tracking_confidence=0.5, 
              min_pose_presence_confidence=0.5,):
    """_summary_

    Args:
        model_path (_type_): _description_
        video (bool, optional): _description_. Defaults to False.
        live_stream (bool, optional): _description_. Defaults to False.
        listener (_type_, optional): Required if in live_stream
        min_pose_detection_confidence (float, optional): _description_. Defaults to 0.5.
        min_tracking_confidence (float, optional): _description_. Defaults to 0.5.
        min_pose_presence_confidence (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    
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
            result_callback = listener)
    else:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence)

    # return initialized landmarker
    return PoseLandmarker.create_from_options(options)

def mediapipe_detect(detector, frame, video=False, video_fps=30, live_stream=False, frame_timestamp_ms=0):
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
    
def mediapipe_format_landmark(landmarker_result, encoded_label=None):
    """
    Format the landmark data for a single frame detection, to input into the classfier.

    Args:
        landmarks: a PoseLandmarkerResult object. Must not be None.
                   comes from detector if image/video or listener if live_stream
        encoded_label: sklearn.preprocessing.LabelEncoder object corresponding to that frame
                       don't pass in if running inference on a live stream

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
       
    # flatten the array
    X = X.reshape(1, MP_N_LANDMARKS * 4)
         
    if encoded_label:
        # update the label to match accuracy function
        y = encoded_label + 1
        return X, y
    else:
        return X

# from the Gemini API

def draw_landmarks_on_image(rgb_image, detection_result=None, landmark_coords=None):
    """Can take in the landmark result object from MP detector or 
    the coordinates directly as an array (landmark, xyz coords)"""
    
    annotated_image = np.copy(rgb_image)

    # can also give the coords directly after processing
    if landmark_coords is not None:
        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark_coords[i,0], y=landmark_coords[i,1], z=landmark_coords[i,2]) for i in range(landmark_coords.shape[0])
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          pose_landmarks_proto,
          solutions.pose.POSE_CONNECTIONS,
          solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image
    
    pose_landmarks_list = detection_result.pose_landmarks

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx] 
        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          pose_landmarks_proto,
          solutions.pose.POSE_CONNECTIONS,
          solutions.drawing_styles.get_default_pose_landmarks_style())
        
    return annotated_image