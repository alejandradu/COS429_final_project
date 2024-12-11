import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
from PIL import Image, ImageOps

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display
from collections import deque

class FeaturesMP():
    """Class to to extract formatted landmarks from images, video, and 
    livestream with the new MediaPipe API"""
    def __init__(self, model_path, image_size=(1080, 1920)):
        self.model_path = model_path
        self.detector = None
        self.live_view_result = None
        self.live_view_result_classifier = None
        self.n_landmarks = 33
        self.height, self.width = image_size
        
       # padding parameters for inference
        self.desired_length = max(image_size)
        self.pad_height_top = max(0, (self.desired_length - self.height) // 2)
        self.pad_height_bottom = max(0, self.desired_length - self.height - self.pad_height_top)
        self.pad_width_left = max(0, (self.desired_length - self.width) // 2)
        self.pad_width_right = max(0, self.desired_length - self.width - self.pad_width_left)
        self.padding = ((self.pad_height_top, self.pad_height_bottom), (self.pad_width_left, self.pad_width_right), (0, 0))
        
        
    # Follows the OutputListener interface in mp
    # Way to retrieve detection results asynchronously - reduces latency
    def live_view_listener(self, result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.live_view_result = result
        
        
    def live_view_listener_classifier(self, result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.live_view_result_classifier = result
        

    def init_detector(self, 
              video=False, 
              live_stream=False, 
              min_pose_detection_confidence=0.5,
              min_tracking_confidence=0.5, 
              min_pose_presence_confidence=0.5,
              double=False):

        # create a landmarker model
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode 

        # fine-tuning the model
        if video:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.VIDEO,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence)
        elif live_stream:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
                result_callback = self.live_view_listener)
        else:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.IMAGE,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence)

        # initialize detector
        self.detector = PoseLandmarker.create_from_options(options)
        
        if double:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
                result_callback = self.live_view_listener_classifier)
            self.detector_classifier = PoseLandmarker.create_from_options(options)


    def detect(self, frame, video=False, video_fps=30, live_stream=False, frame_timestamp_ms=0, double=False):
        """frame must be a numpy array from OpenCV, corresponding to an RGB image"""
        # NOTE: guarantee this for video/livestream?? (already for images)
        # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
        # returns landmark object - use format_landmarks in utils.py
        # frame must be a np.array in C
        # double if you are using a detector for both a live stream with any size
        # and inference, where we care about the size
        # square size if the max dimension for the image for the current camera to pad into a square

        if double:
            inference_image = np.pad(frame, self.padding, mode='constant', constant_values=0)
            mp_inference_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=inference_image)
            
        # Load the input image from a numpy array.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        if video:
            return self.detector.detect_for_video(mp_image, video_fps)
        
        elif live_stream:
            if double:
                self.detector_classifier.detect_async(mp_inference_image, frame_timestamp_ms)
                self.detector.detect_async(mp_image, frame_timestamp_ms)
                return self.live_view_result, self.live_view_result_classifier
            else:
                self.detector.detect_async(mp_image, frame_timestamp_ms)
                return self.live_view_result
            
        else:
            return self.detector.detect(mp_image)
        
    
    def format_landmark(self, landmarker_result, encoded_label=None, rot_invariant=False):
        """
        Format the landmark data for a single frame detection, to input into the classfier.

        Args:
            landmarks: a PoseLandmarkerResult object. Must not be None.
                    comes from detector if image/video or listener if live_stream
            encoded_label: sklearn.preprocessing.LabelEncoder object corresponding to that frame
                        don't pass in if running inference on a live stream

        Returns:
            X: a flattened numpy array of shape (1, self.n_landmarks * 4) containing the x, y, z, and visibility of each landmark.
            y: the encoded label of the frame, adjusted for accuracy function
        """
        X = np.zeros((self.n_landmarks, 4))   
        
        # returns List<List<NormalizedLandmark>> for some reason???
        landmark_list = landmarker_result.pose_landmarks[0]
        
        # extract list of landmarks for 33 locations
        j = 0
        for landmark in landmark_list:
            x = landmark.x
            y = landmark.y
            z = landmark.z
            v = landmark.visibility
            X[j] = [x, y, z, v]
            j += 1
            
        if rot_invariant:
            X = self.make_rot_invariant(X)

        # flatten the array
        X = X.reshape(1, self.n_landmarks * 4)
            
        # # catch bug was causing false with 0 label
        # if encoded_label is not None:
        #     # update the label to match accuracy function
        #     y = encoded_label + 1
        #     return X, y
        # else:
        #     return X
        y = encoded_label + 1
        return X, y


    def draw_landmarks_on_image(self, rgb_image, detection_result=None, landmark_coords=None):
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
    
            
    def make_rot_invariant(self, landmark_array, init_norm=False):
        # landmark_array is a np.array with landmarks as rows and coords + visibility as columns for a single pose
        # of shape n_landmarks x 4
        # init_norm if you want to renormalize like a square before rotating
        
        # NOTE: will adding this copy introduce too much lag??
        X = np.copy(landmark_array)
        
        if init_norm:
            square_size = max(self.height, self.width)
            # normalize as if in a square picture
            X[:, 0] = X[:, 0] * self.width/square_size
            X[:, 1] = X[:, 1] * self.height/square_size
            X[:, 2] = X[:, 2] * self.width/square_size
            
        # origin is landmark 24
        origin = X[24, 0:3]
        # translate all landmarks to new origin
        X[:, 0:3] = X[:, 0:3] - origin
        # get angle of vector between landmark 24 and 11 (shoulder) - likely to always be there
        vector = X[11, 0:3]
        angle1 = np.arctan2(vector[1], vector[0])
        angle2 = np.arctan2(vector[2], vector[0])
        # Rotation matrix around the x-axis
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(-angle1), -np.sin(-angle1)],
            [0, np.sin(-angle1), np.cos(-angle1)]
        ])
        # Rotation matrix around the y-axis
        rotation_matrix_y = np.array([
            [np.cos(-angle2), 0, np.sin(-angle2)],
            [0, 1, 0],
            [-np.sin(-angle2), 0, np.cos(-angle2)]
        ])
        # Combine the rotations
        rotation_matrix = np.dot(rotation_matrix_y, rotation_matrix_x)
        # Apply the rotation to all landmarks
        X[:, 0:3] = np.dot(X[:, 0:3], rotation_matrix)
        
        return X