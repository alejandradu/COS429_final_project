"""Use Blazepose to detect 33 keypoints on image frames coming from real-time
video stream using opencv"""

import cv2
import mediapipe as mp
import pickle
from training_pipeline.extract_features import FeaturesMP
import numpy as np
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


# Retrieve pre-trained model
mp_model_path = "/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/pretrained_models/pose_landmarker_full.task"
# Initialize FeaturesMP object
features_mp = FeaturesMP(mp_model_path)
# Initialize detector
features_mp.init_detector(live_stream=True, min_pose_detection_confidence=0.7)
# load the label encoder
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/training_pipeline/label_encoder.pkl', 'rb') as f:    
    label_encoder = pickle.load(f)

# load the trained classifier
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/trained_classifiers/rf_pad.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

# Create a loop to read the latest frame from the camera
while cap.isOpened():
    ret, frame = cap.read()
    # timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    timestamp = int(round(time.time()*1000))
    
    # assess how different is it from 256 256??
    print(np.shape(frame))
    
    if not ret:
        print("Error: Unable to fetch the frame.")
        break

    # Run inference on the image - internally the listener returns the landmarks
    landmarks = features_mp.detect(frame, live_stream=True, frame_timestamp_ms=timestamp)
    
    # if too different need a way to resize it after detection to match
    # the dimensions of normalized coordinates that we trained on
    
    ## MODEL COME IN - process the keypoints detected and classify the pose
    ## and display it
    # PROBLEM HERE: SEEMS LIKE WE NEED 2 LANDMARK DETECTIONS 
    # ONE ON STANDARDIZED IMAGES AND THE OTHER ON THE REAL-TIME IMAGES

    # Draw landmarks if detected
    if landmarks is not None:  
        if len(landmarks.pose_landmarks) != 0:
            # draw landmarks
            # annotated_image = features_mp.draw_landmarks_on_image(frame, detection_result=landmarks) 
            # formatted_landmark = features_mp.format_landmark(landmarks)
            # # run inference
            # predicted_class = classifier.predict(formatted_landmark)
            # # get the string label
            # predicted_name = label_encoder.inverse_transform(predicted_class-1)
            # cv2.putText(frame, f'Predicted Class: {predicted_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            pose_landmarks_list = landmarks.pose_landmarks

            # Loop through the detected poses to visualize.
            for idx in range(len(pose_landmarks_list)):
                pose_landmarks = pose_landmarks_list[idx]

                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    frame,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style())
            
    # Display the output
    cv2.imshow('Mediapipe, RF - Yoga Pose Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

