"""Use Blazepose to detect 33 keypoints on image frames coming from real-time
video stream using opencv"""

import cv2
import mediapipe as mp
import pickle
from training_pipeline.extract_features import *


# Initialize MediaPipe PoseLandmarker
mp_model_path = "../pretrained_models/pose_landmarker_heavy.task"
# initialize listener object
listener = MediapipeListener()
# create detector in live stream mode
detector = mediapipe_detector(mp_model_path, live_stream=True, listener=listener, min_pose_detection_confidence=0.7)
# load the label encoder
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/training_pipeline/label_encoder.pkl', 'rb') as f:    
    label_encoder = pickle.load(f)

# load the trained model
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/trained_classifiers/rf_pad.pkl', 'rb') as f:
    model = pickle.load(f)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

# Create a loop to read the latest frame from the camera
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to fetch the frame.")
        break

    # Run inference on the image
    mediapipe_detect(detector, frame, live_stream=True, frame_timestamp_ms=cap.CAP_PROP_POS_MSEC)
    # results = pose.process(frame_rgb)
    
    # landmarks go to the listener
    landmarks = listener.get()
    
    # assess how different is it from 256 256??
    print(frame.shape)
    
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
            annotated_image = draw_landmarks_on_image(frame, landmarks) 
            formatted_landmark = mediapipe_format_landmark(landmarks)
            # run inference
            predicted_class = model.predict(formatted_landmark)
            # get the string label
            predicted_name = label_encoder.inverse_transform(predicted_class-1)
            cv2.putText(frame, f'Predicted Class: {predicted_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
    # Display the output
    cv2.imshow('Mediapipe, RF - Yoga Pose Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

