"""Use Mediapipe and our classifier to detect 33 keypoints on image frames coming from real-time
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
features_mp = FeaturesMP(mp_model_path, image_size=(1080, 1920))
# Initialize detector
features_mp.init_detector(live_stream=True, min_pose_detection_confidence=0.7)#, double=True)
# load the label encoder
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/training_pipeline/label_encoder.pkl', 'rb') as f:    
    label_encoder = pickle.load(f)

# load the trained classifier
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/trained_classifiers/padded_nn_7.pkl', 'rb') as f:
    classifier = pickle.load(f)
    
# load the sanskrit to english dictionary
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/sanskrit_english_dict.pkl', 'rb') as f:
    sanskrit_english_dict = pickle.load(f)
    
# introduce delay in position predictions
buffer = [-1,-1,-1,-1]

# Define text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 3
font_thickness = 4
text_color = (255, 255, 255)  # White color
bg_color = (0, 0, 0)  # Black color for background rectangle
bg_opacity = 0.6  # Background opacity
coords = (50, 100)  # Coordinates to display the text
text = "Starting pose detection..."

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

# Create a loop to read the latest frame from the camera
while cap.isOpened():
    ret, frame = cap.read()
    timestamp = int(round(time.time()*1000))
    
    if not ret:
        print("Error: Unable to fetch the frame.")
        break

    # Run inference on the image 
    # uncomment if double: landmarks_draw, landmarks = features_mp.detect(frame, live_stream=True, frame_timestamp_ms=timestamp, double=True)
    landmarks = features_mp.detect(frame, live_stream=True, frame_timestamp_ms=timestamp)
    
    # Draw landmarks if detected
    if landmarks is not None: # uncomment if double: and landmarks_draw is not None:  
        if len(landmarks.pose_landmarks) != 0: # uncomment if double: and len(landmarks_draw.pose_landmarks) != 0:
            
            pose_landmarks_list = landmarks.pose_landmarks 
            # uncomment if double: pose_landmarks_draw_list = landmarks_draw.pose_landmarks

            # get only normalized coordinates - improves latency
            pose_landmarks = pose_landmarks_list[0]
            # uncomment if double: pose_landmarks_draw = pose_landmarks_draw_list[0]
            
            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            to_classify = np.zeros((features_mp.n_landmarks, 4))
            to_extend = []
            
            # store normalized landmarks to appends and classify
            for k, landmark in enumerate(pose_landmarks):
                # uncomment if double: to_extend.append(landmark_pb2.NormalizedLandmark(x=pose_landmarks_draw[k].x, y=pose_landmarks_draw[k].y, z=pose_landmarks_draw[k].z))
                to_extend.append(landmark_pb2.NormalizedLandmark(x=landmark.x,y=landmark.y, z=landmark.z))
                # store in an array
                to_classify[k] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
                
            # normalize and rotate to_classify
            to_classify = features_mp.make_rot_invariant(to_classify, init_norm=True)
            to_classify = to_classify.reshape(1, features_mp.n_landmarks * 4)
            
            # draw real-time landmarks
            pose_landmarks_proto.landmark.extend(to_extend)
            solutions.drawing_utils.draw_landmarks(
                frame,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())
                
            # Run inference
            predicted_class = classifier.predict(to_classify)
            # Get the string label
            predicted_name = label_encoder.inverse_transform([int(predicted_class-1)])
            # Append to buffer
            buffer.pop(0)
            buffer.append(predicted_name[0])
            # # if all elements now in the buffer are the same, then we can display the pose
            if buffer[0] == buffer[1] == buffer[2] == buffer[3]:
                text = sanskrit_english_dict[predicted_name[0]]

            cv2.putText(frame, text, coords, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
            
    # Display the output
    cv2.imshow('Yoga Pose Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

