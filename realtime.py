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
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/trained_classifiers/PASTrandom_forest.pkl', 'rb') as f:
    classifier = pickle.load(f)
    
# load the sanskrit to english dictionary
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/sanskrit_english_dict.pkl', 'rb') as f:
    sanskrit_english_dict = pickle.load(f)

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
    
    # shape is very different: 1080 by 1920
    # print(np.shape(frame))
    
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
            
            pose_landmarks_list = landmarks.pose_landmarks

            # Loop through the detected poses to visualize.
            for idx in range(len(pose_landmarks_list)):
                pose_landmarks = pose_landmarks_list[idx]

                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                to_extend = []
                to_classify = np.zeros((1,features_mp.n_landmarks * 4))
                
                # store normalized landmarks to appends and classify
                i = 0
                for landmark in pose_landmarks:
                    to_extend.append(landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z))
                    to_classify[0, i:i+4] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
                    i += 4
                
                pose_landmarks_proto.landmark.extend(to_extend)
                solutions.drawing_utils.draw_landmarks(
                    frame,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style())
                
            # Run inference
            #if to_classify:
            # input_x = np.array(to_classify)[0].reshape(1,-1)
            predicted_class = classifier.predict(to_classify)
            # Get the string label
            predicted_name = label_encoder.inverse_transform([int(predicted_class-1)])
            text = sanskrit_english_dict[predicted_name[0]]
            print(text)
            
            # Define text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 4
            font_thickness = 2
            text_color = (255, 255, 255)  # White color
            bg_color = (0, 0, 0)  # Black color for background rectangle
            bg_opacity = 0.6  # Background opacity

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_offset_x, text_offset_y = 10, 30
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width, text_offset_y - text_height - baseline))

            # Put text on the frame
            cv2.putText(frame, text, (text_offset_x, text_offset_y - baseline), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
            
    # Display the output
    cv2.imshow('Mediapipe, RF - Yoga Pose Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

