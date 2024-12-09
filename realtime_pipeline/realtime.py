"""Use Blazepose to detect 33 keypoints on image frames coming from real-time
video stream using opencv"""

# I don't see the difference between blaze pose and normal pose from mediapipe

import cv2
import mediapipe as mp
from utils import format_landmark
import pickle

# Initialize BlazePose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # For visualizing keypoints

# load the trained model
with open('randomForest.pkl', 'rb') as f:
    model = pickle.load(f)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to fetch the frame.")
        break

    # Convert the frame to RGB (MediaPipe uses RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with BlazePose
    results = pose.process(frame_rgb)
    # TODO: refine the detection - maybe try movenet
    
    ## MODEL COME IN - process the keypoints detected and classify the pose
    ## and display it

    # Draw landmarks if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # for landmark in results.pose_landmarks.landmark:
        #     print(f'Landmark: {landmark.x}, {landmark.y}, {landmark.z}')
        
        formatted_landmark = format_landmark(results.pose_landmarks)
        predicted_class = model.predict(formatted_landmark)
        cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # retrieve from json file the english name corresponding to the predicted class
        
        
        # print(f'Predicted Class: {predicted_class}')
    
    # Correction[real-time landarks, yoga classificatin (template right pose)] = corrections
    
    # Display corrections
    
    # # write text for the predicted class and its name on the mp_drawing frame
    # cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the output
    cv2.imshow('BlazePose - Pose Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

