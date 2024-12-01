"""Use Blazepose to detect 33 keypoints on image frames coming from real-time
video stream using opencv"""

# FROM LATER

import cv2
import mediapipe as mp

# Initialize BlazePose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # For visualizing keypoints

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to fetch the frame.")
        break

    # Convert the frame to RGB (MediaPipe uses RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with BlazePose
    results = pose.process(frame_rgb)

    # Draw landmarks if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            print(f'Landmark: {landmark.x}, {landmark.y}, {landmark.z}')

    # Display the output
    cv2.imshow('BlazePose - Pose Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


# FROM THE BLAZEPOSE REPO:

# from BlazeposeDepthaiEdge import BlazeposeDepthai
# from BlazeposeRenderer import BlazeposeRenderer

# # The argparse stuff has been removed to keep only the important code

# tracker = BlazeposeDepthai(input_src=args.input, 
#             pd_model=args.pd_m,
#             lm_model=args.lm_m,
#             smoothing=not args.no_smoothing,   
#             xyz=args.xyz,           
#             crop=args.crop,
#             internal_fps=args.internal_fps,
#             internal_frame_height=args.internal_frame_height,
#             force_detection=args.force_detection,
#             stats=args.stats,
#             trace=args.trace)   

# renderer = BlazeposeRenderer(
#                 pose, 
#                 show_3d=args.show_3d, 
#                 output=args.output)

# while True:
#     # Run blazepose on next frame
#     frame, body = tracker.next_frame()
#     if frame is None: break
#     # Draw 2d skeleton
#     frame = renderer.draw(frame, body)
#     key = renderer.waitKey(delay=1)
#     if key == 27 or key == ord('q'):
#         break
# renderer.exit()
# tracker.exit()