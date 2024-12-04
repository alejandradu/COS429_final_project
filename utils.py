import numpy as np

def format_landmark(landmarks):
    """
    Format the landmark data for a single frame detection, to input into the classfier.
    Already check if landmarks is not None before using...

    Args:
        landmarks: a Pose().process.pose_landmarks object.

    Returns:
        X: a numpy array of shape (33, 4) containing the x, y, z, and visibility of each landmark.
    """
    X = np.zeros((33, 4))   
    
    # extract list of landmarks for 33 locations
    positions = landmarks.landmark
    j = 0
    for landmark in positions:
        x = landmark.x
        y = landmark.y
        z = landmark.z
        v = landmark.visibility
        X[j] = [x, y, z, v]
        j += 1
        
    # flatten the array
    X = X.reshape(1, 33 * 4)
    
    return X