# Class to process data as keypoints and measure lengths/angles to classify poses

class Structure():
    ### REVISE OUTPUT BELOW FROM GPT
    def __init__(self, keypoints):
        self.keypoints = keypoints
        self.lengths = []
        self.angles = []
        self.calculate_lengths()
        self.calculate_angles()

    def calculate_lengths(self):
        for i in range(len(self.keypoints) - 1):
            for j in range(i + 1, len(self.keypoints)):
                x1, y1 = self.keypoints[i]
                x2, y2 = self.keypoints[j]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                self.lengths.append(length)

    def calculate_angles(self):
        for i in range(len(self.keypoints) - 2):
            for j in range(i + 1, len(self.keypoints) - 1):
                for k in range(j + 1, len(self.keypoints)):
                    x1, y1 = self.keypoints[i]
                    x2, y2 = self.keypoints[j]
                    x3, y3 = self.keypoints[k]
                    a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
                    c = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
                    angle = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                    self.angles.append(angle)

    def get_lengths(self):
        return self.lengths

    def get_angles(self):
        return self.angles

    def get_keypoints(self):
        return self.keypoints