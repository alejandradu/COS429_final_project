## visualize what representation were latent in the optimized model

import os
import numpy as np
import matplotlib.pyplot as plt

# load model
with open('randomForest.pkl', 'rb') as file:
    model = pickle.load(file)
    
# load the 