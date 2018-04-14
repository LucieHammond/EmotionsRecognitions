"""
This file lists all the paths ans the constants of the project
"""

import os

# ABSOLUTE PATHS TO DIRECTORIES
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT_DIR, 'Cohn-Kanade-Dataset')
RESOURCE_PATH = os.path.join(ROOT_DIR, 'resources')
TEMP_PATH = os.path.join(ROOT_DIR, 'temp')
RESULTS_PATH = os.path.join(ROOT_DIR, 'results')

# CONSTANTS

# List of emotions identified in Cohn-Kanade database
CK_EMOTIONS = ["neutral", "anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]

# List of emotions we used in this project
# As contempt is hard to identify and has less data that the others, we removed it for the training
EMOTIONS = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# List of genders we used in this project
GENDERS = ['F', 'M']

# Size of the face images (we will resize photos around the face)
SIZE = 200  # Images will be of size 200x200px