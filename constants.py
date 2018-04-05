import os

# Absolute Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT_DIR, 'Cohn-Kanade-Dataset')
RESOURCE_PATH = os.path.join(ROOT_DIR, 'resources')
TEMP_PATH = os.path.join(ROOT_DIR, 'temp')
RESULTS_PATH = os.path.join(ROOT_DIR, 'results')

# Constants
CK_EMOTIONS = ["neutral", "anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]

# As contempt is hard to identify and has less data that the others, we removed it for the training
EMOTIONS = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# Taille des images de visages (350px x 350px)
SIZE = 200