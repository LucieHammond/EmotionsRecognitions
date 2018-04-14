"""
This file sort the images that will be useful among brute Cohn-Kanade dataset
Sorted data will be saved in directory temp/sorted_set
"""

import glob
import os
from shutil import copyfile
from constants import DATASET_PATH, TEMP_PATH, CK_EMOTIONS


# Emotions used in Cohn-Kanade Dataset (in this order)
participants = glob.glob(DATASET_PATH + "/emotions/*")  # Returns a list of all folders with participant numbers

# Generate path to store sorted set in temp directory
if not os.path.exists(TEMP_PATH + '/sorted_set'):
    os.makedirs(TEMP_PATH + '/sorted_set')
for emotion in CK_EMOTIONS:
    if not os.path.exists(TEMP_PATH + '/sorted_set/' + emotion):
        os.makedirs(TEMP_PATH + '/sorted_set/' + emotion)

for p in participants:
    participant = "%s" % p[-4:]  # current participant number
    for s in glob.glob("%s/*" % p):
        session = "%s" % s[-3:]  # session number for current participant
        for filename in glob.glob("%s/*" % s):
            # There is one or zero emotion file for each session (not all emotions were easily identifiable)
            nb_picture = int(filename[-20:-12])  # nb of last picture

            with open(filename, 'r') as file:
                emotion = int(float(file.readline()))  # emotions are encoded as a float

            # Get corresponding images in images directory, first one is neutral, and last ones contain the emotion
            sourcefile_neutral = DATASET_PATH + "/images/%(p)s/%(s)s/%(p)s_%(s)s_%(nb)s.png" % {
                'p': participant, 's': session, 'nb': str(1).zfill(8)}
            sourcefile_emotion = DATASET_PATH + "/images/%(p)s/%(s)s/%(p)s_%(s)s_%(nb)s.png" % {
                'p': participant, 's': session, 'nb': str(nb_picture).zfill(8)}

            # New path for neutral image
            dest_neut = TEMP_PATH + "/sorted_set/neutral/%s" % sourcefile_neutral[-21:]
            # New path for images containing emotion
            dest_emot = TEMP_PATH + "/sorted_set/%s/%s" % (CK_EMOTIONS[emotion], sourcefile_emotion[-21:])

            # Copy files to new path
            copyfile(sourcefile_neutral, dest_neut)
            copyfile(sourcefile_emotion, dest_emot)