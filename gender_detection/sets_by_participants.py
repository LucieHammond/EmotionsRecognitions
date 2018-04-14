"""
This file build 2 sets (a training set and a test set) in the form of numpy arrays.
To construct the sets, we divide randomly the set of participants between 2 groups (with a given number of test participants)
The numpy arrays are saved as .npy files in the temp directory

This file looks like deeplearning/prepared_sets but it differs in 2 points :
- the sets are built by separating participants randomly and not images
- we construct labels not only for emotions but also for genders
"""

from constants import TEMP_PATH, EMOTIONS, SIZE, DATASET_PATH, GENDERS
from deeplearning.prepare_sets import get_dataset_size
import numpy as np
import cv2
import re
import os
import glob
import random

width = SIZE
height = SIZE
NB_TEST_PARTICIPANTS = 15


def get_gender_dict():
    """ Create a dictionary that links each participant with a gender by reading the gender.txt file """
    genders = {}
    with open(os.path.join(DATASET_PATH, 'genders.txt'), 'r') as file:
        for line in file:
            infos = re.split(r"\s+", line)
            genders[infos[0]] = infos[1]
    return genders


def get_participants():
    """ Return list of participants by id """
    paths = glob.glob(DATASET_PATH + "/images/*")
    return [path[-4:] for path in paths]


def fill_datasets(size, isTrain):
    """ Create 3 arrays of given size that contains all the data about images corresponding to the participants of a
    given set (TEST_PEOPLE if isTrain = False and the other participants if isTrain = True)
    - inputs : list all images of chosen participants as 2-dimensional arrays of pixels
    - labels_emotions : list emotion labels for all those images as integers between 0 and 6
    - labels_gender : list gender labels for all thos images as integers between 0 and 1
    """

    # Create numpy arrays for all inputs and all labels
    inputs = np.empty((size, width, height), dtype=np.float32)
    labels_emotions = np.empty(size, dtype=np.int32)
    labels_gender = np.empty(size, dtype=np.int32)

    genders = get_gender_dict()

    # Scan all images in final set to fill inputs and labels
    index = 0
    for emotion in range(len(EMOTIONS)):
        images = glob.glob(TEMP_PATH + "/final_set/%s/*" % EMOTIONS[emotion])  # Get list of all images with emotion
        for img_path in images:
            participant = img_path[-8: -4]
            skip = participant in TEST_PEOPLE if isTrain else participant not in TEST_PEOPLE
            if skip:
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            gender = GENDERS.index(genders[participant])

            inputs[index] = img
            labels_emotions[index] = emotion
            labels_gender[index] = gender
            index += 1

    return inputs, labels_emotions, labels_gender


def get_test_size(participants):
    """ Return number of images that represent given participants (this will be the size of test set) """
    nb_images = 0
    for emotion in EMOTIONS:
        images_paths = glob.glob(TEMP_PATH + "/final_set/%s/*" % emotion)
        nb_images += len([path for path in images_paths if path[-8:-4] in participants])
    return nb_images



if __name__ == "__main__":

    # Get total size
    total_size = get_dataset_size()
    print("Après traitement, le dataset comporte au total %i images labelisées" % total_size)  # 427 images

    # Get test size (number of images with test participants (TEST_PEOPLE)
    ALL_PEOPLE = get_participants()
    random.shuffle(ALL_PEOPLE)
    TEST_PEOPLE = ALL_PEOPLE[:NB_TEST_PARTICIPANTS]  # Take randomly NB_TEST_PARTICIPANTS people
    test_size = get_test_size(TEST_PEOPLE)

    # Shape the data for test in 3 numpy arrays
    print("Le test set est constitué des images des participants %s" % ", ".join(TEST_PEOPLE))
    print("Taille test set : %i" % test_size)
    test_inputs, test_labels_emotions, test_labels_gender = fill_datasets(test_size, False)

    # Shape the data for training in 3 numpy arrays
    print("Le training set est constitué de toutes les autres images")
    print("Taille test set : %i" % (total_size - test_size))
    train_inputs, train_labels_emotions, train_labels_gender = fill_datasets(total_size - test_size, True)

    # Save all prepared numpy datasets
    print("Save train_inputs...")
    np.save(TEMP_PATH + '/prepared_sets/train_inputs.npy', train_inputs)
    print("Save train_labels_emotions...")
    np.save(TEMP_PATH + '/prepared_sets/train_labels_emotion.npy', train_labels_emotions)
    print("Save train_labels_gender...")
    np.save(TEMP_PATH + '/prepared_sets/train_labels_gender.npy', train_labels_gender)
    print("Save test_inputs...")
    np.save(TEMP_PATH + '/prepared_sets/test_inputs.npy', test_inputs)
    print("Save test_labels_emotions...")
    np.save(TEMP_PATH + '/prepared_sets/test_labels_emotion.npy', test_labels_emotions)
    print("Save test_labels_gender...")
    np.save(TEMP_PATH + '/prepared_sets/test_labels_gender.npy', test_labels_gender)
