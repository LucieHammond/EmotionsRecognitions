"""
Ce fichier construit des ensembles d'apprentissage (training) et de test sous forme de tableaux numpy.
Les arrays sont enregistrés sous forme de fichiers .npy dans le dossier temp
"""

from constants import TEMP_PATH, EMOTIONS, SIZE, DATASET_PATH
import numpy as np
import cv2
import re
import os
import glob
import random

width = SIZE
height = SIZE


def get_gender_dict():
    genders = {}
    with open(os.path.join(DATASET_PATH, 'genders.txt'), 'r') as file:
        for line in file:
            infos = re.split(r"\s+", line)
            genders[infos[0]] = infos[1]
    return genders


def get_dataset_size():
    nb_images = 0
    for emotion in EMOTIONS:
        nb_images += len(glob.glob(TEMP_PATH + "/final_set/%s/*" % emotion))
    return nb_images


def fill_dataset(size):

    # Create numpy arrays for all inputs and all labels
    inputs = np.empty((size, width, height), dtype=int)
    labels = np.empty(size, dtype=int)

    genders = get_gender_dict()

    # Scan all images in final set to fill inputs and labels
    index = 0
    for emotion in range(len(EMOTIONS)):
        images = glob.glob(TEMP_PATH + "/final_set/%s/*" % EMOTIONS[emotion])  # Get list of all images with emotion
        for img_path in images:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            participant = img_path[-8: -4]
            gender = 0 if genders[participant] == 'F' else 1

            inputs[index] = img
            labels[index] = gender * len(EMOTIONS) + emotion
            index += 1

    return inputs, labels


def pick_elems(inputs, labels, elems):
    sub_inputs = np.empty((len(elems), width, height), dtype=np.float32)
    sub_labels = np.empty(len(elems), dtype=np.int32)

    for ind, index in enumerate(elems):
        sub_inputs[ind] = inputs[index]
        sub_labels[ind] = labels[index]

    return sub_inputs, sub_labels


if __name__ == "__main__":

    # Get total size
    total_size = get_dataset_size()
    print("Après traitement, le dataset comporte au total %i images labelisées" % total_size)

    # Shape all the data of final set in 2 numpy arrays
    all_inputs, all_labels = fill_dataset(total_size)

    # Split the inputs in 2 sets : training and test (we will use cross validation on the training set)
    print("\nOn sépare les données d'entrée en deux ensembles distincts:")
    train_size = int(0.85 * total_size)  # Compute the size of all sets
    test_size = total_size - train_size
    print("training + validation set : %i" % train_size)
    print("test set : %i" % test_size)
    print()

    # Pick elements randomly to complete the 2 sets
    elems = list(range(total_size))
    random.shuffle(elems)
    train_elems = elems[:train_size]
    test_elems = elems[train_size:]

    train_inputs, train_labels = pick_elems(all_inputs, all_labels, train_elems)
    test_inputs, test_labels = pick_elems(all_inputs, all_labels, test_elems)

    # Save all prepared numpy datasets
    print("Save train_inputs_gender...")
    np.save(TEMP_PATH + '/prepared_sets/train_inputs_gender.npy', train_inputs)
    print("Save train_labels_gender...")
    np.save(TEMP_PATH + '/prepared_sets/train_labels_gender.npy', train_labels)
    print("Save test_inputs_gender...")
    np.save(TEMP_PATH + '/prepared_sets/test_inputs_gender.npy', test_inputs)
    print("Save test_labels_gender...")
    np.save(TEMP_PATH + '/prepared_sets/test_labels_gender.npy', test_labels)
