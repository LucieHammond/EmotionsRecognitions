"""
This file build 2 sets (a training set and a test set) in the form of numpy arrays.
To construct the sets, we divide randomly the set of images between 2 groups (85% for trining and 15% for test)
The numpy arrays are saved as .npy files in the temp directory
"""

from constants import TEMP_PATH, EMOTIONS, SIZE
import numpy as np
import cv2
import glob
import random

width = SIZE
height = SIZE


def get_dataset_size():
    """ List all images in final_set in order to count them and return the number of face images """
    nb_images = 0
    for emotion in EMOTIONS:
        nb_images += len(glob.glob(TEMP_PATH + "/final_set/%s/*" % emotion))
    return nb_images


def fill_dataset(size):
    """ Create 2 global arrays (inputs and labels) and return them
    - inputs : list all images as 2 dimensional arrays od pixels
    - labels : list the emotion labels of each image as integers between 0 ad 6
    """

    # Create numpy arrays for all inputs and all labels
    inputs = np.empty((size, width, height), dtype=int)
    labels = np.empty(size, dtype=int)

    # Scan all images in final set to fill inputs and labels
    index = 0
    for emotion in range(len(EMOTIONS)):
        images = glob.glob(TEMP_PATH + "/final_set/%s/*" % EMOTIONS[emotion])  # Get list of all images with emotion
        for img_path in images:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            inputs[index] = img
            labels[index] = emotion
            index += 1

    return inputs, labels


def pick_elems(inputs, labels, elems):
    """Create and return new sub-arrays of inputs and labels by picking only the elements from indexes given in elems"""

    sub_inputs = np.empty((len(elems), width, height), dtype=np.float32)
    sub_labels = np.empty(len(elems), dtype=np.int32)

    for ind, index in enumerate(elems):
        sub_inputs[ind] = inputs[index]
        sub_labels[ind] = labels[index]

    return sub_inputs, sub_labels


if __name__ == "__main__":

    # Get total size
    total_size = get_dataset_size()
    print("Après traitement, le dataset comporte au total %i images labelisées" % total_size)  # 945 images

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
    print("Save train_inputs...")
    np.save(TEMP_PATH + '/prepared_sets/train_inputs.npy', train_inputs)
    print("Save train_labels...")
    np.save(TEMP_PATH + '/prepared_sets/train_labels.npy', train_labels)
    print("Save test_inputs...")
    np.save(TEMP_PATH + '/prepared_sets/test_inputs.npy', test_inputs)
    print("Save test_labels...")
    np.save(TEMP_PATH + '/prepared_sets/test_labels.npy', test_labels)