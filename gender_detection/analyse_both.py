"""
This file tests the CNNs models trained with cnn_model.py file (which state is stored in temp/cnn_model directory) and
cnn_model_gender.py file (which state is stored in temp/cnn_model_gender directory)

Those 2 models must have been trained previously on the same training images (with respectively emotions and gender
labels) and will be tested on the same test images with respectively emotions and gender labels)

We compute the accuracy for each model and for the combination of the models and we print the 3 matrix of confusion
We save the results for the combination of 2 models as a repartition of the images in result/classes_gender directory
We print the errors made in a special file named errors_gender.txt
"""

import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from constants import EMOTIONS, RESULTS_PATH, TEMP_PATH, GENDERS
from deeplearning.cnn_model import model as model_emotions
from gender_detection.cnn_model_gender import model as model_gender
from deeplearning.analysis import add_colored_icon, plot_confusion_matrix


def save_results(inputs, labels_emotion, labels_gender, results_emotion, results_gender):
    """
    Compare labels_emotions with results_emotions and labels_gender with results_gender for each index
    Save the corresponding test_input image in a folder corresponding to the emotion and the gender found
    with a small icon at its bottom right hand corner showing if this result was right or not.

    Write the results and the true labels in errors_gender.txt file if one or the two models were wrong
    """

    # Create folders for emotions
    if not os.path.exists(RESULTS_PATH + '/classes_gender'):
        os.makedirs(RESULTS_PATH + '/classes_gender')
    for emotion in EMOTIONS:
        for gender in GENDERS:
            if not os.path.exists(RESULTS_PATH + '/classes_gender/' + emotion + '_' + gender):
                os.makedirs(RESULTS_PATH + '/classes_gender/' + emotion + '_' + gender)

    # Create error.txt file that points out errors
    with open(RESULTS_PATH + '/errors_gender.txt', "w") as error:
        error.write('Id \t Results \t Labels  \t Nb Errors\n')

        id = 0
        for index in range(len(inputs)):
            img = inputs[index]
            emotion_label = labels_emotion[index]
            emotion_result = results_emotion[index]
            gender_label = labels_gender[index]
            gender_result = results_gender[index]

            # If there is only one error
            if emotion_label != emotion_result and gender_label != gender_result :
                error.write('%(id)i \t %(res)s \t %(lab)s \t %(nb)i\n' %
                            {'id': id,
                             'res': EMOTIONS[emotion_result].ljust(10) + GENDERS[gender_result],
                             'lab': EMOTIONS[emotion_label].ljust(10) + GENDERS[gender_label],
                             'nb': 2})
                img = add_colored_icon(img, False)
            # If there are 2 errors
            elif emotion_label != emotion_result or gender_label != gender_result :
                error.write('%(id)i \t %(res)s \t %(lab)s \t %(nb)i\n' %
                            {'id': id,
                             'res': EMOTIONS[emotion_result].ljust(10) + GENDERS[gender_result],
                             'lab': EMOTIONS[emotion_label].ljust(10) + GENDERS[gender_label],
                             'nb': 1})
                img = add_colored_icon(img, False)
            # If there is no error
            else:
                img = add_colored_icon(img, True)

            # Write image in folder corresponding to the emotion found
            global_label = EMOTIONS[emotion_result] + '_' + GENDERS[gender_result]
            cv2.imwrite(RESULTS_PATH + "/classes_gender/%s/%s.png" % (global_label, id), img)

            id += 1


if __name__ == "__main__":
    print("Load test dataset...")
    test_inputs = np.load(TEMP_PATH + '/prepared_sets/test_inputs.npy')
    test_labels_emotion = np.load(TEMP_PATH + '/prepared_sets/test_labels_emotion.npy')
    test_labels_gender = np.load(TEMP_PATH + '/prepared_sets/test_labels_gender.npy')

    # Create the Emotion Estimator
    emotions_classifier = tf.estimator.Estimator(
        model_fn=model_emotions, model_dir=TEMP_PATH + '/cnn_model')

    # Create the Gender Estimator
    gender_classifier = tf.estimator.Estimator(
        model_fn=model_gender, model_dir=TEMP_PATH + '/cnn_model_gender')

    # Predict emotions results for the test set
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_inputs},
        y=test_labels_emotion,
        num_epochs=1,
        shuffle=False)
    test_results = emotions_classifier.predict(input_fn=test_input_fn)
    predicted_emotions = [predict["classes"] for predict in test_results]

    # Predict gender results for the test set
    test_input_fn2 = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_inputs},
        y=test_labels_gender,
        num_epochs=1,
        shuffle=False)
    test_results = gender_classifier.predict(input_fn=test_input_fn2)
    predicted_genders = [predict["classes"] for predict in test_results]

    # Save results
    save_results(test_inputs, test_labels_emotion, test_labels_gender, predicted_emotions, predicted_genders)

    # Compute accuracy for emotions
    accuracy_emotions = accuracy_score(test_labels_emotion, predicted_emotions)
    print("Accuracy pour les émotions : %.2f" % (accuracy_emotions * 100), '%\n')

    # Compute accuracy for gender
    accuracy_gender = accuracy_score(test_labels_gender, predicted_genders)
    print("Accuracy pour le genre : %.2f" % (accuracy_gender * 100), '%\n')

    # Compute total accuracy
    test_size = len(test_labels_emotion)
    test_labels = [test_labels_gender[i] * len(EMOTIONS) + test_labels_emotion[i] for i in range(test_size)]
    predicted = [predicted_genders[i] * len(EMOTIONS) + predicted_emotions[i] for i in range(test_size)]
    accuracy = accuracy_score(test_labels, predicted)
    print("Accuracy globale : %.2f" % (accuracy * 100), '%\n')

    # Compute confusion matrix for emotions
    print("Matrice de confusion pour les émotions")
    cnf_matrix_e = confusion_matrix(test_labels_emotion, predicted_emotions)
    plot_confusion_matrix(cnf_matrix_e, classes=EMOTIONS, normalize=False,
                          title='Matrice de confusion pour les émotions')
    plt.show()

    # Compute confusion matrix for gender
    print("Matrice de confusion pour le genre")
    cnf_matrix_g = confusion_matrix(test_labels_gender, predicted_genders)
    plot_confusion_matrix(cnf_matrix_g, classes=GENDERS, normalize=False,
                          title='Matrice de confusion pour les genres')
    plt.show()

    # Compute global confusion matrix
    print("Matrice de confusion globale")
    cnf_matrix = confusion_matrix(test_labels, predicted)
    ALL_CLASSES = [emotion + '_' + gender for emotion in EMOTIONS for gender in GENDERS]
    plot_confusion_matrix(cnf_matrix, classes=ALL_CLASSES, normalize=False,
                          title='Matrice de confusion globale')
    plt.show()