"""
This file tests the CNN model trained with cnn_model.py file (which state is stored in temp/cnn_model directory).
We save the results on test set as a repartition of the images in result/classes directory
We print the errors made in a special file named errors.txt
We then compute the accuracy and the matrix of confusion and display it with matplotlib
"""

import os
import cv2
import itertools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from constants import EMOTIONS, RESULTS_PATH, TEMP_PATH, SIZE
from deeplearning.cnn_model import model


def add_colored_icon(img, success):
    """ Add a small icon in the bottom right hand corner of the image:
    - a green check if success = True
    - a red cross if success = False
    Return the image with the modification
    """

    # Convert to colored image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if success:
        icon = cv2.imread(RESULTS_PATH + '/check.png', cv2.IMREAD_GRAYSCALE)
        color = 1  # green
    else :
        icon = cv2.imread(RESULTS_PATH + '/cross.png', cv2.IMREAD_GRAYSCALE)
        color = 2  # red

    new_img = img_color.astype(dtype=np.int32)
    nb_lines, nb_cols = icon.shape
    for i in range(nb_lines):
        for j in range(nb_cols):
            if (icon[i][j] > 20):
                new_img[SIZE - nb_lines + i][SIZE - nb_cols + j] = (0, 0, 0)
                new_img[SIZE - nb_lines + i][SIZE - nb_cols + j][color] = icon[i][j]

    return new_img


def save_results(test_inputs, test_labels, test_results):
    """
    Compare test_results with test_labels for each index and save the corresponding test_input image
    in a folder corresponding to the emotion found with a small icon at its bottom right hand corner
    showing if this result was right or not.

    Write the result and the true label in errors.txt file if the model was wrong
    """

    # Create folders for emotions
    if not os.path.exists(RESULTS_PATH + '/classes'):
        os.makedirs(RESULTS_PATH + '/classes')
    for emotion in EMOTIONS:
        if not os.path.exists(RESULTS_PATH + '/classes/' + emotion):
            os.makedirs(RESULTS_PATH + '/classes/' + emotion)

    # Create error.txt file that points out errors
    with open(RESULTS_PATH + '/errors.txt', "w") as error:
        error.write('Id \t Results \t Label\n')

        id = 0
        for index, result in enumerate(test_results):
            img = test_inputs[index]
            label = test_labels[index]
            emotion_found = EMOTIONS[result]

            if label != result:
                error.write('%i \t %s \t %s\n' %(id, emotion_found.ljust(10), EMOTIONS[label]))
                img = add_colored_icon(img, False)
            else :
                img = add_colored_icon(img, True)

            # Write image in folder corresponding to the emotion found
            cv2.imwrite(RESULTS_PATH + "/classes/%s/%s.png" % (emotion_found, id), img)

            id += 1


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]*100)/100.0,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
    # Load test dataset
    print("Load test dataset...")
    test_inputs = np.load(TEMP_PATH + '/test_inputs.npy')
    test_labels = np.load(TEMP_PATH + '/test_labels.npy')

    # Create the Estimator
    emotions_classifier = tf.estimator.Estimator(
        model_fn=model, model_dir=TEMP_PATH + '/cnn_model')

    # Predict results for the test set
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_inputs},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    test_results = emotions_classifier.predict(input_fn=test_input_fn)
    predicted_classes = [predict["classes"] for predict in test_results]

    # Save results
    save_results(test_inputs, test_labels, predicted_classes)

    # Compute accuracy
    accuracy = accuracy_score(test_labels, predicted_classes)
    print("Accuracy : %.2f" % (accuracy * 100), '%\n')

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_labels, predicted_classes)
    plot_confusion_matrix(cnf_matrix, classes=EMOTIONS, normalize=False,
                          title='Confusion Matrix for Test Dataset')

    plt.show()
