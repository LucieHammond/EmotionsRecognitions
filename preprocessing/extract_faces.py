"""
Ce fichier détecte les visages dans les images du set trié grâce aux librairies haarcascade_frontalface.
Quand on a trouvé un visage, on redimensionne l'image autour de ce visage et on fait passer l'image en grayscale.
Toutes les images enregistrées dans temp/final_set sont de taille 350 x 350 et en niveaux de gris
"""

import cv2
import glob
import os
from constants import RESOURCE_PATH, TEMP_PATH, CK_EMOTIONS, SIZE

faceDet = cv2.CascadeClassifier(RESOURCE_PATH + "/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier(RESOURCE_PATH + "/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier(RESOURCE_PATH + "/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier(RESOURCE_PATH + "/haarcascade_frontalface_alt_tree.xml")


def detect_faces(emotion):
    files = glob.glob(TEMP_PATH + "/sorted_set/%s/*" % emotion)  # Get list of all images with emotion

    filenumber = 0
    for f in files:
        frame = cv2.imread(f)  # Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

        # Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

        # Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            print("Visage non trouvé dans l'image %s" % f)
            facefeatures = ""

        # Cut and save face
        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            gray = gray[y:y + h, x:x + w]  # Cut the frame to size

            try:
                out = cv2.resize(gray, (SIZE, SIZE))  # Resize face so all images have same size
                cv2.imwrite(TEMP_PATH + "/final_set/%s/%s.png" % (emotion, filenumber), out)  # Write image
            except:
                print("Une erreur est survenue lors du redimensionnage de l'image %s" %f)
                pass  # If error, pass file
        filenumber += 1  # Increment image number


if __name__ == "__main__":
    # Generate path to store final set in temp directory
    if not os.path.exists(TEMP_PATH + '/final_set'):
        os.makedirs(TEMP_PATH + '/final_set')
    for emotion in CK_EMOTIONS:
        if not os.path.exists(TEMP_PATH + '/final_set/' + emotion):
            os.makedirs(TEMP_PATH + '/final_set/' + emotion)

    for emotion in CK_EMOTIONS:
        detect_faces(emotion)