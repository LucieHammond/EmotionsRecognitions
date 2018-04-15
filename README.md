# EmotionsRecognitions
Projet de Deep Learning et de Vision par Ordinateur

Ce projet a pour but de reconnaître les émotions sur les visages dans des images.
Nous avons utilisé pour cela la base de donnée Cohn-Kanade CK+ qui peut être trouvée ici http://www.consortium.ri.cmu.edu/ckagree/index.cgi.
En particulier, il faudra télécharger les dataset suivants :
- **images** : extended-cohn-kanade-images.zip
- **emotions labels** : Emotion_labels.zip
Puis les insérer respectivement dans les dossiers _images_ et _emotions_ du dossier Cohn-Kanade-Dataset présent à la racine du projet. Ce dataset contient des séries d'images montrant la transformation des expression d'un visage d'un état neutre à une émotion labelisée. 

Pour construire et tester ce projet, vous aurez besoin d'exécuter des fichiers pythons, soit avec un logiciel dédié comme PyCharm, soit depuis la console. Si vous choisissez la console, ajoutez préalablement le chemin du projet au PYTHONPATH

## Pré-traitement

Pour pouvoir utiliser ce dataset d'images, nous l'avons préalablement trié, classé et modifié.
Les fichiers de prétraitement se trouvent dans de dossier preprocessing.

Exécutez d'abord `preprocessing/reorganize.py` : Cela va créer un dossier temp/sorted_set qui classe les premières et dernières images de chaque session avec un participant dans des dossier du nom de l'émotion représentée (neutre ou autre), si celle-ci est labelisée.

Puis exécutez `preprocessing/extract_faces.py` : Ce fichier repère les visages dans les images de temp/sorted_set et les redimensionne en les recadrant autour des visages et en les passant en noir et blanc. Le tout est sauvegardé dans un dossier temp/final_set.

## Apprentissage des émotions

Nous avons conservé 7 émotions à apprendre : "neutral", "anger", "disgust", "fear", "happiness", "sadness" et "surprise".
Pour cela nous utilisons un réseau neuronal convolutif à 6 couches : 2 couches convolutives chacunes suivie d'une couche de pooling pour l'extraction de features et 2 couches denses (fully connected) pour la classification.
Nous entrainons ce réseau sur un ensemble d'apprentissage et le testons sur un ensemble de test.

Pour préparer les ensembles d'apprentissage et de test, nous séparons le dataset pré-traité suivant des proportions choisies. Le dataset obtenu sous forme de tableau est présauvegardé sous forme de fichiers .npy dans le dossier temp/prapared_sets
- En exécutant `deeplearning/prepare_sets.py`, on sélectionne aléatoirement les images de test sur l'ensemble des images dans une proportion de 15%
- Alternativement, on pourra aussi exécuter `gender_detection/sets_by_participants.py` pour sélectionner plutôt un ensemble de participants (15) dont toutes les photos seront utilisées comme ensemble test.

_Attention, le premier fichier créera seulement des fichiers .npy d'inputs et de labels emotion tandis que le deuxième génèrera des fichiers d'imputs ainsi que 2 types de fichiers de labels (emotion et gender) -> voir section suivante_

Pour entrainer le réseau, il faut exécuter le fichier `deeplearning/cnn_model`. Ce fichier fait tourner 100 epochs (800 steps) en évaluant sur l'ensemble de test toutes les 10 epochs pour tracer une évolution de la perte de training et de généralisation. Il n'y a pas de cross-validation (pas d'ensemble de validation spécifique) mais les hyperparamètres ont été ajustés à la main.

Pour sauvegarder les résultats et les erreurs et afficher les analyses du modèle entrainé sur l'ensemble de test : exécutez `deeplearning/analysis.py`

## Apprentissage des émotions combiné avec le genre

Pour aller plus loin, nous avons aussi voulu combiner les détection d'émotions avec la détection du genre de la personne. Pour ce faire, il était plus judicieux qu'une même personne ne se retrouve pas à la fois sur des images du set d'entrainement et du set de test. D'où l'utilisation du fichier `gender_detection/sets_by_participants.py` pour préparer les sets.

L'entrainement du modèle pourra se faire de la même manière qu'avant avec `gender_detection/cnn_model_gender.py`.

L'analyse des deux modèles combinés se fera en exécutant `gender_detection/analyse_both.py` (il faudra préalablement entrainer le modèle cnn_model sur les données générées par sets_by_participants.py)
