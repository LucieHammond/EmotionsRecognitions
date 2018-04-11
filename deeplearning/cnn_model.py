"""
Ce fichier construit un modèle de réseau neuronal convolutif (CNN) avec Tensorflow
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from constants import SIZE, EMOTIONS, TEMP_PATH

tf.logging.set_verbosity(tf.logging.INFO)


def model(features, labels, mode):
    """ Model function for CNN """

    # Input Layer
    # Reshape inputs to 4-D tensor: [batch_size, width, height, channels]
    # Face images are 200 x 200 pixels, and have one color channel (SIZE = 200)
    input_layer = tf.reshape(features["x"], [-1, SIZE, SIZE, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 200, 200, 1]
    # Output Tensor Shape: [batch_size, 200, 200, 64]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 4x4 filter and stride of 4
    # Input Tensor Shape: [batch_size, 200, 200, 64]
    # Output Tensor Shape: [batch_size, 50, 50, 64]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 50, 50, 64]
    # Output Tensor Shape: [batch_size, 50, 50, 128]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 50, 50, 128]
    # Output Tensor Shape: [batch_size, 25, 25, 128]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 25, 25, 128]
    # Output Tensor Shape: [batch_size, 25 * 25 * 128]
    pool2_flat = tf.reshape(pool2, [-1, 25 * 25 * 128])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 35 * 35 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.5 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 7]
    logits = tf.layers.dense(inputs=dropout, units=len(EMOTIONS))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    print("Load training dataset...")
    train_inputs = np.load(TEMP_PATH + '/prepared_sets/train_inputs.npy')
    train_labels = np.load(TEMP_PATH + '/prepared_sets/train_labels.npy')
    print("Load test dataset...")
    eval_inputs = np.load(TEMP_PATH + '/prepared_sets/test_inputs.npy')
    eval_labels = np.load(TEMP_PATH + '/prepared_sets/test_labels.npy')

    # Create the Estimator
    emotions_classifier = tf.estimator.Estimator(
        model_fn=model, model_dir=TEMP_PATH + '/cnn_model')

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    for i in range(10):

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_inputs},
            y=train_labels,
            batch_size=50,
            num_epochs=None,
            shuffle=True)
        emotions_classifier.train(
            input_fn=train_input_fn,
            steps=100,
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_inputs},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = emotions_classifier.evaluate(input_fn=eval_input_fn)
        print("\nRésultats :")
        print(eval_results)


if __name__ == "__main__":
    tf.app.run()