import tensorflow as tf
import numpy as np
import random

alpha_ = 0.1

number_of_predictors = 20
number_of_samples = 100
number_of_predictions = 3

#
# set up dataset
#

predictors = [np.random.normal(size = number_of_predictors).astype('f') for i in range(number_of_samples)]
labels = [[random.choice([-1., 1.]) for i in range(number_of_predictions)] for i in range(number_of_samples)]

#
# set up model
#

slope = tf.Variable(tf.random.normal(shape=[number_of_predictors, number_of_predictions]))
intercept = tf.Variable(tf.random.normal(shape=[1, number_of_predictions]))

# model prediction
predictions = tf.subtract(tf.matmul(predictors, slope), intercept)

# regularization
l2_norm = tf.reduce_sum(tf.square(slope))
alpha = tf.constant([alpha_])

# assemble model prediction and regularization
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(predictions, labels))))
loss = tf.add(classification_term, tf.multiply(l2_norm, alpha))

prediction = tf.sign(predictions)
print(f"Predictions: {predictions}")
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), dtype = tf.float32))
print(f"Accuracy: {accuracy}")