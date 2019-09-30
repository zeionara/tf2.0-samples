import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import math

PATH_TO_IMAGES = "images"

def function(x):
	return x*x + x


def descent(weights_, step, length):
	weights = tf.Variable(weights_)
	history = [weights.numpy()]
	
	for i in range(length):
		with tf.GradientTape() as tape:
		 		loss = function(weights)

		grad = tape.gradient(loss, weights)

		weights.assign(tf.subtract(weights, grad * step))
		history.append(weights.numpy())

	return history

descent_history = descent([[7.5, -7.5]], step = 0.2, length = 5)

xs = np.arange(-10, 10, 0.001)
plt.plot(xs, list(map(lambda x: function(x), xs)), label = f'Descent')

for point_sets in descent_history:
	plt.scatter(point_sets[0][0], function(point_sets[0][0]), s = 100, c = ['red'])

for point_sets in descent_history:
	plt.scatter(point_sets[0][1], function(point_sets[0][1]), s = 100, c = ['blue'])

plt.savefig(os.path.join(PATH_TO_IMAGES, f'descent.png'))