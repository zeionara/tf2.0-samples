import sys, os
sys.path.append('..')

from utils.numberizers import numberize_labels
from utils.filters import split_by_label_values

import tensorflow as tf
import pandas as pd

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession, Session

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

PATH_TO_DATASET = "datasets/commands.pkl"
PATH_TO_IMAGES = "images"
BASE_LABEL = 'Command'

BATCH_SIZE = 30


NUMBER_OF_PREDICTORS = 100
NUMBER_OF_PREDICTIONS = 1

GAMMA = -20.0
LEARNING_RATE = 0.01

NUMER_OF_BATCHES = 20


# Allow allocation of lots of memory
configp = ConfigProto()
configp.gpu_options.allow_growth = True

tf.compat.v1.disable_eager_execution()

# Read dataset
df = pd.read_pickle(PATH_TO_DATASET)
#df.dropna(inplace = True, how = 'any')
print("Loaded dataset:")
df = df.reset_index(drop=True)
pd.set_option('display.max_rows', df.type.size)
print(df)

# Numberize labels
df = numberize_labels(df, 'type', BASE_LABEL, numeric_labels = [1, -1], label_type = int)
print("Loaded dataset after numberizing labels:")
print(df)
# Split by label values
dfs = split_by_label_values(df, 'type')
for command_type in dfs.keys():
	print(f"Commands with type {command_type}:")
	print(dfs[command_type])

for BATCH_SIZE in np.arange(10, 100, 10):

	fig, axs = plt.subplots(2)
	plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.7, left=0.05, right=0.95)

	fig.set_figwidth(19)
	fig.set_figheight(10)
	fig.suptitle(f"Results of labelling commands as '{BASE_LABEL}' using non-linear SVM with batch size {BATCH_SIZE}", fontsize=16)


	for GAMMA in np.arange(-20.0, 20.0, 1.0, dtype = np.float32):
		print(f"GAMMA = {GAMMA}")
		
		session = Session(config=configp)
		# Set up placeholders for training data
		print(f"Setting up placeholders...")
		x_data = tf.compat.v1.placeholder(shape = [None, NUMBER_OF_PREDICTORS], dtype=tf.float32)
		y_target = tf.compat.v1.placeholder(shape = [None, NUMBER_OF_PREDICTIONS], dtype = tf.float32)
		prediction_grid = tf.compat.v1.placeholder(shape = [None, NUMBER_OF_PREDICTORS], dtype = tf.float32)
		b = tf.Variable(tf.compat.v1.random_normal(shape=[NUMBER_OF_PREDICTIONS, BATCH_SIZE]))

		gamma = tf.compat.v1.constant(GAMMA)
		dist = tf.compat.v1.reduce_sum(tf.compat.v1.square(x_data), NUMBER_OF_PREDICTIONS)
		dist = tf.compat.v1.reshape(dist, [-1, 1])
		sq_dists = tf.compat.v1.add(tf.compat.v1.subtract(dist, tf.compat.v1.multiply(2., tf.compat.v1.matmul(x_data, tf.compat.v1.transpose(x_data)))), tf.compat.v1.transpose(dist))
		kernel = tf.compat.v1.exp(tf.compat.v1.multiply(gamma, tf.compat.v1.abs(sq_dists)))

		# compute loss for the dual optimization problem
		print("Setting up losses...")
		model_output = tf.compat.v1.matmul(b, kernel)
		first_term = tf.compat.v1.reduce_sum(b)
		b_vec_cross = tf.compat.v1.matmul(tf.compat.v1.transpose(b), b)
		y_target_cross = tf.compat.v1.matmul(tf.compat.v1.transpose(y_target), y_target)
		second_term = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(kernel, tf.compat.v1.multiply(b_vec_cross, y_target_cross)))
		loss = tf.compat.v1.negative(tf.compat.v1.subtract(first_term, second_term))

		rA = tf.compat.v1.reshape(tf.compat.v1.reduce_sum(tf.compat.v1.square(x_data), 1), [-1, 1])
		rB = tf.compat.v1.reshape(tf.compat.v1.reduce_sum(tf.compat.v1.square(prediction_grid)), [-1, 1])
		pred_sq_dist = tf.compat.v1.add(tf.compat.v1.subtract(rA, tf.compat.v1.multiply(2., tf.compat.v1.matmul(x_data, tf.compat.v1.transpose(prediction_grid)))), tf.transpose(rB))
		pred_kernel = tf.compat.v1.exp(tf.compat.v1.multiply(gamma, tf.compat.v1.abs(pred_sq_dist)))
		prediction_output = tf.compat.v1.matmul(tf.compat.v1.multiply(tf.compat.v1.transpose(y_target), b), pred_kernel)

		prediction = tf.compat.v1.sign(prediction_output - tf.compat.v1.reduce_mean(prediction_output))
		accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(tf.compat.v1.equal(tf.compat.v1.squeeze(prediction), tf.compat.v1.squeeze(y_target)), tf.float32))

		# Train
		optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARNING_RATE)
		train_step = optimizer.minimize(loss)

		# Start training
		init = tf.compat.v1.global_variables_initializer()
		session.run(init)

		# Really start training
		print("Training...")
		loss_vec = []
		batch_accuracy = []
		for i in range(NUMER_OF_BATCHES):
			rand_indices = np.random.choice(df.text.size, size = BATCH_SIZE)
			#print(f"Step {i}: selected items with indices \n{rand_indices}")
			#print(f"Step {i}: pre-selected items \n{df.text[rand_indices]}")
			rand_x = np.transpose(np.dstack(df.text[rand_indices].to_numpy())[0])
			#print(f"Step {i}: selected items \n{rand_x}\n with shape {rand_x.shape}")
			rand_y = np.transpose([df.type[rand_indices]])
			#print(f"Step {i}: corresponding true labels are \n{rand_y}")
			session.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
			temp_loss =session.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
			loss_vec.append(temp_loss)
			acc_temp = session.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
			batch_accuracy.append(acc_temp)

		print(f"Computed losses: \n{loss_vec}")
		print(f"Computed accuracies: \n{batch_accuracy}")

		#
		# Draw plots
		#

		axs[0].plot(list(map(lambda a: a*100, batch_accuracy)), label = f'Gamma = {GAMMA}')
		axs[0].set_title('Accuracy')
		#axs[0].set_yscale('log')
		axs[0].set_xlabel('Batch number')
		axs[0].set_ylabel('Accuracy (%)')
		axs[0].yaxis.set_major_formatter(ScalarFormatter())
		axs[0].yaxis.set_minor_formatter(ScalarFormatter())

		axs[1].plot(list(map(lambda a: a*100, loss_vec)), label = f'Gamma = {GAMMA}')
		axs[1].set_title('Loss')
		#axs[1].set_yscale('log')
		axs[1].set_xlabel('Batch number')
		axs[1].set_ylabel('Loss')
		axs[1].yaxis.set_major_formatter(ScalarFormatter())
		axs[1].yaxis.set_minor_formatter(ScalarFormatter())

	axs[0].legend(ncol=3)
	axs[1].legend(ncol=3)
	fig.savefig(os.path.join(PATH_TO_IMAGES, f'svm-{BATCH_SIZE}.png'))
