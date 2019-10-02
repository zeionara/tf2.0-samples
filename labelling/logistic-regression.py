import sys, os
sys.path.append('..')

from utils.numberizers import numberize_labels
from utils.filters import split_by_label_values

import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession, Session

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D

#
# Set up environment
#

# Uncomment the following line to output all rows of pandas dataframes
#pd.set_option('display.max_rows', df.type.size)

PATH_TO_DATASET = "datasets/commands.pkl"
PATH_TO_IMAGES = "images"
BASE_LABEL = 'Command'

BATCH_SIZE = 10


NUMBER_OF_PREDICTORS = 100
NUMBER_OF_PREDICTIONS = 1

GAMMA = -20.0
LEARNING_RATE = 0.01

NUMER_OF_BATCHES = 10

# Allow allocation of lots of memory
configp = ConfigProto()
configp.gpu_options.allow_growth = True

#tf.compat.v1.disable_eager_execution()

#
# Set up dataset
#

# Read dataset
df = pd.read_pickle(PATH_TO_DATASET)

# Drop nans and reset index
df.dropna(inplace = True, how = 'any')
df = df.reset_index(drop=True)

# Add dummy columns
#df = pd.get_dummies(df, columns = ['type'])

# Update labels
#dummy_columns = ['type_Command', 'type_IndirectCommand', 'type_Request']
#df[dummy_columns] = df[dummy_columns].replace({0: -1})

# Output loaded dataset
print("Loaded dataset:")
print(df)
print("Datatypes of columns:")
print(df.dtypes)

types = df.type.unique()
type_dict = dict(zip(types, range(len(types))))
df.type = df.replace({"type": type_dict})

DATASET_SIZE = df.shape[0]

#Next we could create tf.data.Dataset if we would use eager execution
dataset = tf.data.Dataset.from_tensor_slices((pd.DataFrame(df.text.tolist()).to_numpy(), tf.one_hot(df.type.tolist(), 3)))

#print(dir(dataset))

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

dataset = dataset.shuffle(10)
train = dataset.take(train_size)
dataset = dataset.skip(train_size)
test = dataset.take(test_size)
dataset = dataset.skip(test_size)
val = dataset.take(val_size)

class LogisticRegression(tf.keras.Model):
	def __init__(self, num_classes):
		super(LogisticRegression, self).__init__() # call the constructor of the parent class (Model)
		self.dense = tf.keras.layers.Dense(num_classes) #create an empty layer called dense with 10 elements.
	
	def call(self, inputs, training=None, mask=None): # required for our forward pass
		output = self.dense(inputs) # copy training inputs into our layer
		# softmax op does not exist on the gpu, so force execution on the CPU
		#with tf.device('/cpu:0'):
		output = tf.nn.softmax(output) # softmax is near one for maximum value in output
		# and near zero for the other values.
		return output

# build the model
model = LogisticRegression(3)
# compile the model
#optimiser = tf.train.GradientDescentOptimizer(learning_rate)
optimiser =tf.keras.optimizers.Adam() #not supported in eager execution mode.
model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'], )
# TF Keras tries to use the entire dataset to determine the shape without this step when using .fit()
# So, use one sample of the provided input dataset size to determine input/output shapes for the model
dummy_x = tf.zeros((1, 100))
print(dummy_x)
model.call(dummy_x)
checkpointer = ModelCheckpoint(filepath="./model.weights.best.hdf5", verbose=2, save_best_only=True, save_weights_only=True)
# train the model
print(dir(train))

trn_x = np.array([list(sample.numpy()) for sample,_ in train])
trn_y = np.array([list(label.numpy()) for _,label in train])
#print(np.array(trn_x).shape)
val_x = np.array([list(sample.numpy()) for sample,_ in val])
val_y = np.array([list(label.numpy()) for _,label in val])

tst_x = np.array([list(sample.numpy()) for sample,_ in test])
tst_y = np.array([list(label.numpy()) for _,label in test])

#print([list(label.numpy()) for _,label in train])
model.fit(trn_x, trn_y, batch_size=128, epochs=30, validation_data=(val_x, val_y), callbacks=[checkpointer], verbose=2)
#load model with the best validation accuracy
model.load_weights("./model.weights.best.hdf5")
# evaluate the model on the test set
scores = model.evaluate(tst_x, tst_y, batch_size, verbose=2)
print("Final test loss and accuracy :", scores)
#y_predictions = model.predict(x_test)

#print(new_data[0])

# Numberize labels
#df = numberize_labels(df, 'type', BASE_LABEL, numeric_labels = [1, -1], label_type = int)
#print("Loaded dataset after numberizing labels:")
#print(df)
# Split by label values
#dfs = split_by_label_values(df, 'type')
#for command_type in dfs.keys():
#	print(f"Commands with type {command_type}:")
#	print(dfs[command_type])

#for BATCH_SIZE in np.arange(10, 100, 10):

# fig, axs = plt.subplots(2)
# plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.7, left=0.05, right=0.95)

# fig.set_figwidth(19)
# fig.set_figheight(10)
# fig.suptitle(f"Results of labelling commands as '{BASE_LABEL}' using non-linear SVM with batch size {BATCH_SIZE}", fontsize=16)


# #	for GAMMA in np.arange(-20.0, 20.0, 1.0, dtype = np.float32):
# print(f"GAMMA = {GAMMA}")

# session = Session(config=configp)
# # Set up placeholders for training data
# print(f"Setting up placeholders...")
# x_data = tf.compat.v1.placeholder(shape = [None, NUMBER_OF_PREDICTORS], dtype=tf.float32)
# y_target = tf.compat.v1.placeholder(shape = [None, NUMBER_OF_PREDICTIONS], dtype = tf.float32)
# prediction_grid = tf.compat.v1.placeholder(shape = [None, NUMBER_OF_PREDICTORS], dtype = tf.float32)
# b = tf.Variable(tf.compat.v1.random_normal(shape=[NUMBER_OF_PREDICTIONS, BATCH_SIZE]))

# gamma = tf.compat.v1.constant(GAMMA)
# dist = tf.compat.v1.reduce_sum(tf.compat.v1.square(x_data), NUMBER_OF_PREDICTIONS)
# dist = tf.compat.v1.reshape(dist, [-1, 1])
# sq_dists = tf.compat.v1.add(tf.compat.v1.subtract(dist, tf.compat.v1.multiply(2., tf.compat.v1.matmul(x_data, tf.compat.v1.transpose(x_data)))), tf.compat.v1.transpose(dist))
# kernel = tf.compat.v1.exp(tf.compat.v1.multiply(gamma, tf.compat.v1.abs(sq_dists)))

# # compute loss for the dual optimization problem
# print("Setting up losses...")
# model_output = tf.compat.v1.matmul(b, kernel)
# first_term = tf.compat.v1.reduce_sum(b)
# b_vec_cross = tf.compat.v1.matmul(tf.compat.v1.transpose(b), b)
# y_target_cross = tf.compat.v1.matmul(tf.compat.v1.transpose(y_target), y_target)
# second_term = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(kernel, tf.compat.v1.multiply(b_vec_cross, y_target_cross)))
# loss = tf.compat.v1.negative(tf.compat.v1.subtract(first_term, second_term))

# rA = tf.compat.v1.reshape(tf.compat.v1.reduce_sum(tf.compat.v1.square(x_data), 1), [-1, 1])
# rB = tf.compat.v1.reshape(tf.compat.v1.reduce_sum(tf.compat.v1.square(prediction_grid)), [-1, 1])
# pred_sq_dist = tf.compat.v1.add(tf.compat.v1.subtract(rA, tf.compat.v1.multiply(2., tf.compat.v1.matmul(x_data, tf.compat.v1.transpose(prediction_grid)))), tf.transpose(rB))
# pred_kernel = tf.compat.v1.exp(tf.compat.v1.multiply(gamma, tf.compat.v1.abs(pred_sq_dist)))
# prediction_output = tf.compat.v1.matmul(tf.compat.v1.multiply(tf.compat.v1.transpose(y_target), b), pred_kernel)

# prediction = tf.compat.v1.sign(prediction_output - tf.compat.v1.reduce_mean(prediction_output))
# accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(tf.compat.v1.equal(tf.compat.v1.squeeze(prediction), tf.compat.v1.squeeze(y_target)), tf.float32))

# # Train
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARNING_RATE)
# train_step = optimizer.minimize(loss)

# # Start training
# init = tf.compat.v1.global_variables_initializer()
# session.run(init)

# # Really start training
# print("Training...")
# loss_vec = []
# batch_accuracy = []
# for i in range(NUMER_OF_BATCHES):
# 	rand_indices = np.random.choice(df.text.size, size = BATCH_SIZE)
# 	#print(f"Step {i}: selected items with indices \n{rand_indices}")
# 	#print(f"Step {i}: pre-selected items \n{df.text[rand_indices]}")
# 	rand_x = np.transpose(np.dstack(df.text[rand_indices].to_numpy())[0])
# 	#print(f"Step {i}: selected items \n{rand_x}\n with shape {rand_x.shape}")
# 	rand_y = np.transpose([df.type_Request[rand_indices]])
# 	#print(f"Step {i}: corresponding true labels are \n{rand_y}")
# 	session.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
# 	temp_loss =session.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
# 	loss_vec.append(temp_loss)
# 	acc_temp = session.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
# 	batch_accuracy.append(acc_temp)

# print(f"Computed losses: \n{loss_vec}")
# print(f"Computed accuracies: \n{batch_accuracy}")

# #
# # Draw plots
# #

# axs[0].plot(list(map(lambda a: a*100, batch_accuracy)), label = f'Gamma = {GAMMA}')
# axs[0].set_title('Accuracy')
# #axs[0].set_yscale('log')
# axs[0].set_xlabel('Batch number')
# axs[0].set_ylabel('Accuracy (%)')
# axs[0].yaxis.set_major_formatter(ScalarFormatter())
# axs[0].yaxis.set_minor_formatter(ScalarFormatter())

# axs[1].plot(list(map(lambda a: a*100, loss_vec)), label = f'Gamma = {GAMMA}')
# axs[1].set_title('Loss')
# #axs[1].set_yscale('log')
# axs[1].set_xlabel('Batch number')
# axs[1].set_ylabel('Loss')
# axs[1].yaxis.set_major_formatter(ScalarFormatter())
# axs[1].yaxis.set_minor_formatter(ScalarFormatter())

# axs[0].legend(ncol=3)
# axs[1].legend(ncol=3)
# fig.savefig(os.path.join(PATH_TO_IMAGES, f'svm-{BATCH_SIZE}.png'))
