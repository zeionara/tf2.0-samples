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

PATH_TO_DATASET = "datasets/commands.pkl"
PATH_TO_MODEL = "models/logistic-regression.hdf5"
PATH_TO_IMAGES = "images"
BASE_LABEL = 'Command'

BATCH_SIZE = 10
NUMBER_OF_PREDICTORS = 100
NUMBER_OF_PREDICTIONS = 3
NUMER_OF_BATCHES = 10
NUMBER_OF_EPOCHS = 30

TRAIN_RELATIVE_SIZE = 0.8
TEST_RELATIVE_SIZE = 0.2
#VALIDATION_RELATIVE_SIZE = 0.15

SHUFFLE_BUFFER_SIZE = 20

CV_FOLDS = 10

# Allow allocation of lots of memory
configp = ConfigProto()
configp.gpu_options.allow_growth = True

#
# Set up dataset
#

# Read dataset
df = pd.read_pickle(PATH_TO_DATASET)

# Drop nans and reset index
df.dropna(inplace = True, how = 'any')
df = df.reset_index(drop=True)

# Replace string types with numbers
types = df.type.unique()
type_dict = dict(zip(types, range(len(types))))
df.type = df.replace({"type": type_dict})

DATASET_SIZE = df.shape[0]

#Create tf.data.Dataset from pandas
dataset = tf.data.Dataset.from_tensor_slices((pd.DataFrame(df.text.tolist()).to_numpy(), tf.one_hot(df.type.tolist(), 3)))

train_size = int(TRAIN_RELATIVE_SIZE * DATASET_SIZE)
#val_size = int(VALIDATION_RELATIVE_SIZE * DATASET_SIZE)
test_size = int(TEST_RELATIVE_SIZE * DATASET_SIZE)

dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
train = dataset.take(train_size)
dataset = dataset.skip(train_size)
test = dataset.take(test_size)
#dataset = dataset.skip(test_size)
#val = dataset.take(val_size)

class LogisticRegression(tf.keras.Model):
	def __init__(self, num_classes):
		super(LogisticRegression, self).__init__()
		self.dense = tf.keras.layers.Dense(num_classes)
	
	def call(self, inputs, training=None, mask=None):
		output = self.dense(inputs)
		output = tf.nn.softmax(output)
		return output

folds = []
fold_size = DATASET_SIZE // CV_FOLDS
for i in range(CV_FOLDS):
	folds.append(train.take(fold_size))
	train.skip(fold_size)

test_x = np.array([list(sample.numpy()) for sample, _ in test])
test_y = np.array([list(label.numpy()) for _,label in test])

print(len(folds))
results={}
test_results={}
for BATCH_SIZE in range(10, 30, 5):
	all_scores = []
	test_scores = []
	for i in range(len(folds)):
		# build the model
		model = LogisticRegression(NUMBER_OF_PREDICTIONS)
		optimiser =tf.keras.optimizers.Adam()
		model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'], )

		# check compiled model
		dummy_x = tf.zeros((1, NUMBER_OF_PREDICTORS))
		model.call(dummy_x)

		# Make checkpointer
		checkpointer = ModelCheckpoint(filepath=PATH_TO_MODEL, verbose=2, save_best_only=True, save_weights_only=True)

		# Fine grain datasets
		train_x = np.array([list(sample.numpy()) for j in range(len(folds)) for sample,_ in folds[j] if j != i])
		train_y = np.array([list(label.numpy()) for j in range(len(folds)) for _,label in folds[j] if j != i])
		validate_x = np.array([list(sample.numpy()) for sample,_ in folds[i]])
		validate_y = np.array([list(label.numpy()) for _,label in folds[i]])
		

		# Fit and test model
		model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=NUMBER_OF_EPOCHS, validation_data=(validate_x, validate_y), callbacks=[checkpointer], verbose=2)
		
		scores = model.evaluate(validate_x, validate_y, 128, verbose=2)
		all_scores.append(scores)
		test_scores.append(model.evaluate(test_x, test_y, 128, verbose=2))

	results[BATCH_SIZE] = tf.reduce_mean(tf.constant(all_scores), axis=0).numpy()
	test_results[BATCH_SIZE] = tf.reduce_mean(tf.constant(test_scores), axis=0).numpy()

print(results)
print(test_results)

fig, axs = plt.subplots(2)
plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.7, left=0.05, right=0.95)

fig.set_figwidth(19)
fig.set_figheight(10)
fig.suptitle(f"Logistic regression", fontsize=16)

axs[0].plot(list(results.keys()), [value[1] for value in results.values()], label = f'Validation accuracy')
axs[0].plot(list(test_results.keys()), [value[1] for value in test_results.values()], label = f'Test Accuracy')
axs[0].set_title('Accuracy')
#axs[0].set_yscale('log')
axs[0].set_xlabel('Batch size')
axs[0].set_ylabel('Accuracy')
axs[0].yaxis.set_major_formatter(ScalarFormatter())
axs[0].yaxis.set_minor_formatter(ScalarFormatter())

axs[1].plot(list(results.keys()), [value[0] for value in results.values()], label = f'Validation Loss')
axs[1].plot(list(test_results.keys()), [value[0] for value in test_results.values()], label = f'Test Loss')
axs[1].set_title('Loss')
#axs[1].set_yscale('log')
axs[1].set_xlabel('Batch size')
axs[1].set_ylabel('Loss')
axs[1].yaxis.set_major_formatter(ScalarFormatter())
axs[1].yaxis.set_minor_formatter(ScalarFormatter())

axs[0].legend()
axs[1].legend()
fig.savefig(os.path.join(PATH_TO_IMAGES, f'logistic-regression.png'))

	#model.load_weights(PATH_TO_MODEL)
#test_x = np.array([list(sample.numpy()) for sample,_ in test])
#	test_y = np.array([list(label.numpy()) for _,label in test])
##scores = model.evaluate(test_x, test_y, 128, verbose=2)
#print("Final test loss and accuracy :", scores)
