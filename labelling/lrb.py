import numpy as np
import tensorflow as tf
import keras
from tensorflow.python.keras.datasets import fashion_mnist #this is our dataset
from keras.callbacks import ModelCheckpoint

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
dataset = tf.data.Dataset.from_tensor_slices((pd.DataFrame(df.text.tolist()).to_numpy(), tf.one_hot(df.type.tolist(), 3)))
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

dataset = dataset.shuffle(10)
train = dataset.take(train_size)
dataset = dataset.skip(train_size)
test = dataset.take(test_size)
dataset = dataset.skip(test_size)
val = dataset.take(test_size)
trn_x = np.array([list(sample.numpy()) for sample,_ in train])
trn_y = np.array([list(label.numpy()) for _,label in train])
#print(np.array(trn_x).shape)
val_x = np.array([list(sample.numpy()) for sample,_ in val])
val_y = np.array([list(label.numpy()) for _,label in val])

tst_x = np.array([list(sample.numpy()) for sample,_ in test])
tst_y = np.array([list(label.numpy()) for _,label in test])

# important constants
batch_size = 128
epochs = 20
n_classes = 10
learning_rate = 0.1
width = 28 # of our images
height = 28 # of our images

fashion_labels =["Shirt/top","Trousers","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
#indices 0 1 2 3 4 5 6 7 8 9
# Next, we load our fashion data set,
# load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# normalize the features for better training
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# flatten the feature set for use by the training algorithm
x_train = x_train.reshape((60000, width * height))
x_test = x_test.reshape((10000, width * height))

split = 50000
#split training sets into training and validation sets
(x_train, x_valid) = x_train[:split], x_train[split:]
(y_train, y_valid) = y_train[:split], y_train[split:]

# then convert back to numpy as we cannot combine numpy
# and tensors as input to keras later
y_train_ohe = tf.one_hot(y_train, depth=n_classes).numpy()
y_valid_ohe = tf.one_hot(y_valid, depth=n_classes).numpy()
y_test_ohe = tf.one_hot(y_test, depth=n_classes).numpy()
#or use tf.keras.utils.to_categorical(y_train,10)

# show difference between original label and one-hot-encoded label
i=5
print(y_train[i]) # 'ordinairy' number value of label at index i
print (tf.one_hot(y_train[i], depth=n_classes))# same value as a 1. in correct position in an length 10 1D tensor
print(y_train_ohe[i]) # same value as a 1. in correct position in an length 10 1D numpy arr

# model definition (the canonical Google way)
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
model.call(dummy_x)
checkpointer = ModelCheckpoint(filepath="./model.weights.best.hdf5", verbose=2, save_best_only=True, save_weights_only=True)
# train the model
#print(np.array(x_train).shape)
#print(type())
#print(np.array(y_train_ohe).shape)

print(val_x)
print(val_y)

model.fit(trn_x, trn_y, batch_size=batch_size, epochs=epochs,validation_data=(val_x, val_y), callbacks=[checkpointer], verbose=2)
#load model with the best validation accuracy