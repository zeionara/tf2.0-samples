import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

configp = ConfigProto()
configp.gpu_options.allow_growth = True
session = InteractiveSession(config=configp)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(f"x_train shape: {x_train.shape}")
print(f"y_train: {y_train.shape}")

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print(f"x_train shape: {x_train.shape}")
print(f"y_train: {y_train.shape}")

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
print(f"train dataset shuffled and batched: {train_ds}")
print(f"test dataset batched: {test_ds}")

class MyModel(Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.conv1 = Conv2D(32, 3, activation = 'relu')
		self.flatten = Flatten()
		self.d1 = Dense(128, activation = 'relu')
		self.d2 = Dense(10, activation = 'softmax')

	def call(self, x):
		x = self.conv1(x)
		x = self.flatten(x)
		x = self.d1(x)
		return self.d2(x)

model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
	with tf.GradientTape() as tape:
		predictions = model(images)
		loss = loss_object(labels, predictions)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	train_loss(loss)
	train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
	predictions = model(images)
	loss = loss_object(labels, predictions)

	test_loss(loss)
	test_accuracy(labels, predictions)

EPOCHS = 5
for epoch in range(EPOCHS):
	for images, labels in train_ds:
		train_step(images, labels)

	for test_images, test_labels in test_ds:
		test_step(test_images, test_labels)

	print(f'Epoch {epoch+1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()*100}, Test Loss: {test_accuracy.result()}, Test Accuracy: {test_accuracy.result()*100}')

	train_loss.reset_states()
	train_accuracy.reset_states()
	test_loss.reset_states()
	test_accuracy.reset_states()