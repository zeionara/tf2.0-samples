#from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import os
import time

from tensorflow.compat.v1 import ConfigProto

# Allow growth to eliminate error of lacking cuDNN implementation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


path_to_file = "datasets/aneks-shrank.txt"
preview_length = 20
seq_length = 100

# Read, then decode for py2 compat.
text = open(path_to_file, 'r').read()
# length of text is the number of characters in it
print (f'Length of text: {len(text)} characters')

# Print beginning of the text
print(text[:250])

# make vocabulary
vocab = sorted(set(text))
print(f'Vocabulary size: {len(vocab)}')

# map chars to numbers
char2idx = {char: i for i, char in enumerate(vocab)}
idx2char = np.array(vocab)
text_numberized = np.array([char2idx[char] for char in text])

# show fragment of mapped text
print("Fragment of mapped text")
print(f"{repr(text[:preview_length])} --> {text_numberized[:preview_length]}")

# train
examples_per_epoch = len(text) // (seq_length + 1)
dataset = tf.data.Dataset.from_tensor_slices(text_numberized)

print("Fragment of dataset:")
for i in dataset.take(preview_length):
	print(idx2char[i])

sequences = dataset.batch(seq_length + 1, drop_remainder = True)

def split_input_target(chunk):
	input_text = chunk[:-1]
	target_text = chunk[1:]
	return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
	print(f"Input: {repr(''.join(idx2char[input_example.numpy()]))}")
	print(f"Target: {repr(''.join(idx2char[target_example.numpy()]))}")


# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

print(dataset)
print(model.summary())

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
  sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
  sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
  print(f'Sampled indices: {sampled_indices}')
  print(f"Input: \n{repr(''.join(idx2char[input_example_batch[0]]))}")
  print()
  print(f"Next Char Predictions: \n{repr(''.join(idx2char[sampled_indices ]))}")

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=3

print(idx2char)
#history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
print('loading')
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
#tf.train.latest_checkpoint(checkpoint_dir)


model.build(tf.TensorShape([1, None]))
print(model.summary())

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []
  symbs_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])
      symbs_generated.append(predicted_id)

  print([idx2char[s] for s in symbs_generated])
  return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"как заставить гея трахнуть"))

