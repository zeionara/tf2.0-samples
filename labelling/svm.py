import sys
sys.path.append('..')

from utils.numberizers import numberize_labels

import tensorflow as tf
import pandas as pd

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

PATH_TO_DATASET = "datasets/commands.pkl"

# Allow allocation of lots of memory
configp = ConfigProto()
configp.gpu_options.allow_growth = True
session = InteractiveSession(config=configp)

# Read dataset
df = pd.read_pickle(PATH_TO_DATASET)
print("Loaded dataset:")
print(df)

# Numberize labels
df = numberize_labels(df, 'type', "Command")
print("Loaded dataset after numberizing labels:")
print(df)