import os
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# Retrieves tensorflow dataset that we use for original training

DATASET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "mini_speech_commands.zip"
)
FILE_NAME = 'mini_speech_commands.zip'

# Auto downloads dataset by default. If already in current parent directory, does not re-download
if not os.path.exists("mini_speech_commands"):
    data_path = tf.keras.utils.get_file(
        fname= FILE_NAME,
        origin=DATASET_URL,
        extract=True,
        cache_dir = ".",  # download to current directory
        cache_subdir = ''  # no subdirectory
    )
    # path to the dataset
    data_dir = pathlib.Path(data_path).parent / "mini_speech_commands"

else:
    data_dir = pathlib.Path("mini_speech_commands")

# checks if path is properly set (donwloaded and extracted correctly)
print("Data directory:", data_dir)

# paths of each word we want to use
commands = np.array(['up', 'down', 'left', 'right', 'on', 'off'])
# search for each wav file within each command subfile
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.wav')

# Train Validate Test Split
random.shuffle(filenames)
num_samples = len(filenames)

train_size = int(0.8 * num_samples)
val_size = int(0.1 * num_samples)
test_size = num_samples - train_size - val_size

train_files = filenames[:train_size]
val_files = filenames[train_size: train_size + val_size]
test_files = filenames[train_size + val_size:]

# checks if properly separated
print(f"Training set size: {len(train_files)}")
print(f"Validation set size: {len(val_files)}")
print(f"Test set size: {len(test_files)}")


# classifying files

# labels
label_to_index = dict((name, i) for i, name in enumerate(commands))
BACKGROUND_LABEL = len(commands)  # add additional index for background

def get_label(file_path):
    """Labels a given file based on command, else labels as background noise"""
    parts = tf.strings.split(file_path, os.path.sep)
    folder = parts[-2]  # folder name is the lat index
    # If label in commands, return label. Else return as background (copy and paste)
    return tf.cond(
        tf.reduce_any(tf.equal(folder, commands)),
        lambda: tf.cast(label_to_index[folder.numpy().decode('utf-8')], tf.int32),
        lambda: tf.cast(BACKGROUND_LABEL, tf.int32)
    )

# labeling function
@tf.function
def label_wav(file_path):
    """Function that labels a wav file"""
    # Load audio file
    # run the get_label function in-place using tf.numpy_function(more efficient)
    # label audio
    label = tf.numpy_function(get_label, [file_path], tf.int32)
    label.set_shape(()) # set shape of tensor label to a scalar(int)
    audio_binary = tf.io.read_file(file_path)
    audio, __ = tf.audio.decode_wav(contents = audio_binary)
    audio = tf.squeeze(audio, axis = 1) # reduces the audio to one dimension
    return audio


