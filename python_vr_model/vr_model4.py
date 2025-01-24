import os
import pathlib
import random
import numpy as np
import tensorflow as tf

# Dataset path
data_dir = pathlib.Path(__file__).parent / "mini_speech_commands"
print("Data directory:", data_dir)

# Commands (labels)
commands = np.array(["up", "down", "left", "right", "on", "off"])
filenames = list(data_dir.glob("*/*.wav"))
filenames = [str(f) for f in filenames]

# Check if files are loaded
print(f"Total WAV files found: {len(filenames)}")

# Train/Validation/Test Split
random.shuffle(filenames)
num_samples = len(filenames)
train_size = int(0.8 * num_samples)
val_size = int(0.1 * num_samples)
test_size = num_samples - train_size - val_size

train_files = filenames[:train_size]
val_files = filenames[train_size : train_size + val_size]
test_files = filenames[train_size + val_size :]

# Check splits
print(f"Training set size: {len(train_files)}")
print(f"Validation set size: {len(val_files)}")
print(f"Test set size: {len(test_files)}")

# Labels
label_to_index = {name: i for i, name in enumerate(commands)}
BACKGROUND_LABEL = len(commands)  # Additional index for background noise

def get_label(file_path):
    """Labels a given file based on command, else labels as background noise."""
    parts = tf.strings.split(file_path, os.path.sep)
    folder = parts[-2]  # Folder name is second to last element of the path

    # Check if the folder name matches any command
    is_command = tf.reduce_any(tf.equal(commands, folder))

    # Use TensorFlow operations to find the label
    label_index = tf.argmax(tf.cast(tf.equal(commands, folder), tf.int32))
    return tf.cond(
        is_command,
        lambda: tf.cast(label_index, tf.int32),  # Command label
        lambda: tf.cast(BACKGROUND_LABEL, tf.int32),  # Background label
    )


@tf.function
def label_wav(file_path):
    """Function that labels a wav file."""
    # Get the label
    label = get_label(file_path)

    # Load the WAV file
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)  # Convert from [samples, 1] to [samples]

    return audio, label

# Test the labeling function with a sample file
if len(train_files) > 0:
    for file in train_files[:5]:  # Test on 5 training files
        audio, label = label_wav(file)
        print(f"File: {file}, Label: {label.numpy()}, Audio Shape: {audio.shape}")
