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
