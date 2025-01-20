import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import pandas as pd
import time

# Custom callback to measure training time per epoch
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []  # List to store epoch times

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()  # Record start time of epoch

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time  # Calculate epoch duration
        self.epoch_times.append(epoch_time)  # Store duration

# Function to load and preprocess images from folders
def load_and_preprocess_images(folder_path, subfolders, image_size=(224, 224)):
    """
    Loads images from subfolders, resizes them, and maps labels.
    
    Args:
        folder_path (str): Path to the folder containing subfolders for each class.
        subfolders (list): List of subfolder names corresponding to class labels.
        image_size (tuple): Target size for image resizing.

    Returns:
        tuple: Preprocessed images as numpy array and corresponding labels.
    """
    label_map = {subfolder: idx for idx, subfolder in enumerate(subfolders)}  # Create label-to-index mapping
    images, labels = [], []
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)
            img = cv2.imread(img_path)  # Read image
            if img is not None:
                img_resized = cv2.resize(img, image_size)  # Resize image
                images.append(img_resized)
                labels.append(label_map[subfolder])  # Append label
    images = np.array(images).astype('float32') / 255.0  # Normalize images to [0, 1]
    labels = np.array(labels)
    return images, labels

# Define root directory and paths to training and validation datasets
root_dir = os.getcwd()
train_path = os.path.join(root_dir, "Datasets", "train")
valid_path = os.path.join(root_dir, "Datasets", "valid")
subfolders = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']  # Class names

# Load and preprocess training and validation data
x_train, y_train = load_and_preprocess_images(train_path, subfolders)
x_valid, y_valid = load_and_preprocess_images(valid_path, subfolders)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(subfolders))
y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=len(subfolders))

# Shuffle training and validation datasets
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train, random_state=42)
x_valid, y_valid = shuffle(x_valid, y_valid, random_state=42)

# Define the base model using MobileNetV2 pretrained on ImageNet
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add global average pooling layer
x = Dense(len(subfolders), activation="softmax")(x)  # Add dense layer for classification

# Define the final model
model = Model(inputs=base_model.input, outputs=x)

# Freeze the base model layers to prevent training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callback to record training time
time_callback = TimeHistory()

# Train the model
history = model.fit(
    x_train, y_train, 
    epochs=10,  # Number of epochs
    validation_data=(x_valid, y_valid),  # Validation dataset
    batch_size=128,  # Batch size
    callbacks=[time_callback]  # Callback to monitor time
)

# Save the trained model to a file
model.save("fruit_classifier.keras")

# Save training history to a CSV file
history_data = {
    "epoch": list(range(1, 11)),
    "accuracy": history.history["accuracy"],  # Training accuracy
    "loss": history.history["loss"],  # Training loss
    "val_accuracy": history.history["val_accuracy"],  # Validation accuracy
    "val_loss": history.history["val_loss"],  # Validation loss
    "time": time_callback.epoch_times  # Time taken per epoch
}

# Convert history data to a DataFrame and save it
history_df = pd.DataFrame(history_data)
history_df.to_csv("training_log.csv", index=False)
print("Training log saved to training_log.csv")
