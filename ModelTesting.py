import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Parameters
MODEL_PATH = "fruit_classifier.keras"
TEST_PATH = "Datasets\\test"
TRAIN_PATH = "Datasets\\train"
VALID_PATH = "Datasets\\valid"
SUBFOLDERS = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
OUTPUT_DIR = "Outputs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper function to load and preprocess images
def load_and_preprocess_images(folder_path, subfolders, image_size=(224, 224)):
    # Create a mapping from subfolder names to label indices
    label_map = {subfolder: idx for idx, subfolder in enumerate(subfolders)}
    images, labels = [], []
    # Iterate through each subfolder and load images
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize the image to the target size
                img_resized = cv2.resize(img, image_size)
                images.append(img_resized)
                labels.append(label_map[subfolder])
    # Convert to numpy arrays and normalize the images
    images = np.array(images).astype('float32') / 255.0
    labels = np.array(labels)
    return images, labels

# Visualize dataset statistics and a sample image
def visualize_dataset(folder_path, subfolders, output_dir,file_name,Title):
    class_counts = []
    sample_images = []
    
    # Count the number of images in each class and pick a sample image
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        images = os.listdir(subfolder_path)
        class_counts.append(len(images))
        if images:
            img_path = os.path.join(subfolder_path, images[0])
            img = cv2.imread(img_path)
            if img is not None:
                sample_images.append((subfolder, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    # Plot class distribution in pie chart and bar chart
    plt.figure(figsize=(6, 6))
    plt.bar(subfolders, class_counts, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'violet'])
    plt.title(f'{Title} Class Distribution (Bar Chart)')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')

    class_distribution_path = os.path.join(output_dir, file_name)
    plt.savefig(class_distribution_path)
    print(f"Class distribution saved to {class_distribution_path}")

    # Save sample images of each class
    for subfolder, img in sample_images:
        plt.figure()
        plt.imshow(img)
        plt.title(f"Sample Image: {subfolder}")
        plt.axis('off')
        sample_image_path = os.path.join(output_dir, f"sample_{subfolder}.png")
        plt.savefig(sample_image_path)
        print(f"Sample image for {subfolder} saved to {sample_image_path}")

# Load test data
x_test, y_test = load_and_preprocess_images(TEST_PATH, SUBFOLDERS)
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=len(SUBFOLDERS))

# Visualize dataset (Class distribution and sample images)
visualize_dataset(TEST_PATH, SUBFOLDERS, OUTPUT_DIR,"testclass_distribution.png","Test")
visualize_dataset(TRAIN_PATH, SUBFOLDERS, OUTPUT_DIR,"trainclass_distribution.png","Train")
visualize_dataset(VALID_PATH, SUBFOLDERS, OUTPUT_DIR,"validclass_distribution.png","Valid")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test_categorical, verbose=0)
print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")

# Generate predictions
y_pred = np.argmax(model.predict(x_test), axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SUBFOLDERS)
plt.figure(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
confusion_matrix_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(confusion_matrix_path)
print(f"Confusion matrix saved to {confusion_matrix_path}")

# Classification report
report = classification_report(y_test, y_pred, target_names=SUBFOLDERS)
report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"Classification report saved to {report_path}")

# Training and validation metrics (from the log)
log_path = "training_log.csv"
if os.path.exists(log_path):
    import pandas as pd
    log_data = pd.read_csv(log_path)
    
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(log_data['epoch'], log_data['accuracy'], label='Training Accuracy')
    plt.plot(log_data['epoch'], log_data['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(log_data['epoch'], log_data['loss'], label='Training Loss')
    plt.plot(log_data['epoch'], log_data['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    metrics_plot_path = os.path.join(OUTPUT_DIR, "training_metrics.png")
    plt.savefig(metrics_plot_path)
    print(f"Training metrics saved to {metrics_plot_path}")
else:
    print("Training log not found. Skipping metric plots.")
