import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array
import os

# Define paths for the model and labels
MODEL_PATH = r"C:\Users\mstrg\OneDrive\سطح المكتب\تدريب\python.work\keras_model.h5"
LABELS_PATH = r"C:\Users\mstrg\OneDrive\سطح المكتب\تدريب\python.work\labels.txt"
IMAGE_PATH = r"C:\Users\mstrg\OneDrive\سطح المكتب\animals\ccc.jpg"

# Function to load the labels
def load_labels(labels_path):
    with open(labels_path, "r") as file:
        labels = [line.strip().split(" ", 1)[1] for line in file.readlines()]
    return labels

# Function to preprocess the input image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize image to match model input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Function to predict the class of an image
def predict_class(model, labels, img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    class_name = labels[predicted_class]
    print(f"Predicted Class: {class_name} (Confidence: {confidence:.2f})")
    return class_name

# Main execution
if __name__ == "__main__":
    print("Starting the script...")

    # Check if the image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: File '{IMAGE_PATH}' not found.")
        exit(1)

    # Load the model
    try:
        print("Loading the model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Load the labels
    try:
        print("Loading labels...")
        labels = load_labels(LABELS_PATH)
        print(f"Labels loaded: {labels}")
    except Exception as e:
        print(f"Error loading labels: {e}")
        exit(1)

    # Predict the class of the image
    predict_class(model, labels, IMAGE_PATH)

