import cv2
import pandas as pd
import numpy as np

def preprocess_image(filepath, target_size=(128, 128)):
    image = cv2.imread(filepath)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def get_image_data(path, x_label="filepath", y_label="label"):
    df = pd.read_csv(path)
    X = []
    y = []
    for filepath, label, _ in df.values:  # Ignore magnification for now
        image = preprocess_image(filepath)
        if image is not None:
            X.append(image)
            y.append(1 if label == "malignant" else 0)  # Convert labels to binary (0: benign, 1: malignant)
    
    X = np.array(X)
    y = np.array(y)
    return X, y