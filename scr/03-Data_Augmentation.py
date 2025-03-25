import numpy as np
import cv2
import PIL
import pandas as pd
import os

import tensorflow as tf

data_dir = "../../.cache/kagglehub/datasets/ambarish/breakhis/versions/4"
metadata = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".png"):
            # Extract label from the folder structure
            label = "malignant" if "malignant" in root else "benign"
            
            # Extract magnification 
            
            magnification = None
            for part in root.split(os.sep):
                if part.endswith("X") and part[:-1].isdigit(): 
                    # magnification = part # turn to int
                    magnification = int(part[:-1])
                    break
            
            # Extract tumor subtype 
            tumor_subtype = None
            for part in root.split(os.sep):
                if part in ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma",  # Benign subtypes
                           "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]:                # Malignant subtypes
                    tumor_subtype = part
                    break
            
            # Append filepath, label, magnification, and tumor subtype to metadata
            metadata.append((os.path.join(root, file), label, magnification, tumor_subtype))


# Convert to DataFrame
df = pd.DataFrame(metadata, columns=["filepath", "label", "magnification", "tumor_subtype"])

df.to_csv("../data/metadata.csv")

# Debugging: Check the shape and first few rows of the DataFrame
print(f"DataFrame shape: {df.shape}")
print(df.head())

def augment_and_normalize_images(df, output_folder, target_size=(150, 150)):
    # Create an image data generator with augmentation : rotation and flip
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90, horizontal_flip=True)

    for idx, row in df.iterrows():
        filepath, label = row["filepath"], row["label"]
        
        image = cv2.imread(filepath)  # Read image
        if image is None:
            print(f"Could not load: {filepath}")
            continue
        
        image = cv2.resize(image, target_size)  # Resize
        # Convert BGR to RGB because OpenCV loads in BGR and TensorFlow needs RGB
        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        x = x.astype('float32') / 255.0  # Normalize
        x = x.reshape((1,) + x.shape)  # Reshape for ImageDataGenerator
        
        image = image.astype('float32') / 255.0  # Normalize
        save_path = os.path.join(output_folder, f"original_{idx}.png")

        # save normalized original
        normalized_image = (x[0] * 255).astype(np.uint8)  # Convert back to uint8 for saving
        cv2.imwrite(save_path, cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR))
        
        if label == "malignant":
            continue
        
        # generate one augmented image per original
        for i, batch in enumerate(datagen.flow(x, batch_size=1, save_to_dir=output_folder, save_prefix=f'augmented_{idx}', save_format='png')):
            if i == 0:
                break