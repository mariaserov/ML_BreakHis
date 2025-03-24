import numpy as np
import cv2
import PIL
import pandas as pd
import os

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img


train_df = pd.read_csv("../data/toy_dataset.csv")  # CSV containing file paths & labels

save_dir = "../data/augmented_images"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist


# Create an image data generator with augmentation
datagen = ImageDataGenerator(
    rotation_range = 45,   # Rotate images up to 30 degrees
    horizontal_flip = True,   # Flip images horizontally
    rescale=1./255      #normalise pixel values
)

# Load images in batches
train_generator = datagen.flow_from_dataframe(
    dataframe = train_df,
    directory = "" ,  # Folder where images are stored
    x_col="filepath",  # Column containing image file paths
    y_col="label",  # Column with target labels (label or tumor_subtype)
    
    #target_size=(150, 150),  # resize image
    batch_size=32,  # 32 images per batch
    class_mode='binary',  #outcome ('categprical' for multiclass)

    save_to_dir=save_dir,      # Save augmented images
    save_prefix='aug',         # Prefix for saved images
    save_format='png'         # Format of saved images
)



# List to store new image metadata
augmented_data = []

# Process each image in the dataset
for index, row in train_df.iterrows():
    img_path = row["filepath"]
    
    # Load image
    try:
        image = load_img(img_path)  # Load original image
        image = img_to_array(image)  # Convert to array
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Generate one augmented image
        batch = next(datagen.flow(image, batch_size=1))
        new_filename = f"aug_{index}.png"
        new_filepath = os.path.join(save_dir, new_filename)

        # Save the augmented image
        array_to_img(batch[0]).save(new_filepath)

        # Append new metadata row
        augmented_data.append([new_filepath] + row.tolist()[1:])  # Keep original metadata

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Create new DataFrame with augmented data
augmented_df = pd.DataFrame(augmented_data, columns=train_df.columns)

# Save new CSV with augmented image paths & metadata
augmented_df.to_csv("../data/augmented_dataset.csv", index=False)

print(f"Augmented dataset saved to ../data/augmented_dataset.csv")