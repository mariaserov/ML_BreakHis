import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_df = pd.read_csv("../data/train.csv")  # CSV containing file paths & labels

save_dir = "data/augmented_images"
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
    directory = "data/train_images",  # Folder where images are stored
    x_col="filepath",  # Column containing image file paths
    y_col="label",  # Column with target labels (label or tumor_subtype)
    
    #target_size=(150, 150),  # resize image
    batch_size=32,  # 32 images per batch
    class_mode='binary',  #outcome ('categprical' for multiclass)

    save_to_dir=save_dir,      # Save augmented images
    save_prefix='aug',         # Prefix for saved images
    save_format='png'         # Format of saved images
)


