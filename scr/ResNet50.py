import os
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
import keras
import matplotlib.pyplot as plt


os.chdir("/rds/general/user/ft824/home/ML_BreakHis/scr")

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)        # More efficient than Flatten for ResNet
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

# Define the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Load your CSV files
train_df = pd.read_csv('../data/augmented_train_dataset.csv')
test_df = pd.read_csv('../data/new_test.csv')

#convert labels to string
train_df['label'] = train_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)


image_size = 224  # for ResNet50


# Define the ImageDataGenerator with preprocessing
datagen = ImageDataGenerator(preprocessing_function=
preprocess_input)


# Training generator
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',    # column with image file paths
    y_col='label',       # column with image labels
    target_size=(image_size, image_size),  # resizing to match ResNet50 input size
    batch_size=32,
    class_mode='categorical' # multi-class classification
)

# Test generator
test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',    # column with image file paths
    y_col='label',       # column with image labels
    target_size=(image_size, image_size),  # resizing to match ResNet50 input size
    batch_size=16,
    class_mode='categorical'
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('resnet50_best_model.h5', save_best_only=True)
]

history = model.fit(
        train_generator,
        epochs = 50,
        validation_data=test_generator,
        batch_size =32,
        callbacks=callbacks
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# Save as PDF
plt.savefig("training_history.pdf", format='pdf')  # You can specify a full path too

plt.show()
