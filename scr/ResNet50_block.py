import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ================================
# 1. Parse Block Index from Argument
# ================================
n_epochs = 20

# Array job 

idx = int(os.environ['PBS_ARRAY_INDEX'])
num_blocks_to_unfreeze = idx-1 # 0-5


# ================================
# 2. Block Indices
# ================================
block_start = [0, 7, 29, 81, 143]
block_end   = [6, 28, 80, 142, 177]
total_blocks = len(block_start)

# Validate input
if num_blocks_to_unfreeze > total_blocks:
    print(f"Error: max blocks to unfreeze is {total_blocks}")
    sys.exit(1)

# ================================
# 3. Data
# ================================
os.chdir("/rds/general/user/ft824/home/ML_BreakHis/scr")

train_df = pd.read_csv('../data/augmented_train_dataset.csv')
test_df = pd.read_csv('../data/new_test.csv')
test_df['label'] = test_df['label'].astype(str)

image_size = 224
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='label',
    target_size=(image_size, image_size),
    batch_size=32,
    class_mode='categorical'
)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col='label',
    target_size=(image_size, image_size),
    batch_size=16,
    class_mode='categorical'
)

# ================================
# 4. Model
# ================================
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# ================================
# 5. Unfreeze Last N Blocks
# ================================
blocks_to_unfreeze = list(range(total_blocks - num_blocks_to_unfreeze, total_blocks))

for i in blocks_to_unfreeze:
    for layer in model.layers[block_start[i]:block_end[i] + 1]:
        layer.trainable = True
    print(f"Unfreezing block {i} → layers {block_start[i]} to {block_end[i]}")

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ================================
# 6. Train
# ================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(f"resnet50_last_{num_blocks_to_unfreeze}_blocks.h5", save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=n_epochs,
    validation_data=test_generator,
    callbacks=callbacks
)

# ================================
# 7. Plot
# ================================
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
plt.savefig(f"training_history_last_{num_blocks_to_unfreeze}_blocks.pdf")
plt.close()
