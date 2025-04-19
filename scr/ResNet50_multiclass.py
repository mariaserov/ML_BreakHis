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
num_blocks_to_unfreeze = idx - 1  # 0-5 (number of blocks to unfreeze)

# ================================
# 2. Block Indices for ResNet50
# ================================
block_start = [0, 7, 29, 81, 143]
block_end   = [6, 28, 80, 142, 177]
total_blocks = len(block_start)

# Validate input
if num_blocks_to_unfreeze > total_blocks:
    print(f"Error: max blocks to unfreeze is {total_blocks}")
    sys.exit(1)

# ================================
# 3. Data Preparation
# ================================
os.chdir("/rds/general/user/ft824/home/ML_BreakHis/scr")

train_df = pd.read_csv('../data/augmented_train_dataset.csv')
test_df = pd.read_csv('../data/new_test.csv')

train_df['filepath'] = train_df['filepath'].str.replace(r"^\.\./", "../data/", regex=True)

image_size = 224
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='tumor_subtype',
    target_size=(image_size, image_size),
    batch_size=32,
    class_mode='categorical'
)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col='tumor_subtype',
    target_size=(image_size, image_size),
    batch_size=16,
    class_mode='categorical'
)

# ================================
# 4. Build Model
# ================================
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(8, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# ================================
# 5. Unfreeze Last N Blocks
# ================================
if num_blocks_to_unfreeze == 0:
    print("Training classifier only. All base layers remain frozen.")
else:
    blocks_to_unfreeze = list(range(total_blocks - num_blocks_to_unfreeze, total_blocks))
    for i in blocks_to_unfreeze:
        for layer in model.layers[block_start[i]:block_end[i] + 1]:
            layer.trainable = True
        print(f"Unfreezing block {i} → layers {block_start[i]} to {block_end[i]}")

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ================================
# 6. Train the Model
# ================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(f"resnet50_last_{num_blocks_to_unfreeze}_blocks_multiclass.h5", save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=n_epochs,
    validation_data=test_generator,
    callbacks=callbacks
)

# ================================
# 7. Store and Plot Training and Validation Accuracy Over Epochs for All Models
# ================================
train_accuracies = []
val_accuracies = []

# Store current model's accuracy history
train_accuracies.append(history.history['accuracy'])
val_accuracies.append(history.history['val_accuracy'])

# ================================
# 8. Combine All Models' Results and Plot
# ================================
plt.figure(figsize=(14, 6))

# Plot Training Accuracy for All Models
plt.subplot(1, 2, 1)
for i in range(len(train_accuracies)):
    plt.plot(train_accuracies[i], label=f"Train Acc - {i} blocks unfrozen")
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Validation Accuracy for All Models
plt.subplot(1, 2, 2)
for i in range(len(val_accuracies)):
    plt.plot(val_accuracies[i], label=f"Val Acc - {i} blocks unfrozen")
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined plot as a PDF
plt.savefig(f"combined_accuracy_plots_all_models_multiclass.pdf")
plt.close()
