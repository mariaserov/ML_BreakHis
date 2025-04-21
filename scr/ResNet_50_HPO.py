import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import itertools

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2

# ==========================
# 1. Define HYPERPARAMETERS
# ==========================
n_epochs = 20


# Define the hyperparameter grid
hyperparams = {
    'learning_rate': [1e-3, 1e-4, 1e-5],  # Example learning rates
    'batch_size': [16, 32],            # Example batch sizes
    'optimizer': ['adam', 'sgd'],          # Adam and SGD optimizers
    'weight_decay': [0, 1e-4]        # Example weight decay values (L2 regularization)
}

# Generate all combinations of hyperparameters
keys, values = zip(*hyperparams.items())
param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]


# ==========================
# 2. Get PBS_ARRAY_INDEX
# ==========================
job_index = int(os.environ.get("PBS_ARRAY_INDEX", sys.argv[1])) - 1  # Adjust for 1-based index


config = param_grid[job_index]
print(f"Running job {job_index} with config: {config}")


# ==========================
# 4. Data loading
# ==========================
os.chdir("/rds/general/user/ft824/home/ML_BreakHis/scr")

train_df = pd.read_csv('../data/augmented_train_dataset.csv')
test_df = pd.read_csv('../data/new_test.csv')
train_df['label'] = train_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)
train_df['filepath'] = train_df['filepath'].str.replace(r"^\.\./", "../data/", regex=True)

image_size = 224
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='label',
    target_size=(image_size, image_size),
    batch_size=config['batch_size'],
    class_mode='categorical'
)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col='label',
    target_size=(image_size, image_size),
    batch_size=config['batch_size'],
    class_mode='categorical'
)

# ==========================
# 5. Model setup
# ==========================
layers_to_freeze = [
        "conv1_pad", "conv1_conv", "conv1_bn", "conv1_relu",
        "pool1_pad", "pool1_pool"
    ]

def create_model(weight_decay=0.0):
    base_model = ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    for layer in base_model.layers:
        if layer.name in layers_to_freeze:
            layer.trainable = False
        else:
            layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)
    output = layers.Dense(2, activation='softmax')(x)  # Adjust classes if needed

    return models.Model(inputs=base_model.input, outputs=output)

model = create_model(config['weight_decay'])

# ==========================
# 7. Compile model
# ==========================
if config['optimizer'] == 'adam':
    optimizer = Adam(learning_rate=config['learning_rate'])
elif config['optimizer'] == 'sgd':
    optimizer = SGD(learning_rate=config['learning_rate'])
else:
    raise ValueError("Unsupported optimizer")

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# 8. Train
# ==========================
os.makedirs("models_train4", exist_ok=True)
os.makedirs("logs_train4", exist_ok=True)
os.makedirs("plots_train4", exist_ok=True)

config_name = f"train4_job{job_index}_bs{config['batch_size']}_lr{config['learning_rate']}_{config['optimizer']}_wd{config['weight_decay']}"

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(f"models_train4/{config_name}.h5", save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=n_epochs,
    validation_data=test_generator,
    callbacks=callbacks
)

# ==========================
# 9. Save Plot & History
# ==========================

pd.DataFrame(history.history).to_csv(f"logs_train4/history_{config_name}.csv", index=False)

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(f"plots_train4/plot_{config_name}.pdf")
plt.close()