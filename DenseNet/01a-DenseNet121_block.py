import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)

# ================================
# 1. Parse Block Index
# ================================
n_epochs = 12
idx = int(os.environ['PBS_ARRAY_INDEX'])
num_blocks_to_unfreeze = idx - 1  # 0 = only top classifier trainable

# ================================
# 2. Load Data
# ================================
os.chdir("/rds/general/user/js4124/home/ML_BreakHis/DenseNet")

train_df = pd.read_csv('../data/augmented_train_dataset.csv')
test_df = pd.read_csv('../data/new_test.csv')

train_df['label'] = train_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)
train_df['filepath'] = train_df['filepath'].str.replace(r"^\../", "../data/", regex=True)

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
# 3. Define Model
# ================================
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

print(model)

# ================================
# 4. Define Layer Ranges for 7 Blocks
# ================================
block_ranges = {
    0: (0, 6),      # Conv + Pooling
    1: (7, 86),     # Dense Block 1 + Transition 1
    2: (87, 186),   # Dense Block 2 + Transition 2
    3: (187, 346),  # Dense Block 3 + Transition 3
    4: (347, 482),  # Dense Block 4
    5: (483, 483),  # Global Average Pooling
    6: (484, 486),  # Classifier
}

total_blocks = len(block_ranges)
if num_blocks_to_unfreeze > total_blocks:
    print(f"Error: max blocks to unfreeze is {total_blocks}")
    sys.exit(1)

# ================================
# 5. Freeze Layers
# ================================
for layer in model.layers:
    layer.trainable = False

if num_blocks_to_unfreeze == 0:
    print("Training classifier only. All base layers remain frozen.")
else:
    blocks_to_unfreeze = list(range(total_blocks - num_blocks_to_unfreeze, total_blocks))
    for i in blocks_to_unfreeze:
        start, end = block_ranges[i]
        for layer in model.layers[start:end + 1]:
            layer.trainable = True
        print(f"Unfreezing block {i} → layers {start} to {end}")

# ================================
# 6. Compile and Train
# ================================
model.compile(
    # optimizer=Adam(learning_rate=1),
    # loss='categorical_crossentropy',
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(f"densenet121_last_{num_blocks_to_unfreeze}_blocks.h5", save_best_only=True)
]

# Time the training
start_time = time.time()

history = model.fit(
    train_generator,
    epochs=n_epochs,
    validation_data=test_generator,
    callbacks=callbacks
)

with open(f'densenet_models/model_{idx}.pickle', 'wb') as handle:
    pickle.dump(model, handle)

with open(f'densenet_history/history_{idx}.pickle', 'wb') as handle:
    pickle.dump(history, handle)

end_time = time.time()
print(f"Total runtime: {(end_time - start_time) / 60:.2f} minutes")

