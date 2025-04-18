#####prediction

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

#Define generator for the unseen/test data
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_df = pd.csv("../data/augmented_train_dataset.csv")
#convert labels to string
train_df['label'] = train_df['label'].astype(str)
holdout_data = pd.csv("../data/new_holdout.csv")

image_size = 224  # for ResNet50

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',    # column with image file paths
    y_col='label',       # column with image labels
    target_size=(image_size, image_size),  # resizing to match ResNet50 input size
    batch_size=32,
    class_mode='categorical' # multi-class classification
)

unseen_generator = datagen.flow_from_dataframe(
    dataframe=holdout_data,           
    x_col='filepath',
    y_col=None,                 #labels for unseen data
    target_size=(224, 224),        # <- match input size to model
    class_mode=None,               # <- no class_mode
    batch_size=32,
    shuffle=False                  # <- don't shuffle, to keep predictions in order
)


from tensorflow.keras.models import load_model

#load trained resnet model
model = load_model("resnet50_best_model.h5")
predictions = model.predict(unseen_generator)

### get labels
predicted_classes = np.argmax(predictions, axis=1)  # for categorical output

class_indices = train_generator.class_indices
label_map = {v: k for k, v in class_indices.items()}

predicted_labels = [label_map[i] for i in predicted_classes]
