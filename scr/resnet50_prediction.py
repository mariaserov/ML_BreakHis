#####prediction

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

#set working directory

# List of model filenames to evaluate
model_files = ["/rds/general/user/ft824/home/ML_BreakHis/scr/resnet50_output/resnet50_HPO_output/models_multiclass/multiclass_job20_bs32_lr1e-05_adam_wd0.h5"
    #"resnet50_last_5_blocks_multiclass.h5",
    #"resnet50_last_4_blocks_multiclass.h5",
    #"resnet50_last_3_blocks_multiclass.h5",
    #"resnet50_last_2_blocks_multiclass.h5",
    #"resnet50_last_1_blocks_multiclass.h5",
    #"resnet50_last_0_blocks_multiclass.h5",
    # Add more models here if needed
]

#load train and unseen generator

#Define generator for the unseen/test data
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_df = pd.read_csv("../data/augmented_train_dataset.csv")
train_df['filepath'] = train_df['filepath'].str.replace(r"^\.\./", "../data/", regex=True)

#convert labels to string
#train_df['label'] = train_df['label'].astype(str)

holdout_data = pd.read_csv("../data/new_holdout.csv")

image_size = 224  # for ResNet50

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',    # column with image file paths
    y_col='tumor_subtype',       # column with image labels
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


# Class index mappings
class_indices = train_generator.class_indices
index_to_label = {v: k for k, v in class_indices.items()}
labels = list(class_indices.keys())

# True labels from generator
true_classes = holdout_data['tumor_subtype']
# True labels from holdout dataframe (already strings)
true_labels = holdout_data['tumor_subtype'].tolist()


for model_path in model_files:
    print(f"\n--- Evaluating {model_path} ---")
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Load model and predict
    model = load_model(model_path)
    predictions = model.predict(unseen_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = [index_to_label[i] for i in predicted_classes]

    # Classification report
    report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Add accuracy manually
    accuracy = accuracy_score(true_labels, predicted_labels)
    report_df.loc['accuracy'] = [accuracy, None, None, None]

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    fpr_list = []
    fnr_list = []

    for i in range(len(cm)):
        TP = cm[i, i]
        FN = sum(cm[i, :]) - TP
        FP = sum(cm[:, i]) - TP
        TN = cm.sum() - (TP + FP + FN)
        fpr = FP / (FP + TN) if (FP + TN) != 0 else 0
        fnr = FN / (FN + TP) if (FN + TP) != 0 else 0
        fpr_list.append(fpr)
        fnr_list.append(fnr)

    report_df['FPR'] = fpr_list + [None] * (len(report_df) - len(fpr_list))
    report_df['FNR'] = fnr_list + [None] * (len(report_df) - len(fnr_list))

    # Save metrics and confusion matrix
    metrics_filename = f"{model_name}_metrics.csv"
    cm_filename = f"{model_name}_confusion_matrix.csv"

    report_df.to_csv(metrics_filename)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_filename)

    print(f"Saved: {metrics_filename}, {cm_filename}")

