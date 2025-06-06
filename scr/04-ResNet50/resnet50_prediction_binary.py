import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,roc_auc_score




#Define generator for the unseen/test data
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_df = pd.read_csv("../data/augmented_train_dataset.csv")
train_df['filepath'] = train_df['filepath'].str.replace(r"^\.\./", "../data/", regex=True)

#convert labels to string
train_df['label'] = train_df['label'].astype(str)

holdout_data = pd.read_csv("../data/new_holdout.csv")

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


#load trained resnet model
model = load_model("/rds/general/user/ft824/home/ML_BreakHis/scr/resnet50_output/resnet50_HPO_output/models_train4/train4_job9_bs16_lr0.0001_adam_wd0.0001.h5")
predictions = model.predict(unseen_generator)

### get labels
predicted_classes = np.argmax(predictions, axis=1)  # for categorical output

class_indices = train_generator.class_indices
label_map = {0: "benign", 1: "malignant"}

predicted_labels = [label_map[i] for i in predicted_classes]


###confusion matrix
true_labels = holdout_data['label'].astype(int).values

cm = confusion_matrix(true_labels, predicted_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("binary_confusion_matrix.pdf")

plt.show()

tn, fp, fn, tp = cm.ravel()


acc = round(accuracy_score(true_labels, predicted_classes), 3)
f1 = round(f1_score(true_labels, predicted_classes),3)
precision = round(precision_score(true_labels, predicted_classes),3)
recall = round(recall_score(true_labels, predicted_classes),3)
specificity = round(tn / (tn + fp), 3)

predicted_probs = predictions[:, 1]
# Ensure you have predicted probabilities
auc = round(roc_auc_score(true_labels, predicted_probs), 3)

# Save metrics to CSV
metrics_dict = {
    "Accuracy": [acc],
    "F1 Score": [f1],
    "Precision": [precision],
    "Recall": [recall],
    "Specificity": [specificity],
    "AUC": [auc]
}

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv("binary_model_metrics.csv", index=False)
