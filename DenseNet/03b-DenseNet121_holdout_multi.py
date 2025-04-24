import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Multi class prediction on the holdout set using model config 0")

with (open(f"densenet_hpo/models_pickle.bk/hpo_model_0_multi.pickle", "rb")) as openfile:
    model  = pickle.load(openfile)

val_df = pd.read_csv('../data/holdout.csv')

image_size = 224
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col='tumor_subtype',
    target_size=(image_size, image_size),
    batch_size=32,
    class_mode='categorical'
)

# Make predictions
prediction = model.predict(test_generator, steps=len(test_generator), verbose=1)
pred_classes = np.argmax(prediction, axis=1)

# True labels
true_classes = test_generator.classes
print(f'True classes: {true_classes}')
class_indices = test_generator.class_indices
class_labels = list(class_indices.keys())

# Confusion matrix
cm = confusion_matrix(true_classes, pred_classes)
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

# Classification report
report = classification_report(true_classes, pred_classes, target_names=class_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# One-vs-rest binarization
y_true_bin = label_binarize(true_classes, classes=np.arange(len(class_labels)))

results = []
for i, label in enumerate(class_labels):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    # Rates
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # AUC for this class (One-vs-Rest)
    auc_score = roc_auc_score(y_true_bin[:, i], prediction[:, i])

    results.append({
        'precision': report[label]['precision'],
        'recall': report[label]['recall'],
        'specificity': specificity,
        'f1-score': report[label]['f1-score'],
        'support': report[label]['support'],
        'AUC': auc_score,
        'FPR': FPR,
        'FNR': FNR
    })

# Convert to DataFrame
results_df = pd.DataFrame(results, index=class_labels)

# Calculate macro AUC
macro_auc = roc_auc_score(y_true_bin, prediction, average='macro', multi_class='ovr')

# Add averages
results_df.loc['accuracy'] = {
    'precision': '',
    'recall': '',
    'specificity': '',
    'f1-score': report['accuracy'],
    'support': sum(report[label]['support'] for label in class_labels),
    'AUC': macro_auc,
    'FPR': '',
    'FNR': ''
}

results_df.loc['macro avg'] = {
    'precision': report['macro avg']['precision'],
    'recall': report['macro avg']['recall'],
    'specificity': '',
    'f1-score': report['macro avg']['f1-score'],
    'support': report['macro avg']['support'],
    'AUC': '',
    'FPR': '',
    'FNR': ''
}

results_df.loc['weighted avg'] = {
    'precision': report['weighted avg']['precision'],
    'recall': report['weighted avg']['recall'],
    'specificity': '',
    'f1-score': report['weighted avg']['f1-score'],
    'support': report['weighted avg']['support'],
    'AUC': '',
    'FPR': '',
    'FNR': ''
}

# Save
results_df.to_csv("multi_0_full_metrics_summary.csv")
print("Saved multi_0_full_metrics_summary.csv")
