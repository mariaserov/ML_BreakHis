import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Multi class prediction on the holdout set using model config 0: {'batch_size': 16, 'learning_rate': 0.0001, 'dropout_rate': 0.5, 'weight_decay': 0}")

# Load model
with open("densenet_hpo/models/hpo_model_0_multi.pickle", "rb") as openfile:
    model = pickle.load(openfile)

# Load holdout dataframe
val_df = pd.read_csv('../data/holdout.csv')

# Image preprocessing
image_size = 224
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col='tumor_subtype',
    target_size=(image_size, image_size),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predictions (probabilities)
prediction = model.predict(test_generator, steps=len(test_generator), verbose=1)
pred_classes = np.argmax(prediction, axis=1)

# True labels
true_classes = test_generator.classes
class_indices = test_generator.class_indices
# Ensure labels are in index order (e.g., 0, 1, 2, ...)
class_labels = [label for label, _ in sorted(class_indices.items(), key=lambda x: x[1])]

# Confusion matrix and classification report
cm = confusion_matrix(true_classes, pred_classes)
report = classification_report(true_classes, pred_classes, target_names=class_labels, output_dict=True)

# Binarize true labels for AUC
y_true_bin = label_binarize(true_classes, classes=np.arange(len(class_labels)))

# Metrics per class
results = []
for i, label in enumerate(class_labels):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    auc_score = roc_auc_score(y_true_bin[:, i], prediction[:, i])

    results.append({
        'precision': report[label]['precision'],
        'recall': report[label]['recall'],
        'specificity': specificity,
        'f1-score': report[label]['f1-score'],
        'support': int(report[label]['support']),
        'AUC': auc_score,
        'FPR': FPR,
        'FNR': FNR
    })

# Compile into DataFrame
results_df = pd.DataFrame(results, index=class_labels)

# Add summary rows
macro_auc = roc_auc_score(y_true_bin, prediction, average='macro', multi_class='ovr')

results_df.loc['accuracy'] = {
    'precision': '',
    'recall': '',
    'specificity': '',
    'f1-score': report['accuracy'],
    'support': sum(int(report[label]['support']) for label in class_labels),
    'AUC': '',
    'FPR': '',
    'FNR': ''
}

results_df.loc['auc_macro'] = {
    'precision': '',
    'recall': '',
    'specificity': '',
    'f1-score': '',
    'support': '',
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

# Save to CSV
results_df.to_csv("multi_0_full_metrics_summary.csv")
print("Saved multi_0_full_metrics_summary.csv")
