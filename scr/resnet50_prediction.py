#####prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Define generator for the unseen/test data
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_df = pd.read_csv("../data/augmented_train_dataset.csv")
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


from tensorflow.keras.models import load_model

#load trained resnet model
model = load_model("resnet50_best_model.h5")
predictions = model.predict(unseen_generator)

### get labels
predicted_classes = np.argmax(predictions, axis=1)  # for categorical output

class_indices = train_generator.class_indices
label_map = {0: "benign", 1: "malignant"}

predicted_labels = [label_map[i] for i in predicted_classes]

print(train_generator.class_indices)


# Assuming you have a list of file paths and predicted labels
results_df = pd.DataFrame({
    'filename': unseen_generator.filepath,
    'prediction': predicted_labels
})

results_df.to_csv("predictions.csv", index=False)

##evaluate accuracy

loss, accuracy = model.evaluate(unseen_generator)
with open("evaluation.txt", "w") as f:
    f.write(f"Loss: {loss}\n")
    f.write(f"Accuracy: {accuracy}\n")


# Example: saving accuracy and loss from training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(predictions.predictions['accuracy'], label='train_accuracy')
plt.plot(predictions.prediction['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title("Model Accuracy")

plt.subplot(1, 2, 2)
plt.plot(predictions.predictions['loss'], label='train_loss')
plt.plot(predictions.predictions['val_loss'], label='val_loss')
plt.legend()
plt.title("Model Loss")


plt.tight_layout()

# Save as PDF
plt.savefig("training_history.pdf", format='pdf')  # You can specify a full path too

plt.show()

###confusion matrix
true_labels = holdout_data['label'].astype(int).values

cm = confusion_matrix(true_labels, predicted_classes)

# If you're using numeric labels (0 and 1):
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])

# If using string labels, just make sure labels match:
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "malignant"])

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

